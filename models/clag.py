import sys
sys.path.append('..')
import fewshot_re_kit
import torch
from torch import nn
from torch.nn import functional as F

class AttentionModule(nn.Module):
    ''' Scaled Dot-Product Attention '''
    def __init__(self,  hid = 768):
        super().__init__()
        self.hidden_size = hid
        
 
    
    def __euclid_dist__(self, x, y, dim):
        return -(torch.pow(x - y, 2)).sum(dim)

    def __batch_euclid_dist__(self, S, Q):
        return self.__euclid_dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)
    
    def forward(self, support, query, N, K, total_Q):
        support_ = support.view(-1, N * K, self.hidden_size * 2)
        dist_sq = self.__batch_euclid_dist__(support_, query).view(-1, total_Q, N, K)
        query_guided_weights = dist_sq.mean(1).tanh().softmax(-1).unsqueeze(-1)
        support = (support * query_guided_weights).sum(2)
        return support, query_guided_weights

class CLAG(fewshot_re_kit.framework.FewShotREModel):

    def __init__(self, sentence_encoder, dot=False, relation_encoder=None, N=5, Q=1, head=2, hid=1536, d_k=1536,
                 struct_hid=32, attn_hid=32, dropout=0.):
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
        self.fc = nn.Linear(d_k, d_k)
        self.drop = nn.Dropout(dropout)
        self.dot = dot
        self.attn = AttentionModule()
        self.relation_encoder = relation_encoder
        self.hidden_size = 768

        self.alpha1 = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.alpha2 = nn.Parameter(torch.tensor(0.5), requires_grad=True)

    def __pred__(self, w, x, dim):
        if self.dot:
            return (w * x).sum(dim)
        else:
            return -(torch.pow(w - x, 2)).sum(dim)
    def pred(self, S, Q):
        return self.__pred__(S.unsqueeze(1), Q.unsqueeze(2), 3)

    def forward(self, support, query, rel_txt, N, K, total_Q):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances in the query set
        '''
        rel_gol, rel_loc = self.sentence_encoder(rel_txt, cat=False)
        rel_loc = torch.mean(rel_loc[:, 1:, :], 1)  # [B*N, D]
        rel_rep = torch.cat((rel_gol, rel_loc), -1).view(-1, N, 1, rel_gol.shape[1] * 2)

        support_h, support_t, s_loc = self.sentence_encoder(support)
        query_h, query_t, q_loc = self.sentence_encoder(query)

        support_emb = torch.cat((support_h, support_t), -1)
        query_emb = torch.cat((query_h, query_t), -1)
        support = self.drop(support_emb)
        query = self.drop(query_emb)

       
        support = support.view(-1, N, K, self.hidden_size*2)
        query = query.view(-1, total_Q, self.hidden_size*2)
        B = support.size(0)
        ins_loss = self.InsContrastiveLoss(support, temp = 0.5)
#         rel_loss = self.RelContrastiveLoss(support, rel_rep, temp = 0.5)
        rel_loss = 0

        support, _ = self.attn(support, query, N, K, total_Q)
        
        support = support.view(B, N, 1, 2*self.hidden_size)
        final_support = self.alpha1 * support  + self.alpha2 * rel_rep
        
        final_support = final_support.view(B, N, -1)
        final_query = query.view(B, total_Q, -1)

        logits = self.pred(final_support, final_query)


        minn, _ = logits.min(-1)
        logits = torch.cat([logits, minn.unsqueeze(2) - 1], 2)
        _, pred = torch.max(logits.view(-1, N + 1), 1)
        return logits, pred, ins_loss, rel_loss

    def generate_weight(self, support, query, rel_txt, N, K, total_Q):


        rel_gol, rel_loc = self.sentence_encoder(rel_txt, cat=False)
        rel_loc = torch.mean(rel_loc[:, 1:, :], 1)  # [B*N, D]
        rel_rep = torch.cat((rel_gol, rel_loc), -1).view(-1, N, 1, rel_gol.shape[1] * 2)

        support_h, support_t, s_loc = self.sentence_encoder(support)
        query_h, query_t, q_loc = self.sentence_encoder(query)

        support_emb = torch.cat((support_h, support_t), -1)
        query_emb = torch.cat((query_h, query_t), -1)
        support = self.drop(support_emb)
        query = self.drop(query_emb)

        support = support.view(-1, N, K, self.hidden_size*2)
        query = query.view(-1, total_Q, self.hidden_size*2)
        B = support.size(0)
        temp_support = torch.cat([support_h, support_t], -1).view(B, N * K, -1)
        
        support, _ = self.attn(support, query, N, K, total_Q)
        
     
        rel_rep = rel_rep.view(B,N,-1)
        final_support = self.alpha1 * support + self.alpha2 * rel_rep
       
        
        final_support = final_support.view(B, N, -1)
        final_query = query.view(B, total_Q, -1)
        
        return final_support.detach(), temp_support.detach(), final_query.detach()

    def generate_visualization_emb(self, support, query, rel_txt, N, K, total_Q):
        support_h, support_t, s_loc = self.sentence_encoder(support)
        return torch.cat((support_h, support_t), -1)


    
    def RelContrastiveLoss(self, support, rel_rep, temp):
        B, N, K, D = support.shape
        x = support.view(B, N * K, D)
        d = rel_rep.view(B, N, D)
        loss = 0
        for i in range(B):
            xi = F.normalize(x[i], dim=1)
            di = F.normalize(d[i], dim=1)
            sim = xi @ di.transpose(0, 1)
            sim = torch.exp(sim / temp)
            sim = torch.softmax(sim,dim=-1)
            mask = torch.zeros((N * K, N), dtype=bool)
            for i in range(N):
                mask[i * K:(i + 1) * K, i] = True
            pos = sim[mask]
            neg, _ = torch.min(sim, dim=-1)
           
            loss += -torch.log(pos / (pos + neg)).sum()

        
        return loss / (B * N * K)


    def InsContrastiveLoss(self, support, temp):
        B, N, K, D = support.shape
        x = support.view(B, N * K, D)
        loss = 0
        for i in range(B):
            xi = F.normalize(x[i], dim=1)
            sim = xi @ xi.transpose(0, 1)
            sim = torch.exp(sim / temp)
            sim = torch.softmax(sim,dim=-1)
            mask = torch.as_tensor([[i for _ in range(K)] for i in range(N)]).reshape(-1).to(x.device)
            mask = torch.eq(mask, mask.unsqueeze(1)).float() - torch.eye(mask.shape[0]).to(x.device)
            if K == 1: mask = torch.eye(mask.shape[0]).to(x.device)
            pos = (sim * mask).sum(-1)
            neg, _ = torch.min(sim, dim=-1)
            
    
            loss += -torch.log(pos / (pos + neg)).sum()

           
        return loss / (B * N * K)
