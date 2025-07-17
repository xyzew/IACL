import os

import sys
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd

import torch
from torch import autograd, optim, nn
from models.visualization import plot_tsne

from transformers import AdamW, get_linear_schedule_with_warmup
import copy
from torch.nn import functional as F
from tqdm import tqdm
import json

class New_CLSF(nn.Module):

    def __init__(self, N, hidden_size=1536):
        nn.Module.__init__(self)
        self.linear = nn.Linear(hidden_size,N, bias = False)
        self.hidden_size = hidden_size
        # self.cls_weight = torch.zeros([1])

    def forward(self, x, use_dot):
        if use_dot:
            self.linear.weight.data = self.linear.weight.data.squeeze()
            output = self.linear(x)
            print(output.shape)
        else:
            x = x.transpose(0,1)
            #output = -(torch.pow(self.linear.weight - x, 2)).sum(2)
            output = (self.linear.weight * x).sum(2)
        return output

    def set_weight(self,w):
        self.linear.weight.data = w.unsqueeze(0)

def warmup_linear(global_step, warmup_step):
    if global_step < warmup_step:
        return global_step / warmup_step
    else:
        return 1.0

class FewShotREModel(nn.Module):
    def __init__(self, my_sentence_encoder):
        '''
        sentence_encoder: Sentence encoder
        
        You need to set self.cost as your own loss function.
        '''
        nn.Module.__init__(self)
        self.sentence_encoder = nn.DataParallel(my_sentence_encoder)
        self.cost = nn.CrossEntropyLoss()
    
    def forward(self, support, query, rel_text, N, K, Q):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        return: logits, pred
        '''
        raise NotImplementedError

    def loss(self, logits, label):
        '''
        logits: Logits with the size (..., class_num)
        label: Label with whatever size. 
        return: [Loss] (A single value)
        '''
        N = logits.size(-1)
        return self.cost(logits.view(-1, N), label.view(-1))

    def accuracy(self, pred, label):
        '''
        pred: Prediction results with whatever size
        label: Label with whatever size
        return: [Accuracy] (A single value)
        '''
        return torch.mean((pred.view(-1) == label.view(-1)).type(torch.FloatTensor))

class FewShotREFramework:

    def __init__(self, train_data_loader, val_data_loader, test_data_loader, adv_data_loader=None, adv=False, d=None):
        '''
        train_data_loader: DataLoader for training.
        val_data_loader: DataLoader for validating.
        test_data_loader: DataLoader for testing.
        '''
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.adv_data_loader = adv_data_loader
        self.adv = adv
        if adv:
            self.adv_cost = nn.CrossEntropyLoss()
            self.d = d
            self.d.cuda()
    
    def __load_model__(self, ckpt):
        '''
        ckpt: Path of the checkpoint
        return: Checkpoint dict
        '''
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt)
            print("Successfully loaded checkpoint '%s'" % ckpt)
            return checkpoint
        else:
            raise Exception("No checkpoint found at '%s'" % ckpt)
    
    def item(self, x):
        '''
        PyTorch before and after 0.4
        '''
        torch_version = torch.__version__.split('.')
        if int(torch_version[0]) == 0 and int(torch_version[1]) < 4:
            return x[0]
        else:
            return x.item()

    def train(self,
              model,
              model_name,
              B, N_for_train, N_for_eval, K, Q,
              na_rate=0,
              learning_rate=1e-1,
              lr_step_size=20000,
              weight_decay=1e-5,
              train_iter=30000,
              val_iter=1000,
              val_step=2000,
              test_iter=3000,
              load_ckpt=None,
              save_ckpt=None,
              pytorch_optim=optim.SGD,
              bert_optim=False,
              warmup=True,
              warmup_step=300,
              grad_iter=1,
              fp16=False,
              pair=False,
              adv_dis_lr=1e-1,
              adv_enc_lr=1e-1,
              use_sgd_for_bert=False):
        '''
        model: a FewShotREModel instance
        model_name: Name of the model
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        ckpt_dir: Directory of checkpoints
        learning_rate: Initial learning rate
        lr_step_size: Decay learning rate every lr_step_size steps
        weight_decay: Rate of decaying weight
        train_iter: Num of iterations of training
        val_iter: Num of iterations of validating
        val_step: Validate every val_step steps
        test_iter: Num of iterations of testing
        '''
        print("Start training...")
        loss_record = []
        accuracy_record = []
        # Init
        if bert_optim:
            print('Use bert optim!')
            parameters_to_optimize = list(model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            parameters_to_optimize = [
                {'params': [p for n, p in parameters_to_optimize 
                    if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in parameters_to_optimize
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            if use_sgd_for_bert:
                optimizer = torch.optim.SGD(parameters_to_optimize, lr=learning_rate)
            else:
                optimizer = AdamW(parameters_to_optimize, lr=learning_rate, correct_bias=False)
            if self.adv:
                optimizer_encoder = AdamW(parameters_to_optimize, lr=1e-5, correct_bias=False)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=train_iter) 
        else:
            optimizer = pytorch_optim(model.parameters(),
                    learning_rate, weight_decay=weight_decay)
            if self.adv:
                optimizer_encoder = pytorch_optim(model.parameters(), lr=adv_enc_lr)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size)

        if self.adv:
            optimizer_dis = pytorch_optim(self.d.parameters(), lr=adv_dis_lr)

        if load_ckpt:
            state_dict = self.__load_model__(load_ckpt)['state_dict']
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    print('ignore {}'.format(name))
                    continue
                print('load {} from {}'.format(name, load_ckpt))
                own_state[name].copy_(param)
            start_iter = 0
        else:
            start_iter = 0

        if fp16:
            from apex import amp
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

        model.train()
        if self.adv:
            self.d.train()

        # Training
        best_acc = 0
        iter_loss = 0.0
        iter_loss_dis = 0.0
        iter_right = 0.0
        iter_right_dis = 0.0
        iter_sample = 0.0
        for it in range(start_iter, start_iter + train_iter):
            if pair:
                batch, label = next(self.train_data_loader)
                if torch.cuda.is_available():
                    for k in batch:
                        batch[k] = batch[k].cuda()
                    label = label.cuda()
                logits, pred = model(batch, N_for_train, K, 
                        Q * N_for_train + na_rate * Q)
            else:
                support, query, label, rel_text = next(self.train_data_loader)
                if torch.cuda.is_available():
                    for k in support:
                        support[k] = support[k].cuda()
                    for k in query:
                        query[k] = query[k].cuda()
                    
                    for k in rel_text:
                        rel_text[k] = rel_text[k].cuda()
                    
                    label = label.cuda()

                logits, pred, ins_loss, rel_loss = model(support, query, rel_text, 
                        N_for_train, K, Q * N_for_train + na_rate * Q)
            loss = model.loss(logits, label) / float(grad_iter)
            loss = loss +  0.2 * ins_loss + 0.2 * rel_loss
            right = model.accuracy(pred, label)
            if fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                # torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 10)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            
            if it % grad_iter == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # Adv part
            if self.adv:
                support_adv = next(self.adv_data_loader)
                if torch.cuda.is_available():
                    for k in support_adv:
                        support_adv[k] = support_adv[k].cuda()

                features_ori = model.sentence_encoder(support)
                features_adv = model.sentence_encoder(support_adv)
                features = torch.cat([features_ori, features_adv], 0) 
                total = features.size(0)
                dis_labels = torch.cat([torch.zeros((total // 2)).long().cuda(),
                    torch.ones((total // 2)).long().cuda()], 0)
                dis_logits = self.d(features)
                loss_dis = self.adv_cost(dis_logits, dis_labels)
                _, pred = dis_logits.max(-1)
                right_dis = float((pred == dis_labels).long().sum()) / float(total)
                
                loss_dis.backward(retain_graph=True)
                optimizer_dis.step()
                optimizer_dis.zero_grad()
                optimizer_encoder.zero_grad()

                loss_encoder = self.adv_cost(dis_logits, 1 - dis_labels)
    
                loss_encoder.backward(retain_graph=True)
                optimizer_encoder.step()
                optimizer_dis.zero_grad()
                optimizer_encoder.zero_grad()

                iter_loss_dis += self.item(loss_dis.data)
                iter_right_dis += right_dis

            iter_loss += self.item(loss.data)
            iter_right += self.item(right.data)
            iter_sample += 1
            if self.adv:
                sys.stdout.write('step: {0:4} | loss: {1:2.6f}, accuracy: {2:3.2f}%, dis_loss: {3:2.6f}, dis_acc: {4:2.6f}'
                    .format(it + 1, iter_loss / iter_sample, 
                        100 * iter_right / iter_sample,
                        iter_loss_dis / iter_sample,
                        100 * iter_right_dis / iter_sample) + '\r')
            else:
                sys.stdout.write('step: {0:4} | loss: {1:2.6f}, accuracy: {2:3.2f}%'.format(it + 1, iter_loss / iter_sample, 100 * iter_right / iter_sample) + '\r')
                
                
                
            sys.stdout.flush()

            if (it + 1) % val_step == 0:
                acc = self.eval(model, B, N_for_eval, K, Q, val_iter, 
                        na_rate=na_rate, pair=pair)
                loss_record.append(float(iter_loss / iter_sample))
                accuracy_record.append(float(100 * iter_right / iter_sample))
                model.train()
                if acc > best_acc:
                    print('Best checkpoint')
                    torch.save({'state_dict': model.state_dict()}, save_ckpt)
                    best_acc = acc
                iter_loss = 0.
                iter_loss_dis = 0.
                iter_right = 0.
                iter_right_dis = 0.
                iter_sample = 0.
                
        print("\n####################\n")
        print("Finish training " + model_name)
        print(loss_record)
        print(accuracy_record)
        
        
        
        
                

    def eval(self,
            model,
            B, N, K, Q,
            eval_iter,
            na_rate=0,
            pair=False,
            ckpt=None): 
        '''
        model: a FewShotREModel instance
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        eval_iter: Num of iterations
        ckpt: Checkpoint path. Set as None if using current model parameters.
        return: Accuracy
        '''
        print("")
        
        model.eval()
        if ckpt is None:
            print("Use val dataset")
            eval_dataset = self.val_data_loader
        else:
            print("Use test dataset")
            if ckpt != 'none':
                state_dict = self.__load_model__(ckpt)['state_dict']
                own_state = model.state_dict()
                for name, param in state_dict.items():
                    if name not in own_state:
                        continue
                    own_state[name].copy_(param)
            eval_dataset = self.val_data_loader
        
        iter_right = 0.0
        iter_sample = 0.0
        with torch.no_grad():
            for it in range(eval_iter):
                if pair:
                    batch, label = next(eval_dataset)
                    if torch.cuda.is_available():
                        for k in batch:
                            batch[k] = batch[k].cuda()

                        label = label.cuda()
                    logits, pred = model(batch, N, K, Q * N + Q * na_rate)
                else:
                    support, query, label, rel_text = next(eval_dataset)
                    if torch.cuda.is_available():
                        for k in support:
                            support[k] = support[k].cuda()
                        for k in query:
                            query[k] = query[k].cuda()
                        
                        for k in rel_text:
                            rel_text[k] = rel_text[k].cuda()    
                        
                        label = label.cuda()
                    logits, pred, ins_loss, rel_loss = model(support, query, rel_text, N, K, Q * N + Q * na_rate)
                
                right = model.accuracy(pred, label)
                iter_right += self.item(right.data)
                iter_sample += 1

                sys.stdout.write('[EVAL] step: {0:4} | accuracy: {1:3.2f}%'.format(it + 1, 100 * iter_right / iter_sample) + '\r')
                sys.stdout.flush()
            print("")
        return iter_right / iter_sample

    def test(self,
                model,
                B, N, K, Q,
                eval_iter=10000,
                na_rate=0,
                pair=False,
                ckpt=None,
                test_output=None,
                test_lr=0.):
            '''
            model: a FewShotREModel instance
            B: Batch size
            N: Num of classes for each batch
            K: Num of instances for each class in the support set
            Q: Num of instances for each class in the query set
            eval_iter: Num of iterations
            ckpt: Checkpoint path. Set as None if using current model parameters.
            return: Accuracy
            '''
            print("")

            all_pred = []
            right_list = []

            model.eval()
            if ckpt is None:
                print("No assigned ckpt")
                assert(0)
                
            else:
                print("Use test dataset")
                if ckpt != 'none':
                    state_dict = self.__load_model__(ckpt)['state_dict']
                    own_state = model.state_dict()
                    for name, param in state_dict.items():
                        if name not in own_state:
                            continue
                        own_state[name].copy_(param)
                eval_dataset = self.test_data_loader

            # with torch.no_grad():
            iter_right = 0.0
            iter_sample = 0.0
            for it in tqdm(range(eval_iter)):
                if pair:
                    batch = next(eval_dataset)
                    if torch.cuda.is_available():
                        for k in batch:
                            batch[k] = batch[k].cuda()
                    logits, pred = model(batch, N, K, Q * N + Q * na_rate)

                else:
                    support, query, rel_text = next(eval_dataset)
                    if torch.cuda.is_available():
                        for k in support:
                            support[k] = support[k].cuda()
                        for k in query:
                            query[k] = query[k].cuda()

                        for k in rel_text:
                            rel_text[k] = rel_text[k].cuda()


                ##################### model generation   #########################
                B = support['word'].shape[0] // (N * K)
                w, support_x, q_in = model.generate_weight(support, query, rel_text, N, K, 1)

                new_clsf = New_CLSF(N)
                new_clsf.set_weight(w.detach().squeeze(0))
                new_clsf.cuda()
                new_clsf.train()
                logits = new_clsf(support_x, use_dot=model.dot).squeeze()

                temp_labels = torch.LongTensor(
                    [i for b in range(B) for i in range(N) for j in range(K)]).cuda()
                optimizer = optim.AdamW(new_clsf.parameters(), lr=test_lr)

                loss = F.nll_loss(logits, temp_labels)
                # loss.requires_grad_(True)
                _, pred = torch.max(logits.view(-1, N), 1)
                right = model.accuracy(pred, temp_labels)
                right_list.append(right)

                # temp_w = copy.deepcopy(new_clsf.linear.weight)
                loss.backward()
                optimizer.step()
                new_clsf.eval()

                # q_in = model.generate_input(support, query, rel_text, N, K, 1)

                logits = new_clsf(q_in, use_dot=model.dot)
                minn, _ = logits.min(-1)

                logits = torch.cat([logits, minn.unsqueeze(1) - 1], 1)  # (B, total_Q, N + 1)

                _, pred = torch.max(logits.view(-1, N + 1), 1)
                # right = model.accuracy(pred, label)
                # if right < 1.0:
                #     wrong_case.append(iter_sample)

                iter_right += self.item(right.data)
                iter_sample += 1

                # sys.stdout.write('[EVAL] step: {0:4} | accuracy: {1:3.2f}%'.format(it + 1,
                #                                                                    100 * iter_right / iter_sample) + '\r')
                # sys.stdout.flush()
                list_pred = pred.cpu().numpy().tolist()
                all_pred.extend(list_pred)
                #     logits, pred = model(support, query, rel_text, N, K, Q * N + Q * na_rate)
                #
                # list_pred = pred.cpu().numpy().tolist()
                # temp_list_pred = []
                #
                # for nn in range(B):
                #     temp_list_pred.append(list_pred[N * nn])
                #
                # all_pred.extend(temp_list_pred)
            print("all pred len:", len(all_pred))
            with open('pred-{}-{}.json'.format(N, K), 'w') as f:
                json.dump(all_pred, f)

    def test_param(self,
                   model,
                   B, N, K, Q,
                   eval_iter,
                   na_rate=0,
                   pair=False,
                   ckpt=None,
                   test_lr=0.2,
                   test_output=None):
        '''
        model: a FewShotREModel instance
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        eval_iter: Num of iterations
        ckpt: Checkpoint path. Set as None if using current model parameters.
        return: Accuracy
        '''
        print("")
        print('Use bert optim!')
        all_pred = []
        right_list = []

        if ckpt is None:
            print("No assigned ckpt")
            assert (0)

        else:
            print("Use val dataset for generation")
            if ckpt != 'none':
                state_dict = self.__load_model__(ckpt)['state_dict']
                own_state = model.state_dict()
                for name, param in state_dict.items():
                    if name not in own_state:
                        continue
                    own_state[name].copy_(param)
            eval_dataset = self.val_data_loader  # Parameter tuning
        model.eval()
        iter_right = 0.0
        iter_sample = 0.0
        eval_iter = len(eval_dataset)
        #eval_iter = 1000  # Parameter tuning
        wrong_case = []
        # get_wrong_case = [15.0, 43.0, 57.0, 75.0, 76.0, 85.0, 90.0, 96.0, 97.0, 115.0, 130.0, 136.0, 137.0, 143.0,
        #                   146.0, 175.0, 181.0, 195.0, 206.0, 235.0, 236.0, 238.0, 239.0, 255.0, 263.0, 287.0, 289.0,
        #                   316.0, 317.0, 329.0, 353.0, 364.0, 378.0, 393.0, 425.0, 441.0, 455.0, 457.0, 470.0, 474.0,
        #                   477.0, 485.0, 486.0, 509.0, 511.0, 521.0, 527.0, 546.0, 552.0, 565.0, 574.0, 576.0, 583.0,
        #                   584.0, 623.0, 624.0, 626.0, 633.0, 636.0, 650.0, 676.0, 679.0, 686.0, 691.0, 697.0, 698.0,
        #                   716.0, 723.0, 726.0, 735.0, 796.0, 804.0, 809.0, 811.0, 814.0, 828.0, 833.0, 836.0, 840.0,
        #                   844.0, 849.0, 898.0, 905.0, 924.0, 941.0, 946.0, 947.0, 953.0, 957.0, 962.0, 968.0, 975.0,
        #                   976.0, 992.0]
        for it in range(eval_iter):
            if pair:
                batch, label = next(eval_dataset)
                if torch.cuda.is_available():
                    for k in batch:
                        batch[k] = batch[k].cuda()
                logits, pred = model(batch, N, K, 1)

            else:
                support, query, label, rel_text = next(eval_dataset)

                if torch.cuda.is_available():
                    for k in support:
                        support[k] = support[k].cuda()
                    for k in query:
                        query[k] = query[k].cuda()
                    label = label.cuda()

                    for k in rel_text:
                        rel_text[k] = rel_text[k].cuda()
                    model.cuda()

                # ##################### visualization   #########################
                visualization = model.generate_visualization_emb(support, query, rel_text, N, K, 1)
                print(visualization.shape)
                label = []
                for i in range(N):
                    temp = [i for j in range(K)]
                    label.extend(temp)
                visualization = visualization.cpu().detach().numpy()

                plot_tsne(visualization, label)
                return

                ##################### model generation   #########################
                B = support['word'].shape[0] // (N * K)

                w, support_x, q_in = model.generate_weight(support, query, rel_text, N, K, 1)

                new_clsf = New_CLSF(N)
                new_clsf.set_weight(w.detach().squeeze(0))
                new_clsf.cuda()
                new_clsf.train()

                logits = new_clsf(support_x, use_dot=model.dot).squeeze()

                temp_labels = torch.LongTensor([i for b in range(B) for i in range(N) for j in range(K)]).cuda()
                optimizer = optim.AdamW(new_clsf.parameters(), lr=test_lr, weight_decay=0.01)

                loss = model.loss(logits, temp_labels)
            
                # loss.requires_grad_(True)
                print(loss)
                _, pred = torch.max(logits.view(-1, N), 1)
                right = model.accuracy(pred, temp_labels)
                right_list.append(right)

                temp_w = copy.deepcopy(new_clsf.linear.weight)
                loss.backward()
                optimizer.step()
                new_clsf.eval()

        

                logits = new_clsf(q_in, use_dot=model.dot)
                minn, _ = logits.min(-1)

                logits = torch.cat([logits, minn.unsqueeze(1) - 1], 1)  # (B, total_Q, N + 1)

                _, pred = torch.max(logits.view(-1, N + 1), 1)
                right = model.accuracy(pred, label)
                # if iter_sample in get_wrong_case and right == 1.0:
                #     wrong_case.append(iter_sample)
                #
                #     existing_w = new_clsf.linear.weight.data.detach().cpu().numpy()
                #     temp_w = temp_w.detach().cpu().numpy()
                #     q_in = q_in.detach().cpu().numpy()
                #
                #     np.savez('tensor_visualize.npz', existing_w, temp_w, q_in)
                #     break

                iter_right += self.item(right.data)
                iter_sample += 1

                sys.stdout.write('[EVAL] step: {0:4} | accuracy: {1:3.2f}%'.format(it + 1,
                                                                                   100 * iter_right / iter_sample) + '\r')
                sys.stdout.flush()
                list_pred = pred.cpu().numpy().tolist()
                all_pred.extend(list_pred)
        print("all pred len:", len(all_pred))
#         with open('pred-{}-{}.json'.format(N, K), 'w') as f:
#             json.dump(all_pred, f)
        print('wrong case:', wrong_case)