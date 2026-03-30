import torch
import matplotlib.pyplot as plt
from torch import nn
from tqdm import tqdm
from sklearn.manifold import TSNE
from utils.metricsTop import MetricsTop
from utils.MCFM_imeocap import MultimodalEmotionAnalyzer_ortho
import random
import numpy as np
from utils.data_loader import data_loader
from itertools import chain
import os
# global variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def dict_to_str(src_dict):
    dst_str = ""
    for key in src_dict.keys():
        dst_str += " %s: %.4f " %(key, src_dict[key]) 
    return dst_str

class EnConfig_com(object):
    """Configuration class to store the configurations of training.
    """
    def __init__(self,
                train_mode = 'classfication',
                loss_weights = {
                    'M':1,
                    'T':1,
                    'A':1,
                    'H':1,
                    'C':1
                },
                 model_save_path = 'result/',
                 text_model = 'bert-base',
                 audio_model = 'sentilar',
                 learning_rate = 1e-5,
                 epochs = 20,
                 dataset_name = 'mosei',
                 early_stop = 8,
                 seed = 0,
                 dropout=0.3,
                 model='cc',
                 batch_size = 16,
                 multi_task = False,
                 model_size = 'small',
                 cme_version = 'v1',
                 num_hidden_layers = 1,
                 tasks = 'MTA',   # 'M' or 'MTA',
                 context = True,
                 text_context_len = 2,
                 audio_context_len = 1,
                ):

        self.train_mode = train_mode
        self.text_model = text_model
        self.audio_model = audio_model
        self.loss_weights = loss_weights
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dataset_name = dataset_name
        self.model_save_path = model_save_path+dataset_name+'/'
        self.early_stop = early_stop
        self.seed = seed
        self.dropout = dropout
        self.model = model
        self.batch_size = batch_size
        self.multi_task = multi_task
        self.model_size = model_size
        self.cme_version = cme_version
        self.num_hidden_layers = num_hidden_layers
        self.tasks = tasks
        self.context = context
        self.text_context_len = text_context_len
        self.audio_context_len = audio_context_len
        self.training_epoch = 0
        self.hidden_dim=768
        self.histo_type = 'attention'
        self.histo_p = 0.1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_dropout_prob = 0.1


tsne = TSNE(n_components=2,
            random_state=42,
            perplexity=25,
            n_iter=550,
            early_exaggeration=350,learning_rate='auto')
        
class EnTrainer():
    def __init__(self, config):
 
        self.config = config
        self.criterion = nn.L1Loss() if config.train_mode == 'regression' else nn.CrossEntropyLoss()
        self.metrics = MetricsTop(config.train_mode).getMetics(config.dataset_name)
        self.tasks = config.tasks

    def do_train(self, model, data_loader):    
        model.train()
        fc_params = []
        fc_params.extend(model.competitiver.class_sys1.parameters())
        fc_params.extend(model.competitiver.class_sys2.parameters())
        fc_params.extend(model.competitiver.class_fc.parameters())
        all_params = list(model.parameters())
        remaining_params = [p for p in all_params if p not in set(fc_params)]
        optimizer = torch.optim.AdamW([{'params':fc_params,'lr':1e-5},{'params':remaining_params}], lr=self.config.learning_rate)

        total_loss = 0
        # Loop over all batches.         
        for batch in tqdm(data_loader):                    
            text_inputs = batch["text_tokens"].to(device)
            text_mask = batch["text_masks"].to(device)
           

            audio_inputs = batch["audio_inputs"].to(device)
            audio_mask = batch["audio_masks"].to(device)


            targets_s = batch["targets_s"].to(device).view(-1, 1).squeeze(1)
            targets_e = batch["targets_e"].to(device).view(-1, 1).squeeze(1)
            
            
            optimizer.zero_grad()                    # To zero out the gradients.
            loss,sentiment_output,emotion_output = model(text_inputs, text_mask, audio_inputs, audio_mask, targets_s,targets_e)
            total_loss += loss[0].item()*text_inputs.size(0) 
  
        
            loss[0].backward()                   
            optimizer.step()                
                
        total_loss = round(total_loss / len(data_loader.dataset), 4)
        print('TRAIN'+" >> loss: ",total_loss)
        return total_loss

    def do_test(self, model, data_loader, mode):
        model.eval()   # Put the model in eval mode.
        if self.config.multi_task:
            y_pred = {'M': [], 'T': [], 'A': []}
            y_true = {'M': [], 'T': [], 'A': []}
            total_loss = 0
            val_loss = {
                'M':0,
                'T':0,
                'A':0,
                'cls':0,
                'ort':0
            }
        else:
            y_pred_sen= []
            y_pred_emo = []
            y_true_sen = []
            y_true_emo = []
            y_pred_co = []
            total_loss = 0
            cls_los = 0
            sentiment_loss = 0
            emotion_loss = 0
            adv_loss = 0
            diff_loss = 0
            out_feature = []
        with torch.no_grad():
            for batch in tqdm(data_loader):                    # Loop over all batches.
                text_inputs = batch["text_tokens"].to(device)
                text_mask = batch["text_masks"].to(device)
              

                audio_inputs = batch["audio_inputs"].to(device)
                audio_mask = batch["audio_masks"].to(device)
               
                targets_s = batch["targets_s"].to(device).view(-1, 1).squeeze(1)
                targets_e = batch["targets_e"].to(device).view(-1, 1).squeeze(1)

                   
                loss,sentiment_output,emotion_output = model(text_inputs, text_mask, audio_inputs, audio_mask, targets_s,targets_e)
                
                total_loss += loss[0].item()*text_inputs.size(0)
                y_pred_sen.append(sentiment_output.cpu())
                y_pred_emo.append(emotion_output.cpu())
                y_true_sen.append(targets_s.cpu())
                y_true_emo.append(targets_e.cpu())
                
                cls_los += loss[1].item()*text_inputs.size(0)
                
                sentiment_loss += loss[2].item()*text_inputs.size(0)
                emotion_loss += loss[3].item()*text_inputs.size(0)
                adv_loss += loss[4].item()*text_inputs.size(0)
                diff_loss += loss[5].item()*text_inputs.size(0)
            total_loss = round(total_loss / len(data_loader.dataset), 4)
            cls_los = round(cls_los / len(data_loader.dataset), 4)
            sentiment_loss = round(sentiment_loss / len(data_loader.dataset), 4)
            emotion_loss = round(emotion_loss / len(data_loader.dataset), 4)
            adv_loss = round(adv_loss / len(data_loader.dataset), 4)
            diff_loss = round(diff_loss / len(data_loader.dataset), 4)
            print(mode+" >> loss: ",total_loss,'CLS loss:',cls_los,'sentiment_loss:',sentiment_loss,'emotion_loss:',emotion_loss,'adv_loss:',adv_loss,'diff_loss:',diff_loss)
            
            pred, true = torch.cat(y_pred_sen), torch.cat(y_true_sen)
            pred1 = pred.squeeze(1)
            eval_results = self.metrics(pred, true)
            print('%s: >> ' %('M') + dict_to_str(eval_results))
            eval_results['Loss'] = total_loss
            # pred_co = torch.cat(y_pred_co)
            # co_eval_results = self.metrics(pred_co, true)
            # print('%s: >> ' %('M') + dict_to_str(co_eval_results))
            # plot_pic(pred1,true)
            #out_feature = torch.cat(out_feature,dim=0).detach().cpu().numpy()

            visualize(pred1,true,num_class=2)
        return eval_results


def visualize(X,label,num_class):
    label = label.detach().cpu().numpy()
    #label = np.argmax(label,axis=1)
    labels = []
    pred_label = []
    pred_index = []
    plt.figure()
    colors = ['blue', 'c', 'y', 'm', 'r', 'g', 'k', 'yellow', 'yellowgreen', 'wheat']
    X = np.argmax(X.detach().cpu().numpy(),axis=1) #2、3分类用
    plt.xlim((0, 50))
    plt.xlabel('samples')
    if num_class == 2:
        for l in label:
            if l != -1:
                labels.append(1)
            else:
                labels.append(0)
        for i,l in enumerate(X):
            if l ==2:
                pred_label.append(0)
                pred_index.append(i)
            elif l==3 or l==4:
                pred_label.append(1)
                pred_index.append(i)
            else:
                pred_label.append(5)
        plt.ylim(-0.5,1.5)
        plt.yticks([0, 1], [r'negative', 'non   \nnegative'])
        plt.ylabel('Sentiment categories')
    elif num_class == 3:
        for l in label:
            if l == -1:
                labels.append(0)
            elif l == 1:
                labels.append(2)
            else:
                labels.append(1)
        for i,l in enumerate(X):
            if l ==2:
                pred_label.append(0)
                pred_index.append(i)
            elif l ==4:
                pred_label.append(2)
                pred_index.append(i)
            elif l == 3:
                pred_label.append(1)
                pred_index.append(i)
            else:
                pred_label.append(5)
        plt.ylim(-0.5,2.5)
        plt.ylabel('Sentiment categories')
        plt.yticks([0, 1, 2], [r'negative', r'neutral', 'positive'])
    elif num_class == 7:
        labels = np.round(label)
        pred_label = np.round(X)
        labels = labels.astype(np.int32)
        plt.ylim(-3.5,4.5)
        plt.yticks(np.arange(-3,4,1))
        plt.ylabel('Sentiment categories')
    # X = tsne.fit_transform(X)
    # X_tsne_2d = np.hstack((X, np.zeros_like(X)))
    # x_index = [i for i in range(len(X))] # 7分类用
    plot_sample = np.random.choice(pred_index,size=50,replace=False) # 2、3分类用
    #plot_sample = np.random.choice(x_index,size=50,replace=False) # 7分类用
    true = []
    pred = []
    for i in plot_sample:
        true.append(labels[i])
        pred.append(pred_label[i])
    x_index1 = [i for i in range(50)]
    plt.scatter(x_index1, true, marker='o', c=colors[0],label='true label')
    plt.scatter(x_index1, pred, marker='^', c=colors[8],label='predict label')
    #plt.legend(bbox_to_anchor=(0.4, 1.03), loc=3, borderaxespad=0)
    plt.legend()
    if num_class == 2:
        plt.savefig('mosei_binary_tsne.png', dpi=1000, bbox_inches='tight')
    elif num_class == 3:
        plt.savefig('mosei_three_level_tsne.png', dpi=1000, bbox_inches='tight')
    elif num_class == 7:
        plt.savefig('mosei_seven_level_tsne.png', dpi=1000, bbox_inches='tight')

    plt.show()


def EnRun_com(config):
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.backends.cudnn.deterministic = True
    
    print('create data loader')
    train_loader, test_loader, val_loader = data_loader(config.batch_size, config.dataset_name,config.text_model,
                                                        text_context_length=config.text_context_len,
                                                        audio_context_length=config.audio_context_len)
    
    print('create model')
    if config.context:
        if config.model == 'competitive':
            model = MultimodalEmotionAnalyzer(config).to(device)
        elif config.model == 'competitive+transformer':
            model = MultimodalEmotionAnalyzer_trans(config).to(device)
        elif config.model == 'competitive+ortho':
            model = MultimodalEmotionAnalyzer_ortho(config).to(device)
        else:
            raise ValueError('输入正确的模型名称')
        for param in model.feature_extractor.audio_model.feature_extractor.parameters():
            param.requires_grad = False
    else:
        if config.model == 'cc':
            model = rob_d2v_cc(config).to(device)
        elif config.model == 'cme':
            model = rob_d2v_cme(config).to(device)
        
        for param in model.data2vec_model.feature_extractor.parameters():
            param.requires_grad = False

    trainer = EnTrainer(config)

    lowest_eval_loss = 100
    highest_eval_acc = 0
    epoch = 0
    best_epoch = 0

    print('start training')
#     while True:
#         print('---------------------EPOCH: ', epoch, '--------------------')
#         epoch += 1
#         config.training_epoch = epoch-1
#         trainer.do_train(model, train_loader)
#         eval_results = trainer.do_test(model, val_loader,"VAL")
    
#         if eval_results['Loss']<lowest_eval_loss:
#             lowest_eval_loss = eval_results['Loss']
#             torch.save(model.state_dict(), os.path.join(config.model_save_path,f'MCFM_lowest_loss.pth'))
#             best_epoch = epoch
#             print('---------------------save lowest loss model---------------------')
#         if eval_results['Has0_acc_2']>=highest_eval_acc:
#             highest_eval_acc = eval_results['Has0_acc_2']
#             torch.save(model.state_dict(), os.path.join(config.model_save_path,f'MCFM_highest_acc.pth'))
#             test_results_loss = trainer.do_test(model, test_loader,"TEST")
#             print('---------------------save highest acc model---------------------')
#         if epoch - best_epoch >= config.early_stop:
#             break
    
#     f = open(f'{config.model}_result.txt','w',encoding='utf-8')

    model.load_state_dict(torch.load(os.path.join(config.model_save_path,f'MCFM_highest_acc.pth')))
    # model.load_state_dict(torch.load('./result/mosei_result/competitive+ortho_highest_acc.pth'))
    test_results_loss = trainer.do_test(model, test_loader,"TEST")
    print('%s: >> ' %('TEST (highest val acc) ') + dict_to_str(test_results_loss))
    # f.write('%s: >> ' %('TEST (highest val acc) ') + dict_to_str(test_results_loss))
    # f.write('\n')
    model.load_state_dict(torch.load(os.path.join(config.model_save_path,f'MCFM_lowest_loss.pth')))
    # model.load_state_dict(torch.load(os.path.join(config.model_save_path,'myplus1',f'{config.model}_lowest_loss.pth')),strict=False)
    test_results_acc = trainer.do_test(model, test_loader,"TEST")
    print('%s: >> ' %('TEST (lowest val loss) ') + dict_to_str(test_results_acc))
    # f.write('%s: >> ' %('TEST (lowest val loss) ') + dict_to_str(test_results_acc))
    # f.write('\n')
    # f.close()
    
    