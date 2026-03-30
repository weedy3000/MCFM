import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch import nn
from tqdm import tqdm
from MCFM_sims import MCFMmodel_sims
from data_loader import data_loader
from metricsTop import MetricsTop

# global variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def dict_to_str(src_dict):
    dst_str = ""
    for key in src_dict.keys():
        dst_str += " %s: %.4f " % (key, src_dict[key])
    return dst_str


class ChConfig_com(object):
    """Configuration class to store the configurations of training.
    """

    def __init__(self,
                 train_mode='regression',
                 loss_weights={
                     'M': 1,
                     'T': 1,
                     'A': 1,
                     'V': 1,

                 },
                 model_save_path='checkpoint',
                 learning_rate=1e-5,
                 text_model='bert-base',
                 audio_model='sentilar',
                 epochs=20,
                 dataset_name='sims',
                 early_stop=8,
                 seed=0,
                 dropout=0.3,
                 model='cme',
                 audio_feature='raw',
                 batch_size=64,
                 cme_version='v1',
                 tasks='MTA',
                 num_hidden_layers=3
                 ):
        self.train_mode = train_mode
        self.loss_weights = loss_weights
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dataset_name = dataset_name
        self.model_save_path = model_save_path
        self.early_stop = early_stop
        self.seed = seed
        self.dropout = dropout
        self.model = model
        self.audio_feature = audio_feature
        self.batch_size = batch_size
        self.cme_version = cme_version
        self.tasks = 'M'
        self.num_hidden_layers = num_hidden_layers
        self.training_epoch = 0
        self.hidden_dim = 768
        self.histo_type = 'attention'
        self.histo_p = 0.1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_dropout_prob = 0.1
        self.text_model = text_model
        self.audio_model = audio_model


tsne = TSNE(n_components=2,
            random_state=42,
            perplexity=25,
            n_iter=550,
            early_exaggeration=350, learning_rate='auto')


class ChTrainer():
        def __init__(self, config):

            self.config = config
            self.criterion = nn.L1Loss() if config.train_mode == 'regression' else nn.CrossEntropyLoss()
            self.metrics = MetricsTop(config.train_mode).getMetics(config.dataset_name)

        def do_train(self, model, data_loader):
            model.train()
            fc_params = []
            fc_params.extend(model.competitiver.class_sys1.parameters())
            fc_params.extend(model.competitiver.class_sys2.parameters())
            fc_params.extend(model.competitiver.class_fc.parameters())
            all_params = list(model.parameters())
            remaining_params = [p for p in all_params if p not in set(fc_params)]
            optimizer = torch.optim.AdamW([{'params': fc_params, 'lr': 1e-5}, {'params': remaining_params}],
                                          lr=self.config.learning_rate)

            total_loss = 0
            # Loop over all batches.
            for batch in tqdm(data_loader):
                text_inputs = batch["text_tokens"].to(device)
                text_mask = batch["text_masks"].to(device)

                audio_inputs = batch["audio_inputs"].to(device)
                audio_mask = batch["audio_masks"].to(device)

                targets = batch["targets"]

                optimizer.zero_grad()  # To zero out the gradients.
                loss, outputs,hidden_feature = model(text_inputs, text_mask, audio_inputs, audio_mask, targets)
                total_loss += loss[0].item() * text_inputs.size(0)

                loss[0].backward()
                optimizer.step()

            total_loss = round(total_loss / len(data_loader.dataset), 4)
            print('TRAIN' + " >> loss: ", total_loss)
            return total_loss

        def do_test(self, model, data_loader, mode):
            model.eval()  # Put the model in eval mode.
            
            y_pred = []
            y_true = []
            total_loss = 0
            cls_los = 0
            regression_loss = 0
            adv_loss = 0
            diff_loss = 0
            final_loss = 0
            with torch.no_grad():
                for i, batch in enumerate(tqdm(data_loader)):  # Loop over all batches.
                    if i < 100:
                        text_inputs = batch["text_tokens"].to(device)
                        text_mask = batch["text_masks"].to(device)

                        audio_inputs = batch["audio_inputs"].to(device)
                        audio_mask = batch["audio_masks"].to(device)

                        targets = batch["targets"]
                        loss, outputs,hidden_feature = model(text_inputs, text_mask, audio_inputs, audio_mask, targets)

                        total_loss += loss[0].item() * text_inputs.size(0)
                        y_pred.append(outputs.cpu())
                        y_true.append(targets['M'].cpu())
                        cls_los += loss[1].item() * text_inputs.size(0)
                        regression_loss += loss[2].item() * text_inputs.size(0)
                        final_loss += loss[3].item() * text_inputs.size(0)
                        adv_loss += loss[4].item() * text_inputs.size(0)
                        diff_loss += loss[5].item() * text_inputs.size(0)
                total_loss = round(total_loss / len(data_loader.dataset), 4)
                cls_los = round(cls_los / len(data_loader.dataset), 4)
                regression_loss = round(regression_loss / len(data_loader.dataset), 4)
                final_loss = round(final_loss / len(data_loader.dataset), 4)
                adv_loss = round(adv_loss / len(data_loader.dataset), 4)
                diff_loss = round(diff_loss / len(data_loader.dataset), 4)
                print(mode + " >> loss: ", total_loss, 'CLS loss:', cls_los, 'regression_loss:', regression_loss,
                      'final_loss:', final_loss, 'adv_loss:', adv_loss, 'diff_loss:', diff_loss)
                pred, true = torch.cat(y_pred), torch.cat(y_true)
                pred1 = pred.squeeze(1)
                eval_results = self.metrics(pred, true)
                print('%s: >> ' % ('M') + dict_to_str(eval_results))
                eval_results['Loss'] = total_loss
                # visualize(pred1, true, num_class=3)
            return eval_results


def plot_pic(pred, true):
    pred = pred.detach().cpu().numpy()[:50]
    true = true.detach().cpu().numpy()[:50]

    x = [i for i in range(len(pred))]
    x = np.array(x)
    plt.xlabel('sample')
    plt.ylabel('value')
    plt.scatter(x, pred, color='blue')
    plt.plot(x, pred, '-', color='blue', label='predict value')

    plt.scatter(x, true, color='#88c999')
    plt.plot(x, true, '-', color='#88c999', label='true value')
    plt.legend()
    plt.savefig('sims result.png')

    plt.show()


def visualize(X, label, num_class):
    label = label.detach().cpu().numpy()
    labels = []
    pred_label = []
    plt.figure()
    colors = ['blue', 'c', 'y', 'm', 'r', 'g', 'k', 'yellow', 'yellowgreen', 'wheat']

    plt.xlim((0, 50))
    plt.xlabel('samples')
    if num_class == 2:
        for l in label:
            if l < 0:
                labels.append(0)
            else:
                labels.append(1)
        for l in X:
            if l < 0:
                pred_label.append(0)
            else:
                pred_label.append(1)
        plt.yticks([-2, -1, 0, 1, 2], [' ', ' ', r'negative', r'non negative', ' '])
    elif num_class == 3:
        for l in label:
            if l < 0:
                labels.append(0)
            elif l > 0:
                labels.append(1)
        for l in X:
            if l < 0:
                pred_label.append(0)
            elif l > 0:
                pred_label.append(1)

        plt.yticks([-2, -1, 0, 1, 2, 3], [' ', ' ', r'negative', 'positive', ' ', ''])
    elif num_class == 7:
        labels = np.round(label)
        pred_label = np.round(X)
        labels = labels.astype(np.int32)
        plt.yticks(np.arange(-3, 4, 1))
    elif num_class == 5:
        for l in label:
            if l < -0.6:
                labels.append(-2)
            elif -0.6 <= l < -0.2:
                labels.append(-1)
            elif -0.2 <= l < 0.2:
                labels.append(0)
            elif 0.2 <= l < 0.6:
                labels.append(1)
            elif 0.6 <= l:
                labels.append(2)
        for l in X:
            if l < -0.6:
                pred_label.append(-2)
            elif -0.6 <= l < -0.2:
                pred_label.append(-1)
            elif -0.2 <= l < 0.2:
                pred_label.append(0)
            elif 0.2 <= l < 0.6:
                pred_label.append(1)
            elif 0.6 <= l:
                pred_label.append(2)
        plt.yticks(np.arange(-2, 3, 1))
        plt.ylabel('sentiment value')
    # X = tsne.fit_transform(X)
    # X_tsne_2d = np.hstack((X, np.zeros_like(X)))
    x_index = [i for i in range(len(labels))]
    plot_sample = np.random.choice(x_index, size=50, replace=False)

    true = []
    pred = []
    for i in plot_sample:
        true.append(labels[i])
        pred.append(pred_label[i])
    x_index1 = [i for i in range(50)]
    plt.scatter(x_index1, true, marker='o', c=colors[0], label='true label')
    plt.scatter(x_index1, pred, marker='^', c=colors[8], label='predict label')
    plt.legend(bbox_to_anchor=(0.4, 1.03), loc=3, borderaxespad=0)
    if num_class == 2:
        plt.savefig('sims_binary_tsne.png', dpi=600, bbox_inches='tight')
    elif num_class == 3:
        plt.savefig('sims_three_level_tsne.png', dpi=600, bbox_inches='tight')
    elif num_class == 5:
        plt.savefig('sims_five_level_tsne.png', dpi=600, bbox_inches='tight')

    plt.show()


def ChRun_com(config):
    # random.seed(config.seed)
    # torch.manual_seed(config.seed)
    # torch.cuda.manual_seed(config.seed)
    # np.random.seed(config.seed)
    # torch.backends.cudnn.deterministic = True

    train_loader, test_loader, val_loader = data_loader(config.batch_size, config.dataset_name, config.text_model)

    model = MCFMmodel_sims(config).to(device)
    for param in model.feature_extractor.hubert_model.feature_extractor.parameters():
        param.requires_grad = False

    trainer = ChTrainer(config)

    lowest_eval_loss = 100
    highest_eval_acc = 0
    epoch = 0
    best_epoch = 0
    while True:
        print('---------------------EPOCH: ', epoch, '--------------------')
        epoch += 1
        trainer.do_train(model, train_loader)
        eval_results = trainer.do_test(model, val_loader, "VAL")

        if eval_results['Loss'] < lowest_eval_loss:
            lowest_eval_loss = eval_results['Loss']
            torch.save(model.state_dict(), os.path.join(config.model_save_path, config.dataset_name,'lowset_loss.pth'))
            best_epoch = epoch
            print('--------------save lowest loss model-----------------')
        if eval_results['Mult_acc_2'] >= highest_eval_acc:
            highest_eval_acc = eval_results['Mult_acc_2']
            torch.save(model.state_dict(), os.path.join(config.model_save_path, config.dataset_name,'highest_acc.pth'))
            print('--------------save highest acc model------------------')
        if epoch - best_epoch >= config.early_stop:
            break
