"""
MCFM模型（sims数据集）
"""
from __future__ import annotations  # 允许在类型注释中使用尚未定义的类名

import math
from typing import Optional, Dict

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoModel, BertModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)


def gelu(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class MLP(nn.Module):
    """
    Multi-layer perceptron

    Parameters
    ----------
    num_layers: number of hidden layers
    """
    activation_classes = {'gelu': GELU, 'relu': nn.ReLU, 'tanh': nn.Tanh}

    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout, batch_norm=False,
                 init_last_layer_bias_to_zero=False, layer_norm=False, activation='gelu'):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm

        assert not (self.batch_norm and self.layer_norm)

        self.layers = nn.Sequential()
        for i in range(self.num_layers):
            n_in = self.input_size if i == 0 else self.hidden_size
            n_out = self.hidden_size if i < self.num_layers - 1 else self.output_size
            self.layers.add_module(f'{i}-Linear', nn.Linear(n_in, n_out))
            if i < self.num_layers - 1:
                self.layers.add_module(f'{i}-Dropout', nn.Dropout(self.dropout))
                if self.batch_norm:
                    self.layers.add_module(f'{i}-BatchNorm1d', nn.BatchNorm1d(self.hidden_size))
                if self.layer_norm:
                    self.layers.add_module(f'{i}-LayerNorm', nn.LayerNorm(self.hidden_size))
                self.layers.add_module(f'{i}-{activation}', self.activation_classes[activation.lower()]())
        if init_last_layer_bias_to_zero:
            self.layers[-1].bias.data.fill_(0)

    def forward(self, input):
        return self.layers(input)


def gram_schmidt(input):
    """
    正交化函数
    input (c, 1, h, w)
    c是通道数也是滤波器个数
    """

    def projection(u, v):
        return (v * u).sum() / (u * u).sum() * u

    output = []
    for x in input:  # x (1,h,w) c个通道
        for y in output:  # y (1,h,w)  每个新通道的滤波器都要和之间计算好的通道滤波器进行正交
            x = x - projection(y, x)
            # print(x.shape) # (1,h,w)
        x = x / x.norm(p=2)  # 单个通道的滤波器
        output.append(x)
    return torch.stack(output)


def initialize_orthogonal_filters(c, w):
    """
    初始化滤波器
    """
    base = torch.randn(c, w)
    q, _ = torch.linalg.qr(base.view(c, -1))
    return q.view(c, w)


class GramSchmidtTransform(torch.nn.Module):
    """
    滤波器正交化
    """
    instance: Dict[int, Optional[GramSchmidtTransform]] = {}
    constant_filter: Tensor

    @staticmethod
    def build(c: int, h: int):
        if c not in GramSchmidtTransform.instance:
            GramSchmidtTransform.instance[(c, h)] = GramSchmidtTransform(c, h)
        return GramSchmidtTransform.instance[(c, h)]

    def __init__(self, c: int, h: int):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            rand_ortho_filters = initialize_orthogonal_filters(c, h).view(c, h)
        self.register_buffer("constant_filter", rand_ortho_filters.detach())

    def forward(self, x):
        B, C, D = x.shape  # (b,2,dim)
        H, W = self.constant_filter.shape  # (2,1)
        # if h != H or w != W: x = torch.nn.functional.adaptive_avg_pool2d(x, (H, W))
        # print((self.constant_filter * x).shape) #(bs,2,dim)
        return (self.constant_filter * x).sum(dim=(-1), keepdim=True)


class Orthogonal_attention(nn.Module):
    """
    多模态正交注意力
    """

    def __init__(self, input_dim, hidden_dim, channels, height):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.channels = channels
        self.height = height

        # Gram-Schmidt 变换初始化
        self.F_C_A = GramSchmidtTransform.build(channels, height)
        # 通道注意力映射（SE Block 结构）
        self.channel_attention = nn.Sequential(
            nn.Linear(channels, channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels, bias=False),
            nn.Sigmoid()
        )
        self.q_proj = nn.Linear(input_dim, hidden_dim)
        self.k_proj = nn.Linear(input_dim, hidden_dim)

    def forward(self, share, private):
        """
        x (b,dim)
        """
        query = self.q_proj(share).unsqueeze(1)  # (bs,1,hidden_dim)
        key = self.k_proj(private).unsqueeze(1)
        # conv_out1 = self.conv(query) #(bs,out_channel)
        # conv_out2 = self.conv(key)
        B, C, D = query.shape



        conv_out = torch.cat((query, key), dim=1)  # (bs,2,dim)
        # Gram-Schmidt 变换
        transformed = self.F_C_A(conv_out)  # (B, 2, 1)
        # print(transformed.shape)

        # 去除空间维度，进入通道注意力网络
        compressed = transformed.view(B, C * 2)
        # 通道注意力生成
        excitation = self.channel_attention(compressed).view(B, C * 2, 1)
        # 加权原始输入特征
        output = conv_out * excitation
        output += conv_out
        # print(output.shape) #(bs,2,dim)

        share_feature, private_feature = torch.chunk(output, 2, dim=1)

        return share_feature, private_feature


class Ortho_loss(nn.Module):
    def __init__(self, lambda1=0.1, lambda2=0.1):
        super(Ortho_loss, self).__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.pred_loss = nn.CrossEntropyLoss()

    def forward(self, share, private):
        dot_f = torch.mean(share * private, dim=-1)  # (b,len)
        ortho_loss = self.lambda1 * torch.mean(torch.abs(dot_f)) \
                     + self.lambda2 * torch.mean(dot_f ** 2)
        return ortho_loss


class Cross_Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_head=8):
        super(Cross_Attention, self).__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.head_dim = hidden_dim // num_head
        assert hidden_dim // num_head == self.head_dim
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.num_head = num_head

    def forward(self, x, y):
        b, d = x.shape
        q = self.query(x).unsqueeze(1).view(b, 1, self.num_head, self.head_dim)
        k = self.key(y).unsqueeze(1).view(b, 1, self.num_head, self.head_dim)
        v = self.value(y).unsqueeze(1).view(b, 1, self.num_head, self.head_dim)

        score = torch.matmul(q.permute(0, 2, 1, 3), k.permute(0, 2, 3, 1)) / math.sqrt(self.head_dim)  # (b,head,1,1)
        prob = nn.Softmax(dim=-1)(score)  # (b,head,1,1)

        att_out = torch.matmul(prob, v.permute(0, 2, 1, 3))  # (b,head,1,dim)
        att_out = att_out.permute(0, 2, 1, 3).contiguous().squeeze(1)  # (b,head,dim)
        att_out = att_out.view(b, d)

        att_out = self.out(att_out)
        return att_out


class Multimodal_Feature_Extracter(nn.Module):
    def __init__(self, text_model, audio_model, hidden_dim=768):
        super(Multimodal_Feature_Extracter, self).__init__()
        self.text_encoder = BertModel.from_pretrained(text_model)
        self.hubert_model = AutoModel.from_pretrained(audio_model)
        # 文本处理器
        self.sys1 = nn.Sequential(nn.Linear(768, hidden_dim),
                                  nn.ReLU(),
                                  nn.Dropout(0.1),
                                  )
        # 音频处理器
        self.sys2 = nn.Sequential(nn.LSTM(768, hidden_dim, batch_first=True),
                                  nn.Linear(hidden_dim, hidden_dim),
                                  nn.Tanh())

        self.sys1_linear = nn.Linear(hidden_dim, hidden_dim)
        self.sys2_linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, text_ids, text_mask, audio_ids, audio_mask):
        text_out = self.text_encoder(text_ids, text_mask)
        text_last_hidden_state = text_out.last_hidden_state
        text_feature = self.sys1(text_last_hidden_state)

        audio_out = self.hubert_model(audio_ids, audio_mask)
        audio_last_hidden_state = audio_out.last_hidden_state  # (bs,len,dim)

        audio_feature, _ = self.sys2[0](audio_last_hidden_state)
        audio_cls = self.sys2[1](audio_feature[:, -1, :])
        audio_cls = self.sys2[2](audio_cls)

        text_feature = self.sys1_linear(text_feature)
        text_cls = text_feature[:, 0, :]
        audio_feature = self.sys2_linear(audio_feature)
        return text_cls, audio_cls, text_feature, audio_feature


class CompetitiveSelectionMechanism(nn.Module):
    def __init__(self, args, critic_head_num=1, input_dim=768, hidden_dim=768):
        super(CompetitiveSelectionMechanism, self).__init__()
        self.args = args
        # 多头分解
        self.critic_head_num = critic_head_num
        self.critic_head_size = hidden_dim
        self.multi_sys1 = nn.Linear(input_dim, self.critic_head_num * (self.critic_head_size // 2) * 2)
        self.multi_sys2 = nn.Linear(input_dim, self.critic_head_num * (self.critic_head_size // 2) * 2)

        # 对抗训练器
        self.critic_layer_list = nn.ModuleList(
            [MLP(
                input_size=self.critic_head_size // 2,
                hidden_size=hidden_dim,
                output_size=2, num_layers=2, dropout=args.hidden_dropout_prob,
                layer_norm=False) for _ in range(self.critic_head_num)])

        # 分类器
        # self.cls_sys1 = nn.Linear(hidden_dim, 1)
        self.cls_sys1 = MLP(input_size=hidden_dim,
                            hidden_size=hidden_dim,
                            output_size=1, num_layers=1, dropout=args.hidden_dropout_prob,
                            layer_norm=False)
        # self.cls_sys2 = nn.Linear(hidden_dim, 1)
        self.cls_sys2 = MLP(input_size=hidden_dim,
                            hidden_size=hidden_dim,
                            output_size=1, num_layers=1, dropout=args.hidden_dropout_prob,
                            layer_norm=False)

        self.class_sys1 = nn.Linear(hidden_dim, 7)
        self.class_sys2 = nn.Linear(hidden_dim, 7)

        # 动态选择机制
        if self.args.histo_type == "cat":
            self.referee = nn.Linear(hidden_dim * 6, 1)
            self.class_fc = nn.Linear(hidden_dim * 6, 7)
        else:
            self.referee = nn.Linear(hidden_dim * 4, 1)
            self.class_fc = nn.Linear(hidden_dim * 4, 7)

        # 历史模式缓存
        self.histo1 = torch.zeros((2, hidden_dim), dtype=torch.float, device=device)
        self.histo2 = torch.zeros((2, hidden_dim), dtype=torch.float, device=device)
        self.histo_cache1 = []
        self.histo_cache2 = []
        self.last_epoch = -1

        self.cross = Cross_Attention(hidden_dim, hidden_dim)
        self.ortho = Orthogonal_attention(self.critic_head_size // 2, self.critic_head_size // 2, 2, 1)
        self.ortho_loss = Ortho_loss()

    def forward(self, text_cls, audio_cls, text_feature, audio_feature, labels):
        if self.last_epoch != -1 and self.last_epoch != self.args.training_epoch:
            self.history_update()
        self.last_epoch = self.args.training_epoch
        batch_size, length = text_feature.size(0), text_feature.size(1)
        M_label = labels['M'].to(device)
        T_label = labels['T'].to(device)
        A_label = labels['A'].to(device)
        m_la = torch.round(M_label) + 3
        m_la = m_la.long()
        t_la = torch.round(T_label) + 3
        t_la = t_la.long()
        a_la = torch.round(A_label) + 3
        a_la = a_la.long()
        o_last_hidden_tensor1 = text_cls.clone()
        o_last_hidden_tensor2 = audio_cls.clone()

        multi_head_h1 = self.multi_sys1(o_last_hidden_tensor1)  # (b,len,dim)
        multi_head_h2 = self.multi_sys2(o_last_hidden_tensor2)

        multi_head_h1 = multi_head_h1.view(batch_size, self.critic_head_num, 2,
                                           self.critic_head_size // 2)  # (b,head,2,dim/2) 分解为共享特征和私有特征
        multi_head_h2 = multi_head_h2.view(batch_size, self.critic_head_num, 2, self.critic_head_size // 2)

        # adv loss: s1 <-> s2   对抗训练（梯度反转）
        adv_logits_list_1, adv_logits_list_2 = [], []
        for ch_i in range(self.critic_head_num):
            _share_1 = multi_head_h1[:, ch_i, 0, :]  # 文本共享特征 (b,1,1,dim/2)
            _share_2 = multi_head_h2[:, ch_i, 0, :]  # 音频共享特征
            # 2*b,self.critic_head_size // 2
            # 放入对抗训练器进行训练
            adv_logits_list_1.append(
                self.critic_layer_list[ch_i](2 * _share_1.detach().clone() - _share_1))  # gradient reverse  梯度反转 (b,2)
            adv_logits_list_2.append(
                self.critic_layer_list[ch_i](2 * _share_2.detach().clone() - _share_2))  # gradient reverse

        adv_logits_list_1 = torch.cat(adv_logits_list_1, dim=0)  # (b,2)
        adv_logits_list_2 = torch.cat(adv_logits_list_2, dim=0)
        adv_logits = torch.cat([adv_logits_list_1, adv_logits_list_2], dim=0)  # (2b,2)

        adv_logits_label = torch.cat([torch.zeros(len(adv_logits_list_1)), torch.ones(len(adv_logits_list_2))]).to(
            device)  # (2b)
        adv_loss = nn.CrossEntropyLoss()(adv_logits, adv_logits_label.long())

        # 特征正交约束
        share1, private1 = multi_head_h1[:, :, 0, :].view(-1, self.critic_head_size // 2), multi_head_h1[:, :, 1,
                                                                                           :].view(-1,
                                                                                                   self.critic_head_size // 2)
        share2, private2 = multi_head_h2[:, :, 0, :].view(-1, self.critic_head_size // 2), multi_head_h2[:, :, 1,
                                                                                           :].view(-1,
                                                                                                   self.critic_head_size // 2)
        # print(share1.shape) #(bs,dim/2)

        share1, private1 = self.ortho(share1, private1)
        share2, private2 = self.ortho(share2, private2)
        diff_loss = self.ortho_loss(share1, private1)
        diff_loss += self.ortho_loss(share2, private2)

        o_last_hidden_tensor1 = torch.cat((share1, private1), dim=-1).mean(1)
        o_last_hidden_tensor2 = torch.cat((share2, private2), dim=-1).mean(1)
        o_last_hidden_tensor1 = o_last_hidden_tensor1.view(batch_size, -1)
        o_last_hidden_tensor2 = o_last_hidden_tensor2.view(batch_size, -1)

        # 子系统预测
        logits1 = self.cls_sys1(o_last_hidden_tensor1)
        logits2 = self.cls_sys2(o_last_hidden_tensor2)
        loss1 = nn.MSELoss(reduction='none')(logits1.squeeze(), T_label)  # 设定reduction参数为none，损失不加和，直接返回各样本损失
        loss2 = nn.MSELoss(reduction='none')(logits2.squeeze(), A_label)

        cls_logits1 = self.class_sys1(o_last_hidden_tensor1)
        cls_logits2 = self.class_sys2(o_last_hidden_tensor2)
        cls_loss1 = nn.CrossEntropyLoss()(cls_logits1, t_la)
        cls_loss2 = nn.CrossEntropyLoss()(cls_logits2, a_la)

        # 历史模式
        if self.args.histo_type != "none":
            N, D = self.histo1.shape
            b1 = torch.equal(self.histo1, torch.zeros((N, D), dtype=torch.float, device=device))
            b2 = torch.equal(self.histo2, torch.zeros((N, D), dtype=torch.float, device=device))
            if not b1 and not b2:
                historical_h1 = self.get_historical_h(text_cls, self.histo1)  # 获取历史信息
                historical_h2 = self.get_historical_h(audio_cls, self.histo2)
                if self.args.histo_type == "mean":
                    o_last_hidden_tensor1 = (o_last_hidden_tensor1 + historical_h1) / 2
                    o_last_hidden_tensor2 = (o_last_hidden_tensor2 + historical_h2) / 2
                elif self.args.histo_type == "cat":
                    o_last_hidden_tensor1 = torch.cat([o_last_hidden_tensor1, historical_h1], dim=-1)  # 融合历史信息
                    o_last_hidden_tensor2 = torch.cat([o_last_hidden_tensor2, historical_h2], dim=-1)
                elif self.args.histo_type == "attention":
                    o_last_hidden_tensor1 = self.cross(o_last_hidden_tensor1, historical_h1)  # 融合历史信息
                    o_last_hidden_tensor2 = self.cross(o_last_hidden_tensor2, historical_h2)
            self.histo_cache1.append((cls_logits1.argmax(dim=-1).eq(t_la),
                                      loss1.detach().clone().to('cpu'),
                                      o_last_hidden_tensor1.detach().clone().to("cpu")))
            self.histo_cache2.append((cls_logits2.argmax(dim=-1).eq(a_la),
                                      loss2.detach().clone().to("cpu"),
                                      o_last_hidden_tensor2.detach().clone().to("cpu")))

        # 动态选择机制
        merge_hidden = torch.cat([text_cls, o_last_hidden_tensor1, audio_cls, o_last_hidden_tensor2], dim=-1)
        select_logits = self.referee(merge_hidden)
        final_logits = select_logits
        cls_final_logits = self.class_fc(merge_hidden)
        # 损失计算

        final_loss = nn.MSELoss()(final_logits.squeeze(), M_label)
        cls_final_loss = nn.CrossEntropyLoss()(cls_final_logits, m_la)

        # total_cls_loss = (cls_loss1 + cls_loss2)*0.5 + 0.5 * cls_final_loss
        regression_loss = (loss1.mean() + loss2.mean()) * 0.5
        total_loss = (loss1.mean() + loss2.mean()) * 0.5 + final_loss + 0.2 * adv_loss + cls_final_loss + diff_loss

        return [total_loss, cls_final_loss, regression_loss, final_loss, adv_loss * 0.2,
                diff_loss], final_logits, merge_hidden

    def get_historical_h(self, syshidden, histo):
        """
        syshidden  (bs,dim)
        histo (2,dim)
        """
        # 新特征由当前特征和历史特征进行加权，权重由当前特征和历史特征的相似度决定
        history_hidden = histo
        att_score = torch.matmul(syshidden, history_hidden.permute(1, 0))
        att_score = att_score / math.sqrt(syshidden.size(1))
        att_prob = nn.Softmax(dim=-1)(att_score)
        historical_h = torch.matmul(att_prob, history_hidden)


        return historical_h

    def history_update(self):
        """
        历史特征更新函数
        """
        with torch.no_grad():
            histo_p = 0.1
            if self.histo_cache1:
                is_correct = torch.cat([t[0] for t in self.histo_cache1], dim=0).float()  # 模态1正确数
                # print(is_correct.shape) #(num_sample)
                loss_list = torch.cat([t[1] for t in self.histo_cache1], dim=0)  # 模态1损失
                # print(loss_list.shape) #(num_sample)
                sysh_list = torch.cat([t[2] for t in self.histo_cache1], dim=0)  # 模态1最终特征
                # print(sysh_list.shape) #(num_sample,dim)
                assert is_correct.shape == loss_list.shape == sysh_list.shape[:-1]

                pos_sysh, neg_sysh, pos_loss, neg_loss = [], [], [], []
                for i in range(is_correct.size(0)):
                    if is_correct[i] == 1:
                        pos_sysh.append(sysh_list[i])
                        pos_loss.append(loss_list[i])
                    else:
                        neg_sysh.append(sysh_list[i])
                        neg_loss.append(loss_list[i])
                pos_sysh = torch.stack(pos_sysh, dim=0)  # (num_pos,dim)
                neg_sysh = torch.stack(neg_sysh, dim=0)  # (num_neg,dim)
                pos_loss = torch.stack(pos_loss, dim=0)  # (num_pos)
                neg_loss = torch.stack(neg_loss, dim=0)  # (num_neg)
                # print(pos_loss.shape)
                avg_pos_loss = torch.mean(pos_loss)  # 预测正确的样本的平均损失
                avg_neg_loss = torch.mean(neg_loss)
                pos_weight = (avg_pos_loss / pos_loss).softmax(dim=0).unsqueeze(0)  # (1,num_pos)
                neg_weight = (neg_loss / avg_neg_loss).softmax(dim=0).unsqueeze(0)
                # print(pos_weight.shape)
                # 1,data_num  data_num,h -> 1,h
                new_pos_h = torch.bmm(pos_weight.unsqueeze(0), pos_sysh.unsqueeze(0)).squeeze().to(device)
                self.histo1[1] = histo_p * new_pos_h + (1 - histo_p) * self.histo1[1]
                new_neg_h = torch.bmm(neg_weight.unsqueeze(0), neg_sysh.unsqueeze(0)).squeeze().to(device)
                self.histo1[0] = histo_p * new_neg_h + (1 - histo_p) * self.histo1[0]
            if self.histo_cache2:
                is_correct = torch.cat([t[0] for t in self.histo_cache2], dim=0).float()
                loss_list = torch.cat([t[1] for t in self.histo_cache2], dim=0)
                sysh_list = torch.cat([t[2] for t in self.histo_cache2], dim=0)
                assert is_correct.shape == loss_list.shape == sysh_list.shape[:-1]

                pos_sysh, neg_sysh, pos_loss, neg_loss = [], [], [], []
                for i in range(is_correct.size(0)):
                    if is_correct[i] == 1:
                        pos_sysh.append(sysh_list[i])
                        pos_loss.append(loss_list[i])
                    else:
                        neg_sysh.append(sysh_list[i])
                        neg_loss.append(loss_list[i])
                pos_sysh = torch.stack(pos_sysh, dim=0)
                neg_sysh = torch.stack(neg_sysh, dim=0)
                pos_loss = torch.stack(pos_loss, dim=0)
                neg_loss = torch.stack(neg_loss, dim=0)

                avg_pos_loss = torch.mean(pos_loss)
                avg_neg_loss = torch.mean(neg_loss)
                pos_weight = avg_pos_loss / pos_loss
                neg_weight = neg_loss / avg_neg_loss

                # pos_weight, pos_indices = torch.topk(pos_weight, k=pos_weight.size(0) // 5, largest=True, sorted=False)
                # neg_weight, neg_indices = torch.topk(neg_weight, k=neg_weight.size(0) // 5, largest=True, sorted=False)
                # pos_sysh = pos_sysh.index_select(dim=0, index=pos_indices)
                # neg_sysh = neg_sysh.index_select(dim=0, index=neg_indices)

                # 1,data_num
                pos_weight = pos_weight.softmax(dim=0).unsqueeze(0)
                neg_weight = neg_weight.softmax(dim=0).unsqueeze(0)

                # 1,data_num  data_num,h -> 1,h
                new_pos_h = torch.bmm(pos_weight.unsqueeze(0), pos_sysh.unsqueeze(0)).squeeze().to(device)
                self.histo2[1] = histo_p * new_pos_h + (1 - histo_p) * self.histo2[1]
                new_neg_h = torch.bmm(neg_weight.unsqueeze(0), neg_sysh.unsqueeze(0)).squeeze().to(device)
                self.histo2[0] = histo_p * new_neg_h + (1 - histo_p) * self.histo2[0]

            self.histo_cache1 = []
            self.histo_cache2 = []


class MCFMmodel_sims(nn.Module):
    def __init__(self, args):
        super(MCFMmodel_sims, self).__init__()
        self.args = args

        self.feature_extractor = Multimodal_Feature_Extracter(args.text_model, args.audio_model, args.hidden_dim)
        self.competitiver = CompetitiveSelectionMechanism(args)

    def forward(self, text_ids, text_mask, audio_ids, audio_mask, labels):
        text_cls, audio_cls, text_feature, audio_feature = self.feature_extractor(text_ids, text_mask, audio_ids,
                                                                                  audio_mask)

        loss, select_feature, hidden = self.competitiver(text_cls, audio_cls, text_feature, audio_feature, labels)

        return loss, select_feature, hidden
