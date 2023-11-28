# -------------------------------------------------------------------------
# Name: network.py
# Coding: utf8
# Author: Xinyang Qian
# Intro: Containing deep learning network classes.
# -------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import torch.nn.functional as F
import math
import os
from sklearn.metrics import accuracy_score, roc_auc_score
from sparsemax import Sparsemax

import utils
from loss import SCELoss, GCELoss, CELoss


class TCRD(nn.Module):
    # TCR detector (TCRD) can predict a TCR sequence's class (e.g. cancer-associated TCR) (binary classification).
    # It can extract the antigen-specific biochemical features of TCRs based on the convolutional neural network.
    def __init__(self, aa_num=24, feature_num=15, kernel_size=None, filter_num=None, filters_num=1, drop_out=0.4):
        super(TCRD, self).__init__()
        self.aa_num = aa_num  # The number of amino acids that one TCR contains.
        self.feature_num = feature_num  # The dimension of the feature vector of one amino acid.
        if kernel_size is None:
            kernel_size = [2, 3, 4, 5, 6, 7]
        self.kernel_size = kernel_size  # The specification of the convolution kernel in the convolution layer.
        if filter_num is None:
            filter_num = [3, 3, 3, 2, 2, 1]
        assert len(filter_num) == len(kernel_size), \
            "The parameters 'kernel_size' and 'filter_num' set do not match!"
        self.filter_num = []  # The number of the corresponding convolution kernels.
        for ftr in filter_num:
            self.filter_num.append(ftr * filters_num)
        self.drop_out = drop_out
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=self.feature_num,
                                    out_channels=self.filter_num[idx],
                                    kernel_size=h,
                                    stride=1),
                          nn.ReLU(),
                          nn.AdaptiveMaxPool1d(1))
            for idx, h in enumerate(self.kernel_size)
        ])
        self.fc = nn.Linear(sum(self.filter_num), 2)
        self.dropout = nn.Dropout(p=self.drop_out)

    def forward(self, x):
        x = x.reshape(-1, self.feature_num, self.aa_num)
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = out.reshape(-1, sum(self.filter_num))
        out = self.dropout(self.fc(out))
        return out

    @staticmethod
    def training(tcrs, lbs, filters_n, lr, ep, dropout, log_inr, model_f,
                 aa_f, device, loss, alpha, beta, gce_q, loss_w=None, shuffle=True):
        # The format of tcrs is [tcr_0, tcr_1, ..., tcr_n].
        # The format of lbs is [lb_0, lb_1, ..., lb_n].
        # Generate input.
        training_sps = []
        for tcr in tcrs:
            training_sps.append([[tcr]])
        aa_v = utils.get_features(aa_f)  # Read amino acid vectors.
        input_batch, label_batch = utils.generate_input_for_training(training_sps, lbs, aa_v, ins_num=1)
        input_batch, label_batch = torch.Tensor(input_batch), torch.LongTensor(label_batch)
        dataset = Data.TensorDataset(input_batch, label_batch)
        loader = Data.DataLoader(dataset, len(input_batch), shuffle)
        # Set model.
        model = TCRD(filters_num=filters_n, drop_out=dropout).to(torch.device(device))
        criterion = nn.CrossEntropyLoss().to(device)  # The default loss function is CE.
        weights = []
        if loss_w is not None:
            for w in loss_w:
                weights.append([w] * 2)
            weights = torch.FloatTensor(weights)
        else:
            weights = None
        if loss == "SCE":
            criterion = SCELoss(alpha=alpha, beta=beta, num_classes=2).to(device)
        elif loss == "GCE":
            criterion = GCELoss(q=gce_q, num_classes=2).to(device)
        elif loss == "CE":
            criterion = CELoss(w=weights, num_classes=2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        # Training model.
        for epoch in range(ep):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                if (epoch + 1) % log_inr == 0:
                    print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        # Save model.
        if os.path.exists(model_f):
            model_f = model_f + "_overlap.pth"
        torch.save(model.state_dict(), model_f)
        print("The trained model has been saved to: " + model_f)
        return 0

    @staticmethod
    def prediction(tcrs, filters_n, model_f, aa_f, device):
        # The format of tcrs is [tcr_0, tcr_1, ..., tcr_n].
        # Load model.
        model = TCRD(filters_num=filters_n).to(device)
        model.load_state_dict(torch.load(model_f))
        model = model.eval()
        # Predict each TCR.
        tcr_scores = []
        aa_v = utils.get_features(aa_f)  # Read amino acid vectors.
        for tcr in tcrs:
            # Generate input.
            input_x = utils.generate_input_for_prediction([[tcr]], aa_v, ins_num=1)
            input_x = torch.Tensor(input_x).to(torch.device(device))
            # Make prediction.
            predict = model(input_x)
            prob = float(1 / (1 + math.exp(-predict[0][1] + predict[0][0])))
            tcr_scores.append(prob)
        return tcr_scores


class MINN_SA(nn.Module):
    # MINN_SA can predict a TCR repertoire's class (e.g. cancer-associated TCR repertoire) (binary classification).
    def __init__(self, tcr_num=100, aa_num=24, feature_num=15, kernel_size=None, filter_num=None, attention_head_num=1,
                 attention_hidden_size=10, drop_out=0.4, topk=0.05):
        super(MINN_SA, self).__init__()
        self.tcr_num = tcr_num  # The number of TCR sequences that one input individual sample contains.
        self.aa_num = aa_num  # The number of amino acids that one TCR contains.
        self.feature_num = feature_num  # The dimension of the feature vector of one amino acid.
        if kernel_size is None:
            kernel_size = [2, 3, 4, 5, 6, 7]
        self.kernel_size = kernel_size  # The specification of the convolution kernel in the convolution layer.
        if filter_num is None:
            filter_num = [3, 3, 3, 2, 2, 1]
        assert len(filter_num) == len(kernel_size), \
            "The parameters 'kernel_size' and 'filter_num' set do not match!"
        self.filter_num = filter_num  # The number of the corresponding convolution kernels.
        self.all_filter_num = sum(self.filter_num)
        self.attention_head_num = attention_head_num
        self.attention_hidden_size = attention_hidden_size
        self.attention_head_size = int(self.attention_hidden_size / self.attention_head_num)
        self.drop_out = drop_out
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=self.feature_num,
                                    out_channels=self.filter_num[idx],
                                    kernel_size=h,
                                    stride=1),
                          nn.ReLU(),
                          nn.AdaptiveMaxPool1d(1))
            for idx, h in enumerate(self.kernel_size)
        ])
        # Attention
        ninp = self.all_filter_num
        self.waint = nn.Linear(ninp, ninp, bias=False)
        self.waout = nn.Linear(ninp, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.sparsemax = Sparsemax(dim=1)
        self.fclayer1 = nn.Linear(ninp, ninp, bias=True)
        self.fclayer2 = nn.Linear(ninp, ninp, bias=True)
        self.bn1 = nn.BatchNorm1d(ninp)
        self.decoder_f = nn.Linear(ninp, ninp)
        self.decoder_s = nn.Linear(ninp, 2)
        self.dropout = nn.Dropout(p=self.drop_out)

    def forward(self, x):
        x = x.reshape(-1, self.feature_num, self.aa_num)
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = out.reshape(-1, self.tcr_num, self.all_filter_num)
        # Attention
        out = out.transpose(0, 1)
        output = torch.relu(self.dropout(self.fclayer1(out)))
        attw = torch.squeeze(self.waout(output), dim=-1)
        attw = torch.transpose(attw, 0, 1)
        attw = self.sparsemax(attw)
        output = torch.transpose(output, 0, 1)
        output = torch.bmm(torch.unsqueeze(attw, 1), output)
        output = torch.squeeze(output, dim=1)
        output = self.decoder_f(output)
        output = torch.relu(self.bn1(output))
        output = self.decoder_s(output)
        return output, attw

    @staticmethod
    def training(sps, lbs, tcr_num, lr, ep, dropout, log_inr, model_f, aa_f, device, shuffle=False, valid_sps=None,
                 valid_lbs=None):
        # The format of sps is [[tcr_0, tcr_1, ..., tcr_n], ...].
        # The format of lbs is [lb_0, lb_1, ..., lb_n].
        # Generate input.
        aa_v = utils.get_features(aa_f)  # Read amino acid vectors.
        input_batch, label_batch = utils.generate_input_for_training(sps, lbs, aa_v, ins_num=tcr_num)
        input_batch, label_batch = torch.Tensor(input_batch), torch.LongTensor(label_batch)
        dataset = Data.TensorDataset(input_batch, label_batch)
        loader = Data.DataLoader(dataset, len(input_batch), shuffle)
        # Set model.
        model = MINN_SA(drop_out=dropout).to(torch.device(device))
        optimizer = optim.Adam(model.parameters(), lr=lr)
        # Training model.
        if os.path.exists(model_f):
            model_f = model_f + "_overlap.pth"
        max_acc, valid_model_f = 0, model_f + "_temp.pth"
        criterion = nn.CrossEntropyLoss().to(device)
        for epoch in range(ep):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred, attn = model(batch_x)
                loss = criterion(pred, batch_y)
                if (epoch + 1) % log_inr == 0:
                    print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
                    # Valid the model.
                    if valid_sps and valid_lbs:
                        torch.save(model.state_dict(), valid_model_f)
                        valid_probs = MINN_SA.prediction(valid_sps, valid_model_f, aa_f, tcr_num, device)
                        # utils.evaluation(valid_probs, valid_lbs)  # Debug #
                        valid_preds = [1 if pred > 0.5 else 0 for pred in valid_probs]
                        valid_acc = accuracy_score(valid_lbs, valid_preds)
                        if valid_acc > max_acc:
                            max_acc = valid_acc
                            if os.path.exists(model_f):
                                os.remove(model_f)
                            os.rename(valid_model_f, model_f)
                        else:
                            os.remove(valid_model_f)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        # Save model.
        if not (valid_sps and valid_lbs):
            torch.save(model.state_dict(), model_f)
        print("The trained model has been saved to: " + model_f)
        return 0

    @staticmethod
    def prediction(sps, model_f, aa_f, tcr_num, device):
        # The format of tcrs is [tcr_0, tcr_1, ..., tcr_n].
        # Load model.
        model = MINN_SA().to(device)
        model.load_state_dict(torch.load(model_f))
        model = model.eval()
        # Predict each TCR.
        repertoire_scores = []
        aa_v = utils.get_features(aa_f)  # Read amino acid vectors.
        for ind, sp in enumerate(sps):
            # Generate input.
            input_x = utils.generate_input_for_prediction(sp, aa_v, ins_num=tcr_num)
            input_x = torch.Tensor(input_x).to(torch.device(device))
            # Make prediction.
            predict, attn = model(input_x)
            prob = float(1 / (1 + math.exp(-predict[0][1] + predict[0][0])))
            repertoire_scores.append(prob)
        return repertoire_scores


class DeepLION(nn.Module):
    # DeepLION can predict a TCR repertoire's class (e.g. cancer-associated TCR repertoire) (binary classification).
    # It is a deep multi-instance learning model, containing TCRD for caTCRs identification.
    def __init__(self, tcr_num=100, aa_num=24, feature_num=15, kernel_size=None, filter_num=None, drop_out=0.4):
        super(DeepLION, self).__init__()
        self.tcr_num = tcr_num  # The number of TCR sequences that one input individual sample contains.
        self.aa_num = aa_num  # The number of amino acids that one TCR contains.
        self.feature_num = feature_num  # The dimension of the feature vector of one amino acid.
        if kernel_size is None:
            kernel_size = [2, 3, 4, 5, 6, 7]
        self.kernel_size = kernel_size  # The specification of the convolution kernel in the convolution layer.
        if filter_num is None:
            filter_num = [3, 3, 3, 2, 2, 1]
        assert len(filter_num) == len(kernel_size), \
            "The parameters 'kernel_size' and 'filter_num' set do not match!"
        self.filter_num = filter_num  # The number of the corresponding convolution kernels.
        self.drop_out = drop_out
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=self.feature_num,
                                    out_channels=self.filter_num[idx],
                                    kernel_size=h,
                                    stride=1),
                          nn.ReLU(),
                          nn.AdaptiveMaxPool1d(1))
            for idx, h in enumerate(self.kernel_size)
        ])
        self.cnn_fc = nn.Linear(sum(self.filter_num), 1)
        self.mil_fc = nn.Linear(self.tcr_num, 2)
        self.dropout = nn.Dropout(p=self.drop_out)

    def forward(self, x):
        x = x.reshape(-1, self.feature_num, self.aa_num)
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = out.reshape(-1, 1, sum(self.filter_num))
        out = self.dropout(self.cnn_fc(out))
        out = out.reshape(-1, self.tcr_num)
        out = self.dropout(self.mil_fc(out))
        return out

    def forward_tcr(self, x):
        x = x.reshape(-1, self.feature_num, self.aa_num)
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = out.reshape(-1, 1, sum(self.filter_num))
        out = self.dropout(self.cnn_fc(out))
        out = out.reshape(-1, self.tcr_num)
        return out

    def forward_motifs(self, x):
        x = x.reshape(-1, self.feature_num, self.aa_num)
        motif_index = [torch.argmax(conv[0](x), dim=-1) for conv in self.convs]
        motif_index = torch.cat(motif_index, dim=1)
        motif_length = []
        for ind, size in enumerate(self.kernel_size):
            for f in range(self.filter_num[ind]):
                motif_length.append(size)
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = out.reshape(-1, 1, sum(self.filter_num))
        out = self.dropout(self.cnn_fc(out))
        motif_score = out * self.cnn_fc.weight[0] + self.cnn_fc.bias[0] / out.shape[-1]  # 只能计算每一序列层面的，还需要加上对于序列的权重
        out = out.reshape(-1, self.tcr_num)
        out_0 = out * self.mil_fc.weight[0] + self.mil_fc.bias[0] / out.shape[-1]
        out_1 = out * self.mil_fc.weight[1] + self.mil_fc.bias[1] / out.shape[-1]
        out = torch.ones(out.shape) / (torch.ones(out.shape) +
                                           torch.exp(-out_1 + out_0))
        motif_score = motif_score.reshape(-1, self.tcr_num, sum(self.filter_num))
        motif_score_0 = motif_score * self.mil_fc.weight[0].unsqueeze(-1) + \
                        self.mil_fc.bias[0].unsqueeze(-1) / motif_score.shape[-2]
        motif_score_1 = motif_score * self.mil_fc.weight[1].unsqueeze(-1) + \
                        self.mil_fc.bias[1].unsqueeze(-1) / motif_score.shape[-2]
        motif_score = torch.ones(motif_score.shape) / (torch.ones(motif_score.shape) +
                                       torch.exp(-motif_score_1 + motif_score_0))
        return out, motif_index, motif_length, motif_score

    @staticmethod
    def training(sps, lbs, tcr_num, lr, ep, dropout, log_inr, model_f, aa_f, device, loss, alpha, beta, gce_q,
                 loss_w=None, shuffle=True):
        # The format of sps is [[tcr_0, tcr_1, ..., tcr_n], ...].
        # The format of lbs is [lb_0, lb_1, ..., lb_n].
        # Generate input.
        aa_v = utils.get_features(aa_f)  # Read amino acid vectors.
        input_batch, label_batch = utils.generate_input_for_training(sps, lbs, aa_v, ins_num=tcr_num)
        input_batch, label_batch = torch.Tensor(input_batch), torch.LongTensor(label_batch)
        dataset = Data.TensorDataset(input_batch, label_batch)
        loader = Data.DataLoader(dataset, len(input_batch), shuffle)
        # Set model.
        model = DeepLION(drop_out=dropout).to(torch.device(device))
        criterion = nn.CrossEntropyLoss().to(device)  # The default loss function is CE.
        weights = []
        if loss_w is not None:
            for w in loss_w:
                weights.append([w] * 2)
            weights = torch.FloatTensor(weights)
        else:
            weights = None
        if loss == "SCE":
            criterion = SCELoss(alpha=alpha, beta=beta, num_classes=2).to(device)
        elif loss == "GCE":
            criterion = GCELoss(q=gce_q, num_classes=2).to(device)
        elif loss == "CE":
            criterion = CELoss(w=weights, num_classes=2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        # Training model.
        for epoch in range(ep):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                if (epoch + 1) % log_inr == 0:
                    print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        # Save model.
        if os.path.exists(model_f):
            model_f = model_f + "_overlap.pth"
        torch.save(model.state_dict(), model_f)
        print("The trained model has been saved to: " + model_f)
        return 0

    @staticmethod
    def prediction(sps, model_f, aa_f, tcr_num, device):
        # The format of sps is [[tcr_0, tcr_1, ..., tcr_n], ...].
        # The format of lbs is [lb_0, lb_1, ..., lb_n].
        # Load model.
        model = DeepLION().to(device)
        model.load_state_dict(torch.load(model_f))
        model = model.eval()
        # Predict each TCR repertoire.
        repertoire_scores = []
        aa_v = utils.get_features(aa_f)  # Read amino acid vectors.
        for sp in sps:
            # Generate input.
            input_x = utils.generate_input_for_prediction(sp, aa_v, ins_num=tcr_num)
            input_x = torch.Tensor(input_x).to(torch.device(device))
            # Make prediction.
            predict = model(input_x)
            prob = float(1 / (1 + math.exp(-predict[0][1] + predict[0][0])))
            repertoire_scores.append(prob)
        return repertoire_scores

    @staticmethod
    def predict_tcrs(sps, model_f, aa_f, tcr_num, device):
        # The format of sps is [[tcr_0, tcr_1, ..., tcr_n], ...].
        # The format of lbs is [lb_0, lb_1, ..., lb_n].
        # Load model.
        model = DeepLION().to(device)
        model_paras = torch.load(model_f)
        model.load_state_dict(model_paras)
        w0, w1 = model_paras["mil_fc.weight"][0], model_paras["mil_fc.weight"][1]
        model = model.eval()
        # Predict each TCR repertoire.
        tcr_scores, repertoire_scores = [], []
        aa_v = utils.get_features(aa_f)  # Read amino acid vectors.
        for sp in sps:
            # Generate input.
            input_x = utils.generate_input_for_prediction(sp, aa_v, ins_num=tcr_num)
            input_x = torch.Tensor(input_x).to(torch.device(device))
            # Make prediction.
            predict = model.forward_tcr(input_x).tolist()
            scores, probs = [], []
            for tcr in range(tcr_num):
                scores.append(predict[0][tcr])
                probs.append(
                    float(math.exp(-predict[0][tcr] * w1[tcr] + predict[0][tcr] * w0[tcr])))
            tcr_scores.append(scores)
            repertoire_scores.append(probs)
        return tcr_scores, repertoire_scores

    @staticmethod
    def motif_identification(sps, sps_name, model_f, aa_f, tcr_num, device):
        # The format of tcrs is [tcr_0, tcr_1, ..., tcr_n].
        # Load model.
        model = DeepLION().to(device)
        model.load_state_dict(torch.load(model_f))
        model = model.eval()
        # Predict each TCR.
        sequence_scores = []
        aa_v = utils.get_features(aa_f)  # Read amino acid vectors.
        for ind, sp in enumerate(sps):
            # Generate input.
            input_x = utils.generate_input_for_prediction(sp, aa_v, ins_num=tcr_num)
            input_x = torch.Tensor(input_x).to(torch.device(device))
            # Make prediction.
            predict, motif_index, motif_length, motif_score = model.forward_motifs(input_x)
            for i, s in enumerate(sp):
                result = [sps_name[ind]]
                seq = s[0]
                result.append(seq)
                result.append(predict[0][i])
                motifs = []
                for j, m in enumerate(motif_length):
                    motifs.append([seq[motif_index[i][j]: motif_index[i][j] + m], float(motif_score[0][i][j])])
                motifs = sorted(motifs, key=lambda x: x[1], reverse=True)
                result.append(motifs)
                sequence_scores.append(result)
        return sequence_scores


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.linear = nn.Linear(n_heads * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        context, attn = self.scaled_dot_product_attention(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        output = self.linear(context)
        return self.layer_norm(output + residual), attn
        # return context, attn

    def scaled_dot_product_attention(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


class TransMIL(nn.Module):
    # TransMIL can predict a TCR repertoire's class (e.g. cancer-associated TCR repertoire) (binary classification).
    # It contains TCRD and the self-attention mechanism for caTCRs identification.
    def __init__(self, tcr_num=100, aa_num=24, feature_num=15, kernel_size=None, filter_num=None, attention_head_num=1,
                 attention_hidden_size=10, n_layers=1, drop_out=0.4):
        super(TransMIL, self).__init__()
        self.tcr_num = tcr_num  # The number of TCR sequences that one input individual sample contains.
        self.aa_num = aa_num  # The number of amino acids that one TCR contains.
        self.feature_num = feature_num  # The dimension of the feature vector of one amino acid.
        if kernel_size is None:
            kernel_size = [2, 3, 4, 5, 6, 7]
        self.kernel_size = kernel_size  # The specification of the convolution kernel in the convolution layer.
        if filter_num is None:
            filter_num = [3, 3, 3, 2, 2, 1]
        assert len(filter_num) == len(kernel_size), \
            "The parameters 'kernel_size' and 'filter_num' set do not match!"
        self.filter_num = filter_num  # The number of the corresponding convolution kernels.
        self.all_filter_num = sum(self.filter_num)
        self.attention_head_num = attention_head_num
        self.attention_hidden_size = attention_hidden_size
        self.attention_head_size = int(self.attention_hidden_size / self.attention_head_num)
        self.drop_out = drop_out
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=self.feature_num,
                                    out_channels=self.filter_num[idx],
                                    kernel_size=h,
                                    stride=1),
                          nn.ReLU(),
                          nn.AdaptiveMaxPool1d(1))
            for idx, h in enumerate(self.kernel_size)
        ])
        # Attention
        self.attention_layers = nn.ModuleList([MultiHeadAttention(self.all_filter_num, self.attention_head_size,
                                                                  self.attention_head_size, self.attention_head_num)
                                               for _ in range(n_layers)])
        # Method I: Average #
        self.mil_fc = nn.Linear(self.all_filter_num, 2)
        # Method II: MLP #
        # self.fc = nn.Linear(self.attention_hidden_size, 1)
        # self.mil_fc = nn.Linear(self.tcr_num, 2)
        self.dropout = nn.Dropout(p=self.drop_out)

    def forward(self, x, attn_mask=None):
        x = x.reshape(-1, self.feature_num, self.aa_num)
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = out.reshape(-1, self.tcr_num, self.all_filter_num)
        # Attention
        enc_self_attns = []
        if not attn_mask:
            attn_mask = torch.full([out.shape[0], self.tcr_num, self.tcr_num], False, dtype=torch.bool)
        else:
            attn_mask = torch.Tensor(attn_mask)  # Question: Can't transform to the Tensor with data type of torch.bool directly.
            attn_mask = attn_mask.data.eq(1)
        for layer in self.attention_layers:
            out, self_attn = layer(out, out, out, attn_mask)
            enc_self_attns.append(self_attn)
        # Pooling.
        # Method I #
        out = self.mil_fc(out)
        # Avg pooling.
        out = torch.sum(out, dim=1) / self.tcr_num
        # Method II #
        # out = self.fc(hidden_states)
        # out = out.reshape(-1, self.tcr_num)
        # out = self.mil_fc(out)
        out = self.dropout(out)
        return out

    def forward_tcr(self, x, attn_mask=None):
        x = x.reshape(-1, self.feature_num, self.aa_num)
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = out.reshape(-1, self.tcr_num, self.all_filter_num)
        # Attention
        enc_self_attns = []
        if not attn_mask:
            attn_mask = torch.full([out.shape[0], self.tcr_num, self.tcr_num], False, dtype=torch.bool)
        else:
            attn_mask = torch.Tensor(attn_mask)
            attn_mask = attn_mask.data.eq(1)
        for layer in self.attention_layers:
            out, self_attn = layer(out, out, out, attn_mask)
            enc_self_attns.append(self_attn)
        # Pooling.
        # Method I #
        out = self.mil_fc(out)
        return out

    def forward_attention(self, x, attn_mask=None):
        x = x.reshape(-1, self.feature_num, self.aa_num)
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = out.reshape(-1, self.tcr_num, self.all_filter_num)
        # Attention
        enc_self_attns = []
        if not attn_mask:
            attn_mask = torch.full([out.shape[0], self.tcr_num, self.tcr_num], False, dtype=torch.bool)
        else:
            attn_mask = torch.Tensor(attn_mask)
            attn_mask = attn_mask.data.eq(1)
        for layer in self.attention_layers:
            out, self_attn = layer(out, out, out, attn_mask)
            enc_self_attns.append(self_attn)
        return enc_self_attns

    def forward_motifs(self, x, attn_mask=None):
        x = x.reshape(-1, self.feature_num, self.aa_num)
        motif_index = [torch.argmax(conv[0](x), dim=-1) for conv in self.convs]
        motif_index = torch.cat(motif_index, dim=1)
        motif_length = []
        for ind, size in enumerate(self.kernel_size):
            for f in range(self.filter_num[ind]):
                motif_length.append(size)
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = out.reshape(-1, self.tcr_num, self.all_filter_num)
        # Attention
        enc_self_attns = []
        if not attn_mask:
            attn_mask = torch.full([out.shape[0], self.tcr_num, self.tcr_num], False, dtype=torch.bool)
        else:
            attn_mask = torch.Tensor(
                attn_mask)  # Question: Can't transform to the Tensor with data type of torch.bool directly.
            attn_mask = attn_mask.data.eq(1)
        for layer in self.attention_layers:
            out, self_attn = layer(out, out, out, attn_mask)
            enc_self_attns.append(self_attn)
        # Pooling.
        # Method I #
        context_layer = out
        out = self.mil_fc(out)
        motif_score_0 = context_layer * self.mil_fc.weight[0] + self.mil_fc.bias[0] / context_layer.shape[-1]
        motif_score_1 = context_layer * self.mil_fc.weight[1] + self.mil_fc.bias[1] / context_layer.shape[-1]
        motif_score = torch.ones(context_layer.shape) / (torch.ones(context_layer.shape) +
                                                         torch.exp(-motif_score_1 + motif_score_0))
        return out, motif_index, motif_length, enc_self_attns, motif_score

    @staticmethod
    def training(sps, lbs, tcr_num, lr, ep, dropout, log_inr, model_f, aa_f, device, loss, alpha, beta, gce_q,
                 loss_w=None, shuffle=False, valid_sps=None, valid_lbs=None, attn_mask=None, valid_attn_mask=None):
        # The format of sps is [[tcr_0, tcr_1, ..., tcr_n], ...].
        # The format of lbs is [lb_0, lb_1, ..., lb_n].
        # Generate input.
        aa_v = utils.get_features(aa_f)  # Read amino acid vectors.
        input_batch, label_batch = utils.generate_input_for_training(sps, lbs, aa_v, ins_num=tcr_num)
        input_batch, label_batch = torch.Tensor(input_batch), torch.LongTensor(label_batch)
        dataset = Data.TensorDataset(input_batch, label_batch)
        loader = Data.DataLoader(dataset, len(input_batch), shuffle)
        # Set model.
        model = TransMIL(drop_out=dropout).to(torch.device(device))
        criterion = nn.CrossEntropyLoss().to(device)  # The default loss function is CE.
        weights = []
        if loss_w is not None:
            for w in loss_w:
                weights.append([w] * 2)
            weights = torch.FloatTensor(weights)
        else:
            weights = None
        if loss == "SCE":
            criterion = SCELoss(alpha=alpha, beta=beta, num_classes=2).to(device)
        elif loss == "GCE":
            criterion = GCELoss(q=gce_q, num_classes=2).to(device)
        elif loss == "CE":
            criterion = CELoss(w=weights, num_classes=2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        # Training model.
        if os.path.exists(model_f):
            model_f = model_f + "_overlap.pth"
        max_acc, valid_model_f = 0, model_f + "temp.pth"
        for epoch in range(ep):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred = model(batch_x, attn_mask)
                loss = criterion(pred, batch_y)
                if (epoch + 1) % log_inr == 0:
                    print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
                    # Valid the model.
                    if valid_sps and valid_lbs:
                        torch.save(model.state_dict(), valid_model_f)
                        valid_probs = TransMIL.prediction(valid_sps, valid_model_f, aa_f, tcr_num, device,
                                                              valid_attn_mask)
                        # utils.evaluation(valid_probs, valid_lbs)  # Debug #
                        valid_preds = [1 if pred > 0.5 else 0 for pred in valid_probs]
                        valid_acc = accuracy_score(valid_lbs, valid_preds)
                        if valid_acc > max_acc:
                            max_acc = valid_acc
                            if os.path.exists(model_f):
                                os.remove(model_f)
                            os.rename(valid_model_f, model_f)
                        else:
                            os.remove(valid_model_f)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        # Save model.
        if not (valid_sps and valid_lbs):
            torch.save(model.state_dict(), model_f)
        print("The trained model has been saved to: " + model_f)
        return 0

    @staticmethod
    def prediction(sps, model_f, aa_f, tcr_num, device, attn_mask=None):
        # The format of tcrs is [tcr_0, tcr_1, ..., tcr_n].
        # Load model.
        model = TransMIL().to(device)
        model.load_state_dict(torch.load(model_f))
        model = model.eval()
        # Predict each TCR.
        repertoire_scores = []
        aa_v = utils.get_features(aa_f)  # Read amino acid vectors.
        for ind, sp in enumerate(sps):
            # Generate input.
            input_x = utils.generate_input_for_prediction(sp, aa_v, ins_num=tcr_num)
            input_x = torch.Tensor(input_x).to(torch.device(device))
            # Make prediction.
            if attn_mask:
                predict = model(input_x, [attn_mask[ind]])
            else:
                predict = model(input_x, attn_mask)
            prob = float(1 / (1 + math.exp(-predict[0][1] + predict[0][0])))
            repertoire_scores.append(prob)
        return repertoire_scores

    @staticmethod
    def predict_tcrs(sps, model_f, aa_f, tcr_num, device, attn_mask=None):
        # The format of sps is [[tcr_0, tcr_1, ..., tcr_n], ...].
        # The format of lbs is [lb_0, lb_1, ..., lb_n].
        # Load model.
        model = TransMIL().to(device)
        model_paras = torch.load(model_f)
        model.load_state_dict(model_paras)
        model = model.eval()
        # Predict each TCR repertoire.
        tcr_scores, repertoire_scores = [], []
        aa_v = utils.get_features(aa_f)  # Read amino acid vectors.
        for ind, sp in enumerate(sps):
            # Generate input.
            input_x = utils.generate_input_for_prediction(sp, aa_v, ins_num=tcr_num)
            input_x = torch.Tensor(input_x).to(torch.device(device))
            # Make prediction.
            if attn_mask:
                predict = model.forward_tcr(input_x, [attn_mask[ind]]).tolist()
            else:
                predict = model.forward_tcr(input_x, attn_mask).tolist()
            probs = []
            for tcr in range(tcr_num):
                probs.append(
                    float(math.exp((-predict[0][tcr][1] + predict[0][tcr][0]) / tcr_num)))
            repertoire_scores.append(probs)
        return repertoire_scores

    @staticmethod
    def predict_attention(sps, model_f, aa_f, tcr_num, device, layer_no=0, attn_mask=None):
        # The format of sps is [[tcr_0, tcr_1, ..., tcr_n], ...].
        # The format of lbs is [lb_0, lb_1, ..., lb_n].
        # Load model.
        model = TransMIL().to(device)
        model_paras = torch.load(model_f)
        model.load_state_dict(model_paras)
        model = model.eval()
        # Predict each TCR repertoire.
        attention_scores = []
        aa_v = utils.get_features(aa_f)  # Read amino acid vectors.
        for ind, sp in enumerate(sps):
            # Generate input.
            input_x = utils.generate_input_for_prediction(sp, aa_v, ins_num=tcr_num)
            input_x = torch.Tensor(input_x).to(torch.device(device))
            # Make prediction.
            if attn_mask:
                predict = model.forward_attention(input_x, [attn_mask[ind]])[layer_no].tolist()
            else:
                predict = model.forward_attention(input_x, attn_mask).tolist()
            attention_scores.append(predict)
        return attention_scores

    @staticmethod
    def motif_identification(sps, sps_name, model_f, aa_f, tcr_num, device, attention_head_num=1,
                             attention_hidden_size=10, topk=0.05):
        # The format of tcrs is [tcr_0, tcr_1, ..., tcr_n].
        # Load model.
        model = TransMIL().to(device)
        model_paras = torch.load(model_f)
        model.load_state_dict(model_paras)
        model = model.eval()
        # Predict each TCR.
        sequence_scores = []
        aa_v = utils.get_features(aa_f)  # Read amino acid vectors.
        for ind, sp in enumerate(sps):
            # Generate input.
            input_x = utils.generate_input_for_prediction(sp, aa_v, ins_num=tcr_num)
            input_x = torch.Tensor(input_x).to(torch.device(device))
            # Make prediction.
            predict, motif_index, motif_length, attn, motif_score = model.forward_motifs(input_x)
            for i, s in enumerate(sp):
                result = [sps_name[ind]]
                seq = s[0]
                result.append(seq)
                result.append(float(1 / (1 + math.exp(-predict[0][i][1] + predict[0][i][0]))))
                motifs = []
                for j, m in enumerate(motif_length):
                    motifs.append([seq[motif_index[i][j]: motif_index[i][j] + m], float(motif_score[0][i][j])])
                motifs = sorted(motifs, key=lambda x: x[1], reverse=True)
                result.append(motifs)
                sequence_scores.append(result)
        return sequence_scores


class BiFormer(nn.Module):
    # BiFormer can predict a TCR repertoire's class (e.g. cancer-associated TCR repertoire) (binary classification).
    # It contains TCRD and the self-attention mechanism with topk for caTCRs identification.
    def __init__(self, tcr_num=100, aa_num=24, feature_num=15, kernel_size=None, filter_num=None, attention_head_num=1,
                 attention_hidden_size=10, drop_out=0.4, topk=0.05):
        super(BiFormer, self).__init__()
        self.tcr_num = tcr_num  # The number of TCR sequences that one input individual sample contains.
        self.aa_num = aa_num  # The number of amino acids that one TCR contains.
        self.feature_num = feature_num  # The dimension of the feature vector of one amino acid.
        if kernel_size is None:
            kernel_size = [2, 3, 4, 5, 6, 7]
        self.kernel_size = kernel_size  # The specification of the convolution kernel in the convolution layer.
        if filter_num is None:
            filter_num = [3, 3, 3, 2, 2, 1]
        assert len(filter_num) == len(kernel_size), \
            "The parameters 'kernel_size' and 'filter_num' set do not match!"
        self.filter_num = filter_num  # The number of the corresponding convolution kernels.
        self.all_filter_num = sum(self.filter_num)
        self.attention_head_num = attention_head_num
        self.attention_hidden_size = attention_hidden_size
        self.attention_head_size = int(self.attention_hidden_size / self.attention_head_num)
        self.drop_out = drop_out
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=self.feature_num,
                                    out_channels=self.filter_num[idx],
                                    kernel_size=h,
                                    stride=1),
                          nn.ReLU(),
                          nn.AdaptiveMaxPool1d(1))
            for idx, h in enumerate(self.kernel_size)
        ])
        # Attention
        self.topk = int(self.tcr_num * topk)
        self.query = nn.Linear(self.all_filter_num, self.attention_hidden_size)
        self.key = nn.Linear(self.all_filter_num, self.attention_hidden_size)
        self.value = nn.Linear(self.all_filter_num, self.attention_hidden_size)
        self.attn_fc = nn.Linear(self.attention_hidden_size, self.all_filter_num)
        # Method I: Average #
        self.mil_fc = nn.Linear(self.all_filter_num, 2)
        # Method II: MLP #
        # self.fc = nn.Linear(self.attention_hidden_size, 1)
        # self.mil_fc = nn.Linear(self.tcr_num, 2)
        self.dropout = nn.Dropout(p=self.drop_out)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.attention_head_num, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        x = x.reshape(-1, self.feature_num, self.aa_num)
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = out.reshape(-1, self.tcr_num, self.all_filter_num)
        # Attention
        mixed_query_layer = self.query(out)
        mixed_key_layer = self.key(out)
        mixed_value_layer = self.value(out)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        # * Topk self-attention
        a_r = torch.matmul(query_layer.detach(), key_layer.detach().transpose(-1, -2))
        _, idx_r = torch.topk(torch.mean(a_r, 1), k=self.topk, dim=-1)
        # ** Compute topk self-attention
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        idx_r = idx_r.unsqueeze(1).expand(-1, self.attention_head_num, -1, -1)
        value_layer = torch.gather(value_layer.unsqueeze(3).expand(-1, -1, -1, self.tcr_num, -1),
                                   dim=3, index=idx_r.unsqueeze(-1).expand(-1, -1, -1, -1, self.attention_head_size))
        attention_scores = torch.gather(attention_scores, dim=3, index=idx_r)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        attention_probs = attention_probs.unsqueeze(-2)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.flatten(-2, -1)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.attention_hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        context_layer = self.attn_fc(context_layer)
        # Pooling.
        # Method I #
        out = self.mil_fc(context_layer)
        # Avg pooling.
        out = torch.sum(out, dim=1) / self.tcr_num
        # Method II #
        # out = self.fc(hidden_states)
        # out = out.reshape(-1, self.tcr_num)
        # out = self.mil_fc(out)
        out = self.dropout(out)
        return out, attention_probs

    def forward_motifs(self, x):
        x = x.reshape(-1, self.feature_num, self.aa_num)
        motif_index = [torch.argmax(conv[0](x), dim=-1) for conv in self.convs]
        motif_index = torch.cat(motif_index, dim=1)
        motif_length = []
        for ind, size in enumerate(self.kernel_size):
            for f in range(self.filter_num[ind]):
                motif_length.append(size)
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = out.reshape(-1, self.tcr_num, self.all_filter_num)
        # Attention
        mixed_query_layer = self.query(out)
        mixed_key_layer = self.key(out)
        mixed_value_layer = self.value(out)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        # * Topk self-attention
        a_r = torch.matmul(query_layer.detach(), key_layer.detach().transpose(-1, -2))
        _, idx_r = torch.topk(torch.mean(a_r, 1), k=self.topk, dim=-1)
        # ** Compute topk self-attention
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        idx_r = idx_r.unsqueeze(1).expand(-1, self.attention_head_num, -1, -1)
        value_layer = torch.gather(value_layer.unsqueeze(3).expand(-1, -1, -1, self.tcr_num, -1),
                                   dim=3, index=idx_r.unsqueeze(-1).expand(-1, -1, -1, -1, self.attention_head_size))
        attention_scores = torch.gather(attention_scores, dim=3, index=idx_r)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        attention_probs = attention_probs.unsqueeze(-2)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.flatten(-2, -1)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.attention_hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        context_layer = self.attn_fc(context_layer)
        # Pooling.
        # Method I #
        out = self.mil_fc(context_layer)
        motif_score_0 = context_layer * self.mil_fc.weight[0] + self.mil_fc.bias[0] / context_layer.shape[-1]
        motif_score_1 = context_layer * self.mil_fc.weight[1] + self.mil_fc.bias[1] / context_layer.shape[-1]
        motif_score = torch.ones(context_layer.shape) / (torch.ones(context_layer.shape) +
                                                         torch.exp(-motif_score_1 + motif_score_0))
        return out, motif_index, motif_length, attention_probs, motif_score

    @staticmethod
    def training(sps, lbs, tcr_num, lr, ep, dropout, log_inr, model_f, aa_f, device, shuffle=False, valid_sps=None,
                 valid_lbs=None):
        # The format of sps is [[tcr_0, tcr_1, ..., tcr_n], ...].
        # The format of lbs is [lb_0, lb_1, ..., lb_n].
        # Generate input.
        aa_v = utils.get_features(aa_f)  # Read amino acid vectors.
        input_batch, label_batch = utils.generate_input_for_training(sps, lbs, aa_v, ins_num=tcr_num)
        input_batch, label_batch = torch.Tensor(input_batch), torch.LongTensor(label_batch)
        dataset = Data.TensorDataset(input_batch, label_batch)
        loader = Data.DataLoader(dataset, len(input_batch), shuffle)
        # Set model.
        model = BiFormer(drop_out=dropout).to(torch.device(device))
        optimizer = optim.Adam(model.parameters(), lr=lr)
        # Training model.
        if os.path.exists(model_f):
            model_f = model_f + "_overlap.pth"
        max_acc, valid_model_f = 0, model_f + "temp.pth"
        criterion = nn.CrossEntropyLoss().to(device)
        for epoch in range(ep):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred, attn = model(batch_x)
                loss = criterion(pred, batch_y)
                if (epoch + 1) % log_inr == 0:
                    print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
                    # Valid the model.
                    if valid_sps and valid_lbs:
                        torch.save(model.state_dict(), valid_model_f)
                        valid_probs = BiFormer.prediction(valid_sps, valid_model_f, aa_f, tcr_num, device)
                        # utils.evaluation(valid_probs, valid_lbs)  # Debug #
                        valid_preds = [1 if pred > 0.5 else 0 for pred in valid_probs]
                        valid_acc = accuracy_score(valid_lbs, valid_preds)
                        if valid_acc > max_acc:
                            max_acc = valid_acc
                            if os.path.exists(model_f):
                                os.remove(model_f)
                            os.rename(valid_model_f, model_f)
                        else:
                            os.remove(valid_model_f)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        # Save model.
        if not (valid_sps and valid_lbs):
            torch.save(model.state_dict(), model_f)
        print("The trained model has been saved to: " + model_f)
        return 0

    @staticmethod
    def prediction(sps, model_f, aa_f, tcr_num, device):
        # The format of tcrs is [tcr_0, tcr_1, ..., tcr_n].
        # Load model.
        model = BiFormer().to(device)
        model.load_state_dict(torch.load(model_f))
        model = model.eval()
        # Predict each TCR.
        repertoire_scores = []
        aa_v = utils.get_features(aa_f)  # Read amino acid vectors.
        for ind, sp in enumerate(sps):
            # Generate input.
            input_x = utils.generate_input_for_prediction(sp, aa_v, ins_num=tcr_num)
            input_x = torch.Tensor(input_x).to(torch.device(device))
            # Make prediction.
            predict, attn = model(input_x)
            prob = float(1 / (1 + math.exp(-predict[0][1] + predict[0][0])))
            repertoire_scores.append(prob)
        return repertoire_scores

    @staticmethod
    def motif_identification(sps, sps_name, model_f, aa_f, tcr_num, device, attention_head_num=1,
                             attention_hidden_size=10, topk=0.05):
        # The format of tcrs is [tcr_0, tcr_1, ..., tcr_n].
        # Load model.
        model = BiFormer().to(device)
        model_paras = torch.load(model_f)
        model.load_state_dict(model_paras)
        model = model.eval()
        # Predict each TCR.
        sequence_scores = []
        aa_v = utils.get_features(aa_f)  # Read amino acid vectors.
        for ind, sp in enumerate(sps):
            # Generate input.
            input_x = utils.generate_input_for_prediction(sp, aa_v, ins_num=tcr_num)
            input_x = torch.Tensor(input_x).to(torch.device(device))
            # Make prediction.
            predict, motif_index, motif_length, attn, motif_score = model.forward_motifs(input_x)
            for i, s in enumerate(sp):
                result = [sps_name[ind]]
                seq = s[0]
                result.append(seq)
                result.append(float(1 / (1 + math.exp(-predict[0][i][1] + predict[0][i][0]))))
                motifs = []
                for j, m in enumerate(motif_length):
                    motifs.append([seq[motif_index[i][j]: motif_index[i][j] + m], float(motif_score[0][i][j])])
                motifs = sorted(motifs, key=lambda x: x[1], reverse=True)
                result.append(motifs)
                sequence_scores.append(result)
        return sequence_scores


class DeepLION2(nn.Module):
    # DeepLION2 can predict a TCR repertoire's class (e.g. cancer-associated TCR repertoire) (binary classification).
    # It contains TCRD and the self-attention mechanism with topk and self-learning for caTCRs identification.
    def __init__(self, tcr_num=100, aa_num=24, feature_num=15, kernel_size=None, filter_num=None, attention_head_num=1,
                 attention_hidden_size=10, drop_out=0.4, topk=0.05):
        super(DeepLION2, self).__init__()
        self.tcr_num = tcr_num  # The number of TCR sequences that one input individual sample contains.
        self.aa_num = aa_num  # The number of amino acids that one TCR contains.
        self.feature_num = feature_num  # The dimension of the feature vector of one amino acid.
        if kernel_size is None:
            kernel_size = [2, 3, 4, 5, 6, 7]
        self.kernel_size = kernel_size  # The specification of the convolution kernel in the convolution layer.
        if filter_num is None:
            filter_num = [3, 3, 3, 2, 2, 1]
        assert len(filter_num) == len(kernel_size), \
            "The parameters 'kernel_size' and 'filter_num' set do not match!"
        self.filter_num = filter_num  # The number of the corresponding convolution kernels.
        self.all_filter_num = sum(self.filter_num)
        self.attention_head_num = attention_head_num
        self.attention_hidden_size = attention_hidden_size
        self.attention_head_size = int(self.attention_hidden_size / self.attention_head_num)
        self.drop_out = drop_out
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=self.feature_num,
                                    out_channels=self.filter_num[idx],
                                    kernel_size=h,
                                    stride=1),
                          nn.ReLU(),
                          nn.AdaptiveMaxPool1d(1))
            for idx, h in enumerate(self.kernel_size)
        ])
        # Attention
        self.topk = int(self.tcr_num * topk)
        self.query = nn.Linear(self.all_filter_num, self.attention_hidden_size)
        self.key = nn.Linear(self.all_filter_num, self.attention_hidden_size)
        self.value = nn.Linear(self.all_filter_num, self.attention_hidden_size)
        self.attn_fc = nn.Linear(self.attention_hidden_size, self.all_filter_num)
        # Method I: Average #
        self.mil_fc = nn.Linear(self.all_filter_num, 2)
        # Method II: MLP #
        # self.fc = nn.Linear(self.attention_hidden_size, 1)
        # self.mil_fc = nn.Linear(self.tcr_num, 2)
        self.dropout = nn.Dropout(p=self.drop_out)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.attention_head_num, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        x = x.reshape(-1, self.feature_num, self.aa_num)
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = out.reshape(-1, self.tcr_num, self.all_filter_num)
        # Attention
        mixed_query_layer = self.query(out)
        mixed_key_layer = self.key(out)
        mixed_value_layer = self.value(out)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        # * Complete self-attention
        attention_scores_complete = torch.matmul(query_layer.detach(), key_layer.detach().transpose(-1, -2))
        attention_scores_complete = attention_scores_complete / math.sqrt(self.attention_head_size)
        attention_scores_complete = self.dropout(nn.Softmax(dim=-1)(attention_scores_complete))
        context_layer_complete = torch.matmul(attention_scores_complete, value_layer.detach())
        context_layer_complete = context_layer_complete.permute(0, 2, 1, 3).contiguous()
        new_context_layer_complete_shape = context_layer_complete.size()[:-2] + (self.attention_hidden_size,)
        context_layer_complete = context_layer_complete.view(*new_context_layer_complete_shape)
        context_layer_complete = self.attn_fc(context_layer_complete)
        # * Topk self-attention
        a_r = torch.matmul(query_layer.detach(), key_layer.detach().transpose(-1, -2))
        _, idx_r = torch.topk(torch.mean(a_r, 1), k=self.topk, dim=-1)
        # ** Compute topk self-attention
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        idx_r = idx_r.unsqueeze(1).expand(-1, self.attention_head_num, -1, -1)
        value_layer = torch.gather(value_layer.unsqueeze(3).expand(-1, -1, -1, self.tcr_num, -1),
                                   dim=3, index=idx_r.unsqueeze(-1).expand(-1, -1, -1, -1, self.attention_head_size))
        attention_scores = torch.gather(attention_scores, dim=3, index=idx_r)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        attention_probs = attention_probs.unsqueeze(-2)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.flatten(-2, -1)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.attention_hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        context_layer = self.attn_fc(context_layer)
        # Pooling.
        # Method I #
        out = self.mil_fc(context_layer)
        # Avg pooling.
        out = torch.sum(out, dim=1) / self.tcr_num
        # Method II #
        # out = self.fc(hidden_states)
        # out = out.reshape(-1, self.tcr_num)
        # out = self.mil_fc(out)
        out = self.dropout(out)
        return out, attention_probs, context_layer, context_layer_complete

    def forward_motifs(self, x):
        x = x.reshape(-1, self.feature_num, self.aa_num)
        motif_index = [torch.argmax(conv[0](x), dim=-1) for conv in self.convs]
        motif_index = torch.cat(motif_index, dim=1)
        motif_length = []
        for ind, size in enumerate(self.kernel_size):
            for f in range(self.filter_num[ind]):
                motif_length.append(size)
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = out.reshape(-1, self.tcr_num, self.all_filter_num)
        # Attention
        mixed_query_layer = self.query(out)
        mixed_key_layer = self.key(out)
        mixed_value_layer = self.value(out)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        # * Topk self-attention
        a_r = torch.matmul(query_layer.detach(), key_layer.detach().transpose(-1, -2))
        _, idx_r = torch.topk(torch.mean(a_r, 1), k=self.topk, dim=-1)
        # ** Compute topk self-attention
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        idx_r = idx_r.unsqueeze(1).expand(-1, self.attention_head_num, -1, -1)
        value_layer = torch.gather(value_layer.unsqueeze(3).expand(-1, -1, -1, self.tcr_num, -1),
                                   dim=3, index=idx_r.unsqueeze(-1).expand(-1, -1, -1, -1, self.attention_head_size))
        attention_scores = torch.gather(attention_scores, dim=3, index=idx_r)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        attention_probs = attention_probs.unsqueeze(-2)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.flatten(-2, -1)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.attention_hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        context_layer = self.attn_fc(context_layer)
        # Pooling.
        # Method I #
        out = self.mil_fc(context_layer)
        motif_score_0 = context_layer * self.mil_fc.weight[0] + self.mil_fc.bias[0] / context_layer.shape[-1]
        motif_score_1 = context_layer * self.mil_fc.weight[1] + self.mil_fc.bias[1] / context_layer.shape[-1]
        motif_score = torch.ones(context_layer.shape) / (torch.ones(context_layer.shape) +
                                                         torch.exp(-motif_score_1 + motif_score_0))
        return out, motif_index, motif_length, attention_probs, motif_score

    @staticmethod
    def training(sps, lbs, tcr_num, lr, ep, dropout, log_inr, model_f, aa_f, device, shuffle=False, valid_sps=None,
                 valid_lbs=None, attention_head_num=1, attention_hidden_size=10, topk=0.05):
        # The format of sps is [[tcr_0, tcr_1, ..., tcr_n], ...].
        # The format of lbs is [lb_0, lb_1, ..., lb_n].
        # Generate input.
        aa_v = utils.get_features(aa_f)  # Read amino acid vectors.
        input_batch, label_batch = utils.generate_input_for_training(sps, lbs, aa_v, ins_num=tcr_num)
        input_batch, label_batch = torch.Tensor(input_batch), torch.LongTensor(label_batch)
        dataset = Data.TensorDataset(input_batch, label_batch)
        loader = Data.DataLoader(dataset, len(input_batch), shuffle)
        # Set model.
        model = DeepLION2(drop_out=dropout, attention_head_num=1, attention_hidden_size=10, topk=0.05).\
            to(torch.device(device))
        optimizer = optim.Adam(model.parameters(), lr=lr)
        # Training model.
        if os.path.exists(model_f):
            model_f = model_f + "_overlap.pth"
        max_acc, valid_model_f = 0, model_f + "temp.pth"
        criterion_1 = nn.CrossEntropyLoss().to(device)
        criterion_2 = nn.MSELoss().to(device)
        for epoch in range(ep):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred, attn, attn_out, attn_out_complete = model(batch_x)
                loss = criterion_1(pred, batch_y) + criterion_2(attn_out, attn_out_complete)
                if (epoch + 1) % log_inr == 0:
                    print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
                    # Valid the model.
                    if valid_sps and valid_lbs:
                        torch.save(model.state_dict(), valid_model_f)
                        valid_probs = DeepLION2.prediction(valid_sps, valid_model_f, aa_f, tcr_num, device)
                        # utils.evaluation(valid_probs, valid_lbs)  # Debug #
                        valid_preds = [1 if pred > 0.5 else 0 for pred in valid_probs]
                        valid_acc = accuracy_score(valid_lbs, valid_preds)
                        if valid_acc > max_acc:
                            max_acc = valid_acc
                            if os.path.exists(model_f):
                                os.remove(model_f)
                            os.rename(valid_model_f, model_f)
                        else:
                            os.remove(valid_model_f)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        # Save model.
        if not (valid_sps and valid_lbs):
            torch.save(model.state_dict(), model_f)
        print("The trained model has been saved to: " + model_f)
        return 0

    @staticmethod
    def prediction(sps, model_f, aa_f, tcr_num, device, attention_head_num=1, attention_hidden_size=10, topk=0.05):
        # The format of tcrs is [tcr_0, tcr_1, ..., tcr_n].
        # Load model.
        model = DeepLION2(attention_head_num=1, attention_hidden_size=10, topk=0.05).to(device)
        model.load_state_dict(torch.load(model_f))
        model = model.eval()
        # Predict each TCR.
        repertoire_scores = []
        aa_v = utils.get_features(aa_f)  # Read amino acid vectors.
        for ind, sp in enumerate(sps):
            # Generate input.
            input_x = utils.generate_input_for_prediction(sp, aa_v, ins_num=tcr_num)
            input_x = torch.Tensor(input_x).to(torch.device(device))
            # Make prediction.
            predict, attn, _, _ = model(input_x)
            prob = float(1 / (1 + math.exp(-predict[0][1] + predict[0][0])))
            repertoire_scores.append(prob)
        return repertoire_scores

    @staticmethod
    def motif_identification(sps, sps_name, model_f, aa_f, tcr_num, device, attention_head_num=1, attention_hidden_size=10, topk=0.05):
        # The format of tcrs is [tcr_0, tcr_1, ..., tcr_n].
        # Load model.
        model = DeepLION2(attention_head_num=1, attention_hidden_size=10, topk=0.05).to(device)
        model_paras = torch.load(model_f)
        model.load_state_dict(model_paras)
        model = model.eval()
        # Predict each TCR.
        sequence_scores = []
        aa_v = utils.get_features(aa_f)  # Read amino acid vectors.
        for ind, sp in enumerate(sps):
            # Generate input.
            input_x = utils.generate_input_for_prediction(sp, aa_v, ins_num=tcr_num)
            input_x = torch.Tensor(input_x).to(torch.device(device))
            # Make prediction.
            predict, motif_index, motif_length, attn, motif_score = model.forward_motifs(input_x)
            for i, s in enumerate(sp):
                result = [sps_name[ind]]
                seq = s[0]
                result.append(seq)
                result.append(float(1 / (1 + math.exp(-predict[0][i][1] + predict[0][i][0]))))
                motifs = []
                for j, m in enumerate(motif_length):
                    motifs.append([seq[motif_index[i][j]: motif_index[i][j] + m], float(motif_score[0][i][j])])
                motifs = sorted(motifs, key=lambda x: x[1], reverse=True)
                result.append(motifs)
                sequence_scores.append(result)
        return sequence_scores
