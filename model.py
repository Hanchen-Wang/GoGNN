import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGPooling, NNConv
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


class NetModular(torch.nn.Module):
    def __init__(self, args):
        super(NetModular, self).__init__()
        self.args = args
        self.num_features = args.num_features
        # self.ddi_num_features = args.ddi_num_features
        self.num_edge_features = args.num_edge_features
        self.nhid = args.nhid
        self.ddi_nhid = args.ddi_nhid
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio

        self.conv1 = GCNConv(self.num_features, self.nhid).to(args.device)
        self.pool1 = SAGPooling(self.nhid, ratio=self.pooling_ratio).to(args.device)
        self.conv2 = GCNConv(self.nhid, self.nhid).to(args.device)
        self.pool2 = SAGPooling(self.nhid, ratio=self.pooling_ratio).to(args.device)
        self.conv3 = GCNConv(self.nhid, self.nhid).to(args.device)
        self.pool3 = SAGPooling(self.nhid, ratio=self.pooling_ratio).to(args.device)

        self.nn = torch.nn.Linear(self.num_edge_features, 6 * self.nhid * self.ddi_nhid)
        self.conv4 = NNConv(6 * self.nhid, self.ddi_nhid, self.nn).to(args.device)
        self.conv_noattn = GCNConv(6 * self.nhid, self.ddi_nhid).to(args.device)

        self.lin1 = torch.nn.Linear(self.ddi_nhid, self.ddi_nhid)
        self.lin2 = torch.nn.Linear(self.ddi_nhid, self.ddi_nhid)
        self.lin3 = torch.nn.Linear(self.num_edge_features, self.ddi_nhid)


    def forward(self, data):
        # I put the edge weight at the position for edge attribute such that the weight can also be masked.
        modular_data, ddi_edge_index, ddi_edge_attr = data
        modular_output = []

        ids = list(modular_data.keys())
        for modular_id in ids:
            x, edge_index, edge_weight, batch = modular_data[modular_id]
            x = x.to(self.args.device)
            edge_index = edge_index.to(self.args.device)
            edge_weight = edge_weight.to(self.args.device)
            batch = batch.to(self.args.device)

            x = F.relu(self.conv1(x, edge_index, edge_weight))
            batch = batch.long()
            x, edge_index, edge_weight, batch, _, _ = self.pool1(x, edge_index, edge_weight, batch)
            x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

            x = F.relu(self.conv2(x, edge_index, edge_weight))
            x, edge_index, edge_weight, batch, _, _ = self.pool2(x, edge_index, edge_weight, batch)
            x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

            x = F.relu(self.conv3(x, edge_index, edge_weight))
            x, edge_index, edge_weight, batch, _, _ = self.pool3(x, edge_index, edge_weight, batch)
            x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

            out_x = torch.cat((x1, x2, x3), dim=1)
            modular_output.append(out_x)

        modular_feature = torch.cat(tuple(modular_output))
        modular_feature = nn.Dropout(self.args.dropout_ratio)(modular_feature)

        # x = F.relu(self.conv4(modular_feature, ddi_edge_index, ddi_edge_attr))
        x = F.relu(self.conv_noattn(modular_feature, ddi_edge_index))
        pos_source, pos_target, neg_source, neg_target = self.feature_split(x, ddi_edge_index)
        pos_feat_x = F.sigmoid(self.lin1(pos_source))
        pos_feat_y = F.sigmoid(self.lin2(pos_target))
        neg_feat_x = F.sigmoid(self.lin1(neg_source))
        neg_feat_y = F.sigmoid(self.lin2(neg_target))
        pos_attr = F.sigmoid(self.lin3(ddi_edge_attr[0: self.args.batch_size]))
        neg_attr = F.sigmoid(self.lin3(ddi_edge_attr[self.args.batch_size:]))

        loss_pos_vec = pos_feat_x + pos_attr - pos_feat_y
        loss_neg_vec = neg_feat_x + neg_attr - neg_feat_y

        norm_pos = torch.norm(loss_pos_vec, p=2, dim=1)
        norm_neg = torch.norm(loss_neg_vec, p=2, dim=1)

        loss = 2*self.ddi_nhid - torch.norm(loss_pos_vec, p=2, dim=1) + self.args.neg_decay * torch.norm(loss_neg_vec, p=2, dim=1)

        return loss, norm_pos, norm_neg, pos_feat_x

    def feature_split(self, features, edge_index):
        source, target = edge_index
        source_feature = features[source]
        target_feature = features[target]
        pos_source = source_feature[0: self.args.batch_size]
        pos_target = target_feature[0: self.args.batch_size]
        neg_source = source_feature[self.args.batch_size:]
        neg_target = target_feature[self.args.batch_size:]

        return pos_source, pos_target, neg_source, neg_target


class NetSeGraph(torch.nn.Module):
    def __init__(self, args):
        super(NetSeGraph, self).__init__()
        self.args = args
        self.num_features = args.num_features
        # self.ddi_num_features = args.ddi_num_features
        self.num_edge_features = args.num_edge_features
        self.nhid = args.nhid
        self.ddi_nhid = args.ddi_nhid
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio

        self.conv1 = GCNConv(self.num_features, self.nhid).to(args.device)
        self.pool1 = SAGPooling(self.nhid, ratio=self.pooling_ratio).to(args.device)
        self.conv2 = GCNConv(self.nhid, self.nhid).to(args.device)
        self.pool2 = SAGPooling(self.nhid, ratio=self.pooling_ratio).to(args.device)
        self.conv3 = GCNConv(self.nhid, self.nhid).to(args.device)
        self.pool3 = SAGPooling(self.nhid, ratio=self.pooling_ratio).to(args.device)

        self.nn = torch.nn.Linear(self.num_edge_features, 6 * self.nhid * self.ddi_nhid)
        self.conv4 = NNConv(6 * self.nhid, self.ddi_nhid, self.nn).to(args.device)

        self.lin1 = torch.nn.Linear(self.ddi_nhid, self.ddi_nhid)
        self.lin2 = torch.nn.Linear(self.ddi_nhid, self.ddi_nhid)
        self.lin3 = torch.nn.Linear(self.num_edge_features, self.ddi_nhid)


    def forward(self, data):
        # I put the edge weight at the position for edge attribute such that the weight can also be masked.
        modular_data, ddi_edge_index, neg_edge_index, ddi_edge_attr, neg_edge_attr = data
        modular_output = []

        ids = list(modular_data.keys())
        for modular_id in ids:
            x, edge_index, edge_weight, batch = modular_data[modular_id]
            x = x.to(self.args.device)
            edge_index = edge_index.to(self.args.device)
            edge_weight = edge_weight.to(self.args.device)
            batch = batch.to(self.args.device)

            x = F.relu(self.conv1(x, edge_index, edge_weight))
            batch = batch.long()
            x, edge_index, edge_weight, batch, _, _ = self.pool1(x, edge_index, edge_weight, batch)
            x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

            x = F.relu(self.conv2(x, edge_index, edge_weight))
            x, edge_index, edge_weight, batch, _, _ = self.pool2(x, edge_index, edge_weight, batch)
            x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

            x = F.relu(self.conv3(x, edge_index, edge_weight))
            x, edge_index, edge_weight, batch, _, _ = self.pool3(x, edge_index, edge_weight, batch)
            x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

            out_x = torch.cat((x1, x2, x3), dim=1)
            modular_output.append(out_x)

        modular_feature = torch.cat(tuple(modular_output))
        modular_feature = nn.Dropout(self.args.dropout_ratio)(modular_feature)

        x = F.relu(self.conv4(modular_feature, ddi_edge_index, ddi_edge_attr))
        pos_source, pos_target, neg_source, neg_target = self.feature_split(x, ddi_edge_index, neg_edge_index)
        # sigmoid or softmax or nothing, add relu
        pos_feat_x = self.lin1(pos_source)
        pos_feat_y = self.lin2(pos_target)
        neg_feat_x = self.lin1(neg_source)
        neg_feat_y = self.lin2(neg_target)
        pos_attr = self.lin3(ddi_edge_attr)
        neg_attr = self.lin3(neg_edge_attr)

        norm_pos, norm_neg = self.xent_loss(pos_feat_x, pos_feat_y, neg_feat_x, neg_feat_y)
        pos_tgt = torch.ones_like(norm_pos)
        neg_tgt = torch.zeros_like(norm_neg)
        #
        loss = nn.BCEWithLogitsLoss()(norm_pos, pos_tgt) + nn.BCEWithLogitsLoss()(norm_neg, neg_tgt)

        return loss, norm_pos, norm_neg, pos_feat_x

    def feature_split(self, features, edge_index, neg_index):
        # while doing the prediction, maybe we can only use half features.
        source, target = edge_index
        pos_source = features[source]
        pos_target = features[target]
        source, target = neg_index
        neg_source = features[source]
        neg_target = features[target]

        return pos_source, pos_target, neg_source, neg_target

    def xent_loss(self, pos_x, pos_y, neg_x, neg_y):
        pos_score = torch.sum(torch.mul(pos_x, pos_y), 1)
        neg_score = torch.sum(torch.mul(neg_x, neg_y), 1)

        return pos_score, neg_score


