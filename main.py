import networkx as nx
import numpy as np
import torch
import json
import argparse
import pickle
import warnings
# from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader
from sklearn import metrics

from data_process import load_data, split_data, negative_generator, name_to_feature, read_negative
from model import NetModular, NetSeGraph
from batch import batch_feed, graph_batch_feed
from data_loader import dict_dataset

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()

parser.add_argument('--nhid', type=int, default=64,
                    help='nhid')
parser.add_argument('--ddi_nhid', type=int, default=256,
                    help='ddi_nhid')
parser.add_argument('--pooling_ratio', type=float, default=0.5,
                    help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.2,
                    help='dropout ratio')
parser.add_argument('--batch_size', type=int, default=2000,
                    help='batch size')
parser.add_argument('--train_ratio', type=float, default=0.8,
                    help='training ratio')
parser.add_argument('--val_ratio', type=float, default=0.1,
                    help='validation ratio')
parser.add_argument('--test_ratio', type=float, default=0.1,
                    help='test ratio')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='learning rate for training')
parser.add_argument('--num_epoch', type=int, default=30,
                    help='number of training epoch')
parser.add_argument('--neg_decay', type=float, default=1.0,
                    help='negative sample loss decay')
parser.add_argument('--modular_file', type=str, default='./data/decagon_data/id_SMILE.txt',
                    help='file store the modulars information')
parser.add_argument('--ddi_file', type=str, default='./data/decagon_data/bio-decagon-combo.csv',
                    help='file store the ddi information')
parser.add_argument('--model_path', type=str, default='./saved_model/',
                    help='saved model path')
parser.add_argument('--feature_type', type=str, default='onehot',
                    help='the feature type for the atoms in modulars')
parser.add_argument('--train_type', type=str, default='se',
                    help='training type of the model, each batch contains fixed edges or a side effect graph')

args = parser.parse_args()
# print the model parameters.
print(args)

# output contains the modular data like [node_attribute, edge_index, edge_weight, batch]
output, edges, edges_attr, se_name = load_data(args.modular_file, args.ddi_file, 'onehot')
print(len(list(output.keys())))
args.num_edge_features = edges_attr.size(1)
args.device = 'cpu'

# split data into train val test.
num_edges = edges_attr.size(0) // 2
train_num = int(num_edges * args.train_ratio)
val_num = int(num_edges * args.val_ratio)
test_num = int(num_edges * args.test_ratio)
nums = [train_num, val_num, test_num]

# change the input to the the side effect name
train_edges, train_edges_attr, val_edges, val_edges_attr, test_edges, test_edges_attr \
    = split_data(edges, se_name, nums)
# print(train_edges_attr)
train_name = train_edges_attr
val_name = val_edges_attr
test_name = test_edges_attr
train_edges_attr = name_to_feature(train_edges_attr)
val_edges_attr = name_to_feature(val_edges_attr)
test_edges_attr = name_to_feature(test_edges_attr)

# read negative samples from file
neg_train_edges, neg_train_attr, neg_val_edges, neg_val_attr, neg_test_edges, neg_test_attr = read_negative()
print('negative samples generated')

print(args.device)
if args.feature_type == 'onehot':
    args.num_features = 22
else:
    args.num_features = 8

if args.train_type == 'edge':
    model = NetModular(args).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=5e-4)
    for key in list(output.keys()):
        output[key][0].to(args.device)
        output[key][1].to(args.device)
        output[key][2].to(args.device)
        output[key][3].to(args.device)
    # modular_dataset = dict_dataset(output)
    # modular_generator = DataLoader(modular_dataset, batch_size=645)  # input all modular data
    # rol, cow = edges
    # new_edge = torch.cat((torch.unsqueeze(rol[0: 64], 0), torch.unsqueeze(cow[0: 64], 0)), dim=0)
    # ddi_data = [output, new_edge, edges_attr[0:64]]
    # out_vec = model(ddi_data)
    # print(out_vec)

    # train process
    for i in range(args.num_epoch):
        batch_num = 0
        last_epoch = False
        print('At {}th epoch'.format(i))
        while not last_epoch:
            model.train()
            optimizer.zero_grad()
            batch_edges, batch_attr, last_epoch = batch_feed(args.batch_size, batch_num, train_edges, train_edges_attr,
                                                             neg_train_edges, neg_train_attr)
            # for mod_x, mod_index, mod_weight, mod_batch in modular_generator:
            # mod_x.to(args.device)
            batch_edges = batch_edges.to(args.device)
            batch_attr = batch_attr.to(args.device)
            in_data = [output, batch_edges, batch_attr]
            loss, _, _, pf = model(in_data)
            loss = torch.mean(loss)
            if batch_num == 0:
                sum_loss = loss
            else:
                sum_loss += loss


            batch_num += 1
            if batch_num % 100 == 0:
                print('Running on {}th batch'.format(batch_num))
                print(loss)
                # print(pf)
            # if batch_num > 100:
            #     break
        sum_loss.backward()
        optimizer.step()
    print('Start evaluation')


    # evaluation
    def sigmoid_array(x):
        return 1 / (1 + np.exp(-x))

    def softmax_array(x, y):
        newx = []
        newy = []
        for i in range(len(x)):
            e1 = x[i]
            e2 = y[i]
            mx_e = max(e1, e2)
            e1 = e1 - mx_e
            e2 = e2 - mx_e
            new_e1 = np.exp(e1)/(np.exp(e1)+np.exp(e2))
            new_e2 = np.exp(e2)/(np.exp(e1)+np.exp(e2))
            newx.append(new_e1)
            newy.append(new_e2)
        return np.array(newx), np.array(newy)


    last_epoch = False
    batch_num = 0
    auc_sum = 0
    count = 0
    while not last_epoch:
        model.eval()
        batch_edges, batch_attr, last_epoch = batch_feed(args.batch_size, batch_num, test_edges, test_edges_attr,
                                                         neg_test_edges, neg_test_attr)
        batch_edges = batch_edges.to(args.device)
        batch_attr = batch_attr.to(args.device)
        in_data = [output, batch_edges, batch_attr]
        _, pos_pred, neg_pred, _ = model(in_data)
        pos_pred = sigmoid_array(torch.Tensor.numpy(torch.Tensor.cpu(pos_pred.detach())))
        neg_pred = sigmoid_array(torch.Tensor.numpy(torch.Tensor.cpu(neg_pred.detach())))
        pos_pred, neg_pred = softmax_array(pos_pred, neg_pred)
        pred_all = np.hstack([pos_pred, neg_pred])
        pred_label = np.hstack([np.ones_like(pos_pred), np.zeros_like(neg_pred)])
        auc = metrics.roc_auc_score(pred_label, pred_all)
        auc_sum += auc
        count += 1
        batch_num += 1
        if count%100 == 0:
            print('Test at {} batch'.format(count))
            print('current AUC is {}'.format(auc_sum/count))
    print('AUC is {}'.format(auc_sum/count))

if args.train_type == 'se':

    model = NetSeGraph(args).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=5e-4)
    for key in list(output.keys()):
        output[key][0].to(args.device)
        output[key][1].to(args.device)
        output[key][2].to(args.device)
        output[key][3].to(args.device)

    # train process
    for i in range(args.num_epoch):
        batch_num = 0
        flag_num = 0
        last_epoch = False
        print('At {}th epoch'.format(i))
        while not last_epoch:
            model.train()
            optimizer.zero_grad()
            t_edges, n_edges, t_attr, n_attr, last_epoch, args.batch_size = graph_batch_feed(batch_num, train_edges, train_edges_attr,
                                                             neg_train_edges, neg_train_attr, train_name)
            # for mod_x, mod_index, mod_weight, mod_batch in modular_generator:
            # mod_x.to(args.device)
            t_edges = t_edges.to(args.device)
            t_attr = t_attr.to(args.device)
            in_data = [output, t_edges, n_edges, t_attr, n_attr]
            loss, pos_pred, neg_pred, pf = model(in_data)
            if flag_num == 0:
                sum_loss = torch.mean(loss)
            else:
                sum_loss += torch.mean(loss)

            batch_num += 1
            flag_num += 1
            if flag_num % 30 == 0:
                print('Running on {}th batch'.format(batch_num))
                print(loss)
                sum_loss.backward()
                optimizer.step()
                flag_num = 0
        if i <= 0:
            continue
        print('Start evaluation')

        # torch.save(model.state_dict(), args.model_path)

        # evaluation
        def sigmoid_array(x):
            return 1 / (1 + np.exp(-x))


        def softmax_array(x, y):
            newx = []
            newy = []
            for i in range(len(x)):
                e1 = x[i]
                e2 = y[i]
                mx_e = max(e1, e2)
                e1 = e1 - mx_e
                e2 = e2 - mx_e
                new_e1 = np.exp(e1) / (np.exp(e1) + np.exp(e2))
                new_e2 = np.exp(e2) / (np.exp(e1) + np.exp(e2))
                newx.append(new_e1)
                newy.append(new_e2)
            return np.array(newx), np.array(newy)


        last_epoch = False
        batch_num = 0
        auc_orig_sum = 0
        auc_sum = 0
        ap_sum = 0
        count = 0
        while not last_epoch:
            model.eval()
            t_edges, n_edges, t_attr, n_attr, last_epoch, args.batch_size = graph_batch_feed(batch_num, test_edges,
                                                                                             test_edges_attr,
                                                                                             neg_test_edges,
                                                                                             neg_test_attr, test_name)
            t_edges = t_edges.to(args.device)
            t_attr = t_attr.to(args.device)
            in_data = [output, t_edges, n_edges, t_attr, n_attr]
            _, pos_pred, neg_pred, pf = model(in_data)
            pos_pred = torch.Tensor.numpy(torch.squeeze(torch.Tensor.cpu(pos_pred.detach())))
            neg_pred = torch.Tensor.numpy(torch.squeeze(torch.Tensor.cpu(neg_pred.detach())))

            pos_pred_soft, neg_pred_soft = softmax_array(pos_pred, neg_pred)

            pred_soft = np.hstack([pos_pred_soft, neg_pred_soft])
            pred_all = np.hstack([pos_pred, neg_pred])
            pred_label = np.hstack([np.ones_like(pos_pred), np.zeros_like(neg_pred)])
            fp, tp, _ = metrics.roc_curve(pred_label, pred_all)
            auc_orig = metrics.auc(fp, tp)
            auc = metrics.roc_auc_score(pred_label, pred_soft)
            ap = metrics.average_precision_score(pred_label, pred_soft)
            auc_orig_sum += auc_orig
            auc_sum += auc
            ap_sum += ap
            count += 1
            batch_num += 1
            if count % 100 == 0:
                print('Test at {} batch'.format(count))
                print('current AUC origin is {}'.format(auc_orig_sum / count))
                print('current AUC is {}'.format(auc_sum / count))
                print('current AP is {}'.format(ap_sum / count))
        print('AUC origin is {}'.format(auc_orig_sum / count))
        print('AUC is {}'.format(auc_sum / count))
        print('AP is {}'.format(ap_sum / count))

