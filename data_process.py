import networkx as nx
import numpy as np
import torch
import json
import csv
import random
from rdkit import Chem
from rdkit.Chem import GraphDescriptors
from torch_geometric.data import Data
from copy import deepcopy


random.seed(666)


def mol_to_nx(mol):
    G = nx.Graph()

    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(),
                   symbol=atom.GetSymbol(),
                   formal_charge=atom.GetFormalCharge(),
                   implicit_valence=atom.GetImplicitValence(),
                   ring_atom=atom.IsInRing(),
                   degree=atom.GetDegree(),
                   hybridization=atom.GetHybridization())
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   bond_type=bond.GetBondType())
    return G


def load_se_dict(file_name):
    with open(file_name) as f:
        se_dict = json.load(fp=f)
    return se_dict


def process(file_name):
    id_graph_dict = dict()
    with open(file_name) as f:
        for line in f:
            line = line.rstrip()
            id, smiles = line.split()
            mol = Chem.MolFromSmiles(smiles)
            graph = mol_to_nx(mol)
            id_graph_dict[id] = graph
    # with open('/data/hancwang/DDI/code/data/id_graph_dict.json', 'w') as f:
    #     json.dump(id_graph_dict, f)
    return id_graph_dict


def node_feature_process(G, feature_type='onehot'):
    feature = []
    feature_dict = node_feature_dict(feature_type)
    symbols = nx.get_node_attributes(G, 'symbol')
    k = sorted(list(symbols.keys()))
    for key in k:
        feature.append(feature_dict[symbols[key]])
    num_nodes = len(k)
    batch_list = [0] * num_nodes
    batch = torch.IntTensor(batch_list)  # int tensor for batch
    return torch.Tensor(feature), batch


def node_feature_dict(type='onehot'):
    # generate the node feature for each symbol(element)
    # we have 22 different elements in this data set we use the one hot vector
    # or fix_dim 8 dim vector to represent each symbol.
    num_symbols = 22
    fixed_dim = 8
    symbol_dict = dict()
    keys = ['C', 'Co', 'P', 'K', 'Br', 'B', 'As', 'F', 'Ca', 'La', 'O', 'Au', 'Gd', 'Na', 'Se', 'N', 'Pt', 'S', 'Al',
            'Li', 'Cl', 'I']
    if type == 'onehot':
        for i in range(len(keys)):
            temp = [0] * num_symbols
            temp[i] = 1
            feature = temp
            symbol_dict[keys[i]] = deepcopy(feature)  # just in case
    elif type == 'fixed':
        for i in range(len(keys)):
            temp = [0] * fixed_dim
            if i <= 7:
                temp[i] = 1
            elif 8 <= i <= 14:
                temp[0] = 1
                temp[i % 8 + 1] = 1
            elif 15 <= i <= 20:
                temp[1] = 1
                temp[(i + 3) % 8] = 1
            else:
                temp[2] = 1
                temp[3] = 1
            feature = temp
            symbol_dict[keys[i]] = deepcopy(feature)  # just in case
    return symbol_dict


def edge_preprocess(G):
    edge_weight = []
    edge_1 = []
    edge_2 = []
    bond_types_dict = {'SINGLE': 1, 'DOUBLE': 2, 'TRIPLE': 3, 'AROMATIC': 1.5}
    bonds = nx.get_edge_attributes(G, 'bond_type')
    edge_index = list(bonds.keys())
    for edge in edge_index:
        edge_weight.append(bond_types_dict[str(bonds[edge])])  # change the name of bond to the string to match the dict
        edge_1.append(edge[0])
        edge_2.append(edge[1])
        edge_1.append(edge[1])
        edge_2.append(edge[0])
        edge_weight.append(bond_types_dict[str(bonds[edge])])
    edge_output = [edge_1, edge_2]
    return torch.LongTensor(edge_output), torch.Tensor(edge_weight)


def load_data(modular_file_name, DDI_file_name, feature_type='onehot'):
    # load the graph with the one-hot node feature, output the dict whose key is cid,
    # values are [node feature, edge index and edge weight] (a list).
    d = process(modular_file_name)
    output = dict()
    for ids in d.keys():
        graph = d[ids]
        node_feature, batch = node_feature_process(graph, feature_type)
        edge_index, edge_weight = edge_preprocess(graph)
        output[ids] = [node_feature, edge_index, edge_weight, batch]
    edges, edges_attr, se_name = load_DDI_graph(DDI_file_name, list(output.keys()))
    return output, torch.LongTensor(edges), torch.Tensor(edges_attr), se_name  # output includes the data for modular graph(small graph)


def load_DDI_graph(file_name, node_ids):
    # print(node_ids)
    small_se = []
    with open('/data/hancwang/DDI/small_se.txt') as f:
        for line in f:
            line = line.rstrip()
            small_se.append(line)
    edges = []
    edges_1 = []
    edges_2 = []
    edges_attr = []
    se_name = []
    se_dict = load_se_dict('./data/decagon_data/se_embedding.json')
    id_dict = dict()
    for i in range(len(node_ids)):
        new_id = 'CID' + '0' * (9 - len(node_ids[i])) + node_ids[i]
        id_dict[new_id] = i
    with open(file_name) as f:
        sreader = csv.reader(f, delimiter=',')
        next(sreader, None)
        for line in sreader:
            if line[3] in small_se:
                continue
            edges_1.append(id_dict[line[0]])
            edges_2.append(id_dict[line[1]])
            edges_1.append(id_dict[line[1]])
            edges_2.append(id_dict[line[0]])
            # edges.append([id_dict[line[0]], id_dict[line[1]]])
            # edges.append([id_dict[line[1]], id_dict[line[0]]])  # need to store two direction of the edges.
            edges_attr.append(se_dict[line[3]])
            edges_attr.append(se_dict[line[3]])
            se_name.append(line[3])
            se_name.append(line[3])
        edges.append(edges_1)
        edges.append(edges_2)
    return edges, edges_attr, se_name


def new_index(index_1, index_2, index_3):
    train_index = []
    val_index = []
    test_index = []
    for index in index_1:
        train_index.append(2 * index)
        train_index.append(2 * index + 1)
    for index in index_2:
        val_index.append(2 * index)
        val_index.append(2 * index + 1)
    for index in index_3:
        test_index.append(2 * index)
        test_index.append(2 * index + 1)
    return train_index, val_index, test_index


def split_data(edges, edges_attr, nums):
    train_num, val_num, test_num = nums
    edges_attr = np.array(edges_attr)
    total_nums = len(edges_attr) // 2
    index_list = []
    for i in range(total_nums):
        index_list.append(i)
    random.shuffle(index_list)
    train_index = index_list[0:train_num]
    val_index = index_list[train_num: train_num + val_num]
    test_index = index_list[train_num + val_num: train_num + val_num + test_num]
    train_index, val_index, test_index = new_index(train_index, val_index, test_index)
    row, col = edges
    train_edges = torch.cat((torch.unsqueeze(row[train_index], 0), torch.unsqueeze(col[train_index], 0)), dim=0)
    train_attr = edges_attr[train_index]
    val_edges = torch.cat((torch.unsqueeze(row[val_index], 0), torch.unsqueeze(col[val_index], 0)), dim=0)
    val_attr = edges_attr[val_index]
    test_edges = torch.cat((torch.unsqueeze(row[test_index], 0), torch.unsqueeze(col[test_index], 0)), dim=0)
    test_attr = edges_attr[test_index]
    return train_edges, train_attr, val_edges, val_attr, test_edges, test_attr


def is_in(edge_pair, edge_attr, edges, attr):
    if len(attr) == 0:
        return False
    row, col = edges
    for i in range(len(row)):
        if (edge_pair[0] == row[i] and edge_pair[1] == col[i] and edge_attr == attr[i]) or \
                (edge_pair[1] == row[i] and edge_pair[0] == col[i] and edge_attr == attr[i]):
            return True
    return False


def is_member(edge_pair, edge_attr, edge_dict):
    if edge_attr not in edge_dict.keys():
        return False
    edges = edge_dict[edge_attr]
    row, col = edges
    for i in range(len(row)):
        if (edge_pair[0] == row[i] and edge_pair[1] == col[i]) or \
                (edge_pair[1] == row[i] and edge_pair[0] == col[i]):
            return True
    return False


def edge_dict(edges, attr):
    out_dict = {}
    for i in range(len(attr)):
        if attr[i] not in out_dict.keys():
            out_dict[attr[i]] = [[edges[0][i]], [edges[1][i]]]
        else:
            out_dict[attr[i]][0].append(edges[0][i])
            out_dict[attr[i]][1].append(edges[1][i])
    return out_dict


def negative_generator(positive_edges, all_attr, train_edges, train_attr,
                       val_edges, val_attr, test_edges, test_attr):
    # TODO need to redo the input of train attr
    total_nodes = 645
    positive_edges = np.array(positive_edges)
    all_attr = np.array(all_attr)
    train_edges = np.array(train_edges)
    train_attr = np.array(train_attr)
    val_edges = np.array(val_edges)
    val_attr = np.array(val_attr)
    test_edges = np.array(test_edges)
    test_attr = np.array(test_attr)

    # find the se with 500 occurrence
    pos_edge_dict = edge_dict(positive_edges, all_attr)
    print(len(list(pos_edge_dict.keys())))

    neg_train_edges_col = []
    neg_train_edges_row = []
    neg_train_attr = []
    neg_val_edges_col = []
    neg_val_edges_row = []
    neg_val_attr = []
    neg_test_edges_col = []
    neg_test_edges_row = []
    neg_test_attr = []
    neg_train_dict = {}
    neg_val_dict = {}
    neg_test_dict = {}

    for i in range(len(train_edges[0])):
        if i % 2 == 1:
            continue
        node_1 = train_edges[0][i]
        edge_attr = train_attr[i]
        # edge_attr = new_dict[sum(edge_attr)]
        node_2 = random.randint(0, total_nodes - 1)
        count = 0
        while node_1 == node_2 or is_member([node_1, node_2], edge_attr, pos_edge_dict) or \
                is_member([node_1, node_2], edge_attr, neg_train_dict):
            if count >= 10000:
                node_1 = random.randint(0, total_nodes - 1)
                count = 0
            node_2 = random.randint(0, total_nodes - 1)
            count += 1
        neg_train_edges_col.append(node_1)
        neg_train_edges_row.append(node_2)
        neg_train_edges_col.append(node_2)
        neg_train_edges_row.append(node_1)
        neg_train_attr.append(edge_attr)
        neg_train_attr.append(edge_attr)
        # construct the negative train dictionary
        if edge_attr not in neg_train_dict:
            neg_train_dict[edge_attr] = [[node_1, node_2], [node_2, node_1]]
        else:
            neg_train_dict[edge_attr][0].append(node_1)
            neg_train_dict[edge_attr][1].append(node_2)
            neg_train_dict[edge_attr][1].append(node_1)
            neg_train_dict[edge_attr][0].append(node_2)
        if len(neg_train_attr) % 10000 == 0:
            print('constructing {} pairs of negative train edges'.format(len(neg_train_attr)))
        # if len(neg_train_attr) > 5:
        #     break

    for i in range(len(val_edges[0])):
        if i % 2 == 1:
            continue
        node_1 = val_edges[0][i]
        edge_attr = val_attr[i]
        # edge_attr = new_dict[sum(edge_attr)]
        node_2 = random.randint(0, total_nodes - 1)
        count = 0
        while node_1 == node_2 or is_member([node_1, node_2], edge_attr, pos_edge_dict) or \
                is_member([node_1, node_2], edge_attr, neg_val_dict):
            if count >= 10000:
                node_1 = random.randint(0, total_nodes - 1)
                count = 0
            count += 1
            node_2 = random.randint(0, total_nodes - 1)
        neg_val_edges_col.append(node_1)
        neg_val_edges_row.append(node_2)
        neg_val_edges_col.append(node_2)
        neg_val_edges_row.append(node_1)
        neg_val_attr.append(edge_attr)
        neg_val_attr.append(edge_attr)

        if edge_attr not in neg_val_dict:
            neg_val_dict[edge_attr] = [[node_1, node_2], [node_2, node_1]]
        else:
            neg_val_dict[edge_attr][0].append(node_1)
            neg_val_dict[edge_attr][1].append(node_2)
            neg_val_dict[edge_attr][1].append(node_1)
            neg_val_dict[edge_attr][0].append(node_2)

        if len(neg_val_attr) % 10000 == 0:
            print('constructing {} pairs of negative val edges'.format(len(neg_val_attr)))
        # if len(neg_val_attr) > 5:
        #     break

    for i in range(len(test_edges[0])):
        if i % 2 == 1:
            continue
        node_1 = test_edges[0][i]
        edge_attr = test_attr[i]
        # edge_attr = new_dict[sum(edge_attr)]
        node_2 = random.randint(0, total_nodes - 1)
        count = 0
        while node_1 == node_2 or is_member([node_1, node_2], edge_attr, pos_edge_dict) or \
                is_member([node_1, node_2], edge_attr, neg_test_dict):
            if count >= 10000:
                node_1 = random.randint(0, total_nodes - 1)
                count = 0
            node_2 = random.randint(0, total_nodes - 1)
            count += 1
        neg_test_edges_col.append(node_1)
        neg_test_edges_row.append(node_2)
        neg_test_edges_col.append(node_2)
        neg_test_edges_row.append(node_1)
        neg_test_attr.append(edge_attr)
        neg_test_attr.append(edge_attr)

        if edge_attr not in neg_test_dict:
            neg_test_dict[edge_attr] = [[node_1, node_2], [node_2, node_1]]
        else:
            neg_test_dict[edge_attr][0].append(node_1)
            neg_test_dict[edge_attr][1].append(node_2)
            neg_test_dict[edge_attr][1].append(node_1)
            neg_test_dict[edge_attr][0].append(node_2)

        if len(neg_test_attr) % 10000 == 0:
            print('constructing {} pairs of negative test edges'.format(len(neg_test_attr)))
        # if len(neg_test_attr) > 5:
        #     break

    return [neg_train_edges_col, neg_train_edges_row], neg_train_attr, [neg_val_edges_col, neg_val_edges_row], neg_val_attr, [neg_test_edges_col, neg_test_edges_row], neg_test_attr


def name_to_feature(se_name_list):
    se_dict = load_se_dict('./data/decagon_data/se_embedding.json')
    out_list = []
    for se in se_name_list:
        out_list.append(se_dict[se])
    return torch.Tensor(out_list)


def read_negative():
    # this is a totally local function should be modified when published.
    neg_t_edge = []
    neg_val_edge = []
    neg_test_edge = []
    with open('neg_train_edges') as f:
        for line in f:
            line = line.rstrip()
            line = list(map(int, line.split()))
            neg_t_edge.append(line)
    with open('neg_val_edges') as f:
        for line in f:
            line = line.rstrip()
            line = list(map(int, line.split()))
            neg_val_edge.append(line)
    with open('neg_test_edges') as f:
        for line in f:
            line = line.rstrip()
            line = list(map(int, line.split()))
            neg_test_edge.append(line)

    with open('neg_train_attr') as f:
        se_name_train = []
        for line in f:
            line = line.rstrip()
            se_name_train.append(line)
        neg_train_attr = name_to_feature(se_name_train)
    with open('neg_val_attr') as f:
        se_name_val = []
        for line in f:
            line = line.rstrip()
            se_name_val.append(line)
        neg_val_attr = name_to_feature(se_name_val)
    with open('neg_test_attr') as f:
        se_name_test = []
        for line in f:
            line = line.rstrip()
            se_name_test.append(line)
        neg_test_attr = name_to_feature(se_name_test)

    return torch.LongTensor(neg_t_edge), neg_train_attr, torch.LongTensor(neg_val_edge), neg_val_attr, torch.LongTensor(neg_test_edge), neg_test_attr



# TEST

# file_name = '/data/hancwang/DDI/code/data/decagon_data/id_SMILE.txt'
# dic = process(file_name)
# # symbol_set = set()
# # for item in dic:
# #     # print(nx.get_node_attributes(dic[item], 'symbol'))
# #     # print(nx.get_edge_attributes(dic[item], 'bond_type'))
# #     symbols = nx.get_node_attributes(dic[item], 'symbol')
# #     keyss = list(symbols.keys())
# #     for symbol in symbols.values():
# #         symbol_set.add(symbol)
# #
# # print(len(symbol_set))
# # print(symbol_set)
# bond_set = set()
# for item in dic:
#     bonds = nx.get_edge_attributes(dic[item], 'bond_type')
#     print(bonds)
#     for bond in bonds.values():
#         bond_set.add(bond)
# print(len(bond_set))
# print(bond_set)
# diameter_list = []
# for item in dic:
#     try:
#         d = nx.diameter(dic[item])
#         diameter_list.append(d)
#     except:
#         print('item id')
#         print(item)
#         continue
# print(max(diameter_list))
# print(sum(diameter_list)/len(diameter_list))
#
# num_node = []
# for item in dic:
#     num_node.append(len(dic[item].nodes))
#     if len(dic[item].nodes) == 1:
#         print(item)
# print(max(num_node))
# print(sum(num_node)/len(num_node))
# print(min(num_node))
# print(sorted(num_node))

if __name__ == '__main__':
    load_data('/data/hancwang/DDI/code/data/decagon_data/id_SMILE.txt',
              './data/decagon_data/bio-decagon-combo.csv' 'onehot')
