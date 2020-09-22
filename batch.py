import torch


def batch_feed(batch_size, batch_num, train_edges, train_attr, neg_edges, neg_attr):
    # note that the batch_num should start from 0
    last_epoch = False
    if batch_size * (batch_num + 1) > train_attr.size(0):
        start_index = train_attr.size(0) - batch_size
        end_index = train_attr.size(0)
        last_epoch = True
    else:
        start_index = batch_size * batch_num
        end_index = batch_size * (batch_num + 1)
    col, row = train_edges
    train_col = col[start_index: end_index]
    train_row = row[start_index: end_index]
    n_col, n_row = neg_edges
    neg_col = n_col[start_index: end_index]
    neg_row = n_row[start_index: end_index]
    batch_col = torch.cat((train_col, neg_col), dim=0)
    batch_row = torch.cat((train_row, neg_row), dim=0)
    # neg_edge_batch = torch.cat((torch.unsqueeze(neg_col, 0), torch.unsqueeze(neg_row, 0)), dim=0)
    neg_attr_batch = neg_attr[start_index: end_index]
    # train_edge_batch = torch.cat((torch.unsqueeze(train_col, 0), torch.unsqueeze(train_row, 0)), dim=0)
    train_attr_batch = train_attr[start_index: end_index]
    batch_edges = torch.cat((torch.unsqueeze(batch_col, 0), torch.unsqueeze(batch_row, 0)), dim=0)
    batch_attr = torch.cat((train_attr_batch, neg_attr_batch), dim=0)

    return batch_edges, batch_attr, last_epoch


def graph_batch_feed(batch_num, train_edges, train_attr, neg_edges, neg_attr, train_name):
    last_epoch = False
    se_name = set()
    index = []
    for name in train_name:
        se_name.add(name)
    se_name = list(se_name)
    if batch_num == len(se_name) - 1:
        last_epoch = True
    se = se_name[batch_num]
    for i in range(len(train_name)):
        if train_name[i] == se:
            index.append(i)
    col, row = train_edges
    train_col = col[index]
    train_row = row[index]
    n_col, n_row = neg_edges
    neg_col = n_col[index]
    neg_row = n_row[index]
    train_attr_batch = train_attr[index]
    neg_attr_batch = neg_attr[index]
    batch_train_edges = torch.cat((torch.unsqueeze(train_col, 0), torch.unsqueeze(train_row, 0)), dim=0)
    batch_neg_edges = torch.cat((torch.unsqueeze(neg_col, 0), torch.unsqueeze(neg_row, 0)), dim=0)
    batch_size = len(index)

    return batch_train_edges, batch_neg_edges, train_attr_batch, neg_attr_batch, last_epoch, batch_size

