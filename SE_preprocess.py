from bert_serving.client import BertClient
import csv
import json


def sample_pooling(vec):
    new_vec = []
    for i in range(128):
        new_vec.append(sum(vec[i*8:i*8+7]))
    return new_vec

file_path = './data/decagon_data/'

bc = BertClient()
SE_set = set()
with open(file_path + 'bio-decagon-combo.csv') as csv_file:
    sreader = csv.reader(csv_file, delimiter=',')
    next(sreader, None)
    for line in sreader:
        SE_set.add(line[3])

se_emb = dict()
for se in SE_set:
    # print(len(bc.encode([se])[0]))
    se_emb[se] = sample_pooling(bc.encode([se])[0])


with open(file_path+'se_embedding.json', 'w') as outfile:
    json.dump(se_emb,outfile,ensure_ascii=False)
    outfile.write('\n')

