import numpy as np
from sklearn import preprocessing
from data_load import data_load
from generate_cl import apriori
from generate_association import association_rules

if __name__ == '__main__':
    data = data_load()
    L, support_data = apriori(data, min_support=0.3)
    association_rules = association_rules(L, support_data, min_conf=0.7)
    # print(L)
    # print(support_data)
    ass_len = len(association_rules)
    for i in range (ass_len):
        print(association_rules[i])