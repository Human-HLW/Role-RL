# coding=gbk
# -*- coding: utf-8 -*-

import Levenshtein as lev
import copy


def similarity(str1, str2):
    distance = lev.distance(str1, str2)
    similarity = 1 - (distance / max(len(str1), len(str2)))
    return copy.deepcopy(similarity)


the_file_1 = "lsht_sampling_10_checked.txt"
the_file_2 = "lsht_sampling_10_OLP.txt"
the_file_3 = "lsht_sampling_10_Direct.txt"
the_file_4 = "lsht_sampling_10_COT.txt"
the_file_5 = "lsht_sampling_10_Reflex.txt"
the_file_6 = "lsht_sampling_10_COA.txt"
the_file_7 = "lsht_sampling_10_LA.txt"
the_file_8 = "lsht_sampling_10_OLP_Role-RL.txt"
with open(the_file_1, 'r', encoding='utf-8') as file:
    txtcontent_1 = file.read()
with open(the_file_2, 'r', encoding='utf-8') as file:
    txtcontent_2 = file.read()
with open(the_file_3, 'r', encoding='utf-8') as file:
    txtcontent_3 = file.read()
    txtcontent_3 = txtcontent_3.replace("-", "")
with open(the_file_4, 'r', encoding='utf-8') as file:
    txtcontent_4 = file.read()
    txtcontent_4 = txtcontent_4.replace("-", "")
with open(the_file_5, 'r', encoding='utf-8') as file:
    txtcontent_5 = file.read()
    txtcontent_5 = txtcontent_5.replace("-", "")
with open(the_file_6, 'r', encoding='utf-8') as file:
    txtcontent_6 = file.read()
    txtcontent_6 = txtcontent_6.replace("-", "")
with open(the_file_7, 'r', encoding='utf-8') as file:
    txtcontent_7 = file.read()
    txtcontent_7 = txtcontent_7.replace("-", "")
with open(the_file_8, 'r', encoding='utf-8') as file:
    txtcontent_8 = file.read()
    txtcontent_8 = txtcontent_8.replace("-", "")
print("--------------------------")
print("lsht_sampling_10 OLP simu =", similarity(txtcontent_1, txtcontent_2))
print("lsht_sampling_10 dir simu =", similarity(txtcontent_1, txtcontent_3))
print("lsht_sampling_10 COT simu =", similarity(txtcontent_1, txtcontent_4))
print("lsht_sampling_10 Reflex simu =", similarity(txtcontent_1, txtcontent_5))
print("lsht_sampling_10 COA simu =", similarity(txtcontent_1, txtcontent_6))
print("lsht_sampling_10 LA simu =", similarity(txtcontent_1, txtcontent_7))
print("lsht_sampling_10 OLP+Role-RL simu =", similarity(txtcontent_1, txtcontent_8))
#


the_file_1 = "lsht_sampling_20_checked.txt"
the_file_2 = "lsht_sampling_20_OLP.txt"
the_file_3 = "lsht_sampling_20_Direct.txt"
the_file_4 = "lsht_sampling_20_COT.txt"
the_file_5 = "lsht_sampling_20_Reflex.txt"
the_file_6 = "lsht_sampling_20_COA.txt"
the_file_7 = "lsht_sampling_20_LA.txt"
the_file_8 = "lsht_sampling_20_OLP_Role-RL.txt"
with open(the_file_1, 'r', encoding='utf-8') as file:
    txtcontent_1 = file.read()
with open(the_file_2, 'r', encoding='utf-8') as file:
    txtcontent_2 = file.read()
with open(the_file_3, 'r', encoding='utf-8') as file:
    txtcontent_3 = file.read()
    txtcontent_3 = txtcontent_3.replace("-", "")
with open(the_file_4, 'r', encoding='utf-8') as file:
    txtcontent_4 = file.read()
    txtcontent_4 = txtcontent_4.replace("-", "")
with open(the_file_5, 'r', encoding='utf-8') as file:
    txtcontent_5 = file.read()
    txtcontent_5 = txtcontent_5.replace("-", "")
with open(the_file_6, 'r', encoding='utf-8') as file:
    txtcontent_6 = file.read()
    txtcontent_6 = txtcontent_6.replace("-", "")
with open(the_file_7, 'r', encoding='utf-8') as file:
    txtcontent_7 = file.read()
    txtcontent_7 = txtcontent_7.replace("-", "")
with open(the_file_8, 'r', encoding='utf-8') as file:
    txtcontent_8 = file.read()
    txtcontent_8 = txtcontent_8.replace("-", "")
print("--------------------------")
print("lsht_sampling_20 OLP simu =", similarity(txtcontent_1, txtcontent_2))
print("lsht_sampling_20 dir simu =", similarity(txtcontent_1, txtcontent_3))
print("lsht_sampling_20 COT simu =", similarity(txtcontent_1, txtcontent_4))
print("lsht_sampling_20 Reflex simu =", similarity(txtcontent_1, txtcontent_5))
print("lsht_sampling_20 COA simu =", similarity(txtcontent_1, txtcontent_6))
print("lsht_sampling_20 LA simu =", similarity(txtcontent_1, txtcontent_7))
print("lsht_sampling_20 OLP+Role-RL simu =", similarity(txtcontent_1, txtcontent_8))
# #

#
the_file_1 = "lsht_sampling_30_checked.txt"
the_file_2 = "lsht_sampling_30_OLP.txt"
the_file_3 = "lsht_sampling_30_Direct.txt"
the_file_4 = "lsht_sampling_30_COT.txt"
the_file_5 = "lsht_sampling_30_Reflex.txt"
the_file_6 = "lsht_sampling_30_COA.txt"
the_file_7 = "lsht_sampling_30_LA.txt"
the_file_8 = "lsht_sampling_30_OLP_Role-RL.txt"
with open(the_file_1, 'r', encoding='utf-8') as file:
    txtcontent_1 = file.read()
with open(the_file_2, 'r', encoding='utf-8') as file:
    txtcontent_2 = file.read()
with open(the_file_3, 'r', encoding='utf-8') as file:
    txtcontent_3 = file.read()
    txtcontent_3 = txtcontent_3.replace("-", "")
with open(the_file_4, 'r', encoding='utf-8') as file:
    txtcontent_4 = file.read()
    txtcontent_4 = txtcontent_4.replace("-", "")
with open(the_file_5, 'r', encoding='utf-8') as file:
    txtcontent_5 = file.read()
    txtcontent_5 = txtcontent_5.replace("-", "")
with open(the_file_6, 'r', encoding='utf-8') as file:
    txtcontent_6 = file.read()
    txtcontent_6 = txtcontent_6.replace("-", "")
with open(the_file_7, 'r', encoding='utf-8') as file:
    txtcontent_7 = file.read()
    txtcontent_7 = txtcontent_7.replace("-", "")
with open(the_file_8, 'r', encoding='utf-8') as file:
    txtcontent_8 = file.read()
    txtcontent_8 = txtcontent_8.replace("-", "")
print("--------------------------")
print("lsht_sampling_30 OLP simu =", similarity(txtcontent_1, txtcontent_2))
print("lsht_sampling_30 dir simu =", similarity(txtcontent_1, txtcontent_3))
print("lsht_sampling_30 COT simu =", similarity(txtcontent_1, txtcontent_4))
print("lsht_sampling_30 Reflex simu =", similarity(txtcontent_1, txtcontent_5))
print("lsht_sampling_30 COA simu =", similarity(txtcontent_1, txtcontent_6))
print("lsht_sampling_30 LA simu =", similarity(txtcontent_1, txtcontent_7))
print("lsht_sampling_30 OLP+Role-RL simu =", similarity(txtcontent_1, txtcontent_8))
# #



# #
the_file_1 = "multi_news_sampling_10_checked.txt"
the_file_2 = "multi_news_sampling_10_OLP.txt"
the_file_3 = "multi_news_sampling_10_Direct.txt"
the_file_4 = "multi_news_sampling_10_COT.txt"
the_file_5 = "multi_news_sampling_10_Reflex.txt"
the_file_6 = "multi_news_sampling_10_COA.txt"
the_file_7 = "multi_news_sampling_10_LA.txt"
the_file_8 = "multi_news_sampling_10_OLP_Role-RL.txt"
with open(the_file_1, 'r', encoding='utf-8') as file:
    txtcontent_1 = file.read()
    txtcontent_1 = txtcontent_1.replace("-", "")
with open(the_file_2, 'r', encoding='utf-8') as file:
    txtcontent_2 = file.read()
    txtcontent_2 = txtcontent_2.replace("-", "")
with open(the_file_3, 'r', encoding='utf-8') as file:
    txtcontent_3 = file.read()
    txtcontent_3 = txtcontent_3.replace("-", "")
with open(the_file_4, 'r', encoding='utf-8') as file:
    txtcontent_4 = file.read()
    txtcontent_4 = txtcontent_4.replace("-", "")
with open(the_file_5, 'r', encoding='utf-8') as file:
    txtcontent_5 = file.read()
    txtcontent_5 = txtcontent_5.replace("-", "")
with open(the_file_6, 'r', encoding='utf-8') as file:
    txtcontent_6 = file.read()
    txtcontent_6 = txtcontent_6.replace("-", "")
with open(the_file_7, 'r', encoding='utf-8') as file:
    txtcontent_7 = file.read()
    txtcontent_7 = txtcontent_7.replace("-", "")
with open(the_file_8, 'r', encoding='utf-8') as file:
    txtcontent_8 = file.read()
    txtcontent_8 = txtcontent_8.replace("-", "")
print("--------------------------")
print("multi_news_sampling_10 OLP simu =", similarity(txtcontent_1, txtcontent_2))
print("multi_news_sampling_10 dir simu =", similarity(txtcontent_1, txtcontent_3))
print("multi_news_sampling_10 COT simu =", similarity(txtcontent_1, txtcontent_4))
print("multi_news_sampling_10 Reflex simu =", similarity(txtcontent_1, txtcontent_5))
print("multi_news_sampling_10 COA simu =", similarity(txtcontent_1, txtcontent_6))
print("multi_news_sampling_10 LA simu =", similarity(txtcontent_1, txtcontent_7))
print("multi_news_sampling_10 OLP+Role-RL simu =", similarity(txtcontent_1, txtcontent_8))

#


# #
the_file_1 = "multi_news_sampling_20_checked.txt"
the_file_2 = "multi_news_sampling_20_OLP.txt"
the_file_3 = "multi_news_sampling_20_Direct.txt"
the_file_4 = "multi_news_sampling_20_COT.txt"
the_file_5 = "multi_news_sampling_20_Reflex.txt"
the_file_6 = "multi_news_sampling_20_COA.txt"
the_file_7 = "multi_news_sampling_20_LA.txt"
the_file_8 = "multi_news_sampling_20_OLP_Role-RL.txt"
with open(the_file_1, 'r', encoding='utf-8') as file:
    txtcontent_1 = file.read()
with open(the_file_2, 'r', encoding='utf-8') as file:
    txtcontent_2 = file.read()
with open(the_file_3, 'r', encoding='utf-8') as file:
    txtcontent_3 = file.read()
    txtcontent_3 = txtcontent_3.replace("-", "")
with open(the_file_4, 'r', encoding='utf-8') as file:
    txtcontent_4 = file.read()
    txtcontent_4 = txtcontent_4.replace("-", "")
with open(the_file_5, 'r', encoding='utf-8') as file:
    txtcontent_5 = file.read()
    txtcontent_5 = txtcontent_5.replace("-", "")
with open(the_file_6, 'r', encoding='utf-8') as file:
    txtcontent_6 = file.read()
    txtcontent_6 = txtcontent_6.replace("-", "")
with open(the_file_7, 'r', encoding='utf-8') as file:
    txtcontent_7 = file.read()
    txtcontent_7 = txtcontent_7.replace("-", "")
with open(the_file_8, 'r', encoding='utf-8') as file:
    txtcontent_8 = file.read()
    txtcontent_8 = txtcontent_8.replace("-", "")
print("--------------------------")
print("multi_news_sampling_20 OLP simu =", similarity(txtcontent_1, txtcontent_2))
print("multi_news_sampling_20 dir simu =", similarity(txtcontent_1, txtcontent_3))
print("multi_news_sampling_20 COT simu =", similarity(txtcontent_1, txtcontent_4))
print("multi_news_sampling_20 Reflex simu =", similarity(txtcontent_1, txtcontent_5))
print("multi_news_sampling_20 COA simu =", similarity(txtcontent_1, txtcontent_6))
print("multi_news_sampling_20 LA simu =", similarity(txtcontent_1, txtcontent_7))
print("multi_news_sampling_20 OLP+Role-RL simu =", similarity(txtcontent_1, txtcontent_8))
#



# #
the_file_1 = "multi_news_sampling_30_checked.txt"
the_file_2 = "multi_news_sampling_30_OLP.txt"
the_file_3 = "multi_news_sampling_30_Dir.txt"
the_file_4 = "multi_news_sampling_30_COT.txt"
the_file_5 = "multi_news_sampling_30_Reflex.txt"
the_file_6 = "multi_news_sampling_30_COA.txt"
the_file_7 = "multi_news_sampling_30_LA.txt"
the_file_8 = "multi_news_sampling_30_OLP_Role-RL.txt"

with open(the_file_1, 'r', encoding='utf-8') as file:
    txtcontent_1 = file.read()
    txtcontent_1 = txtcontent_1.replace("-", "")
with open(the_file_2, 'r', encoding='utf-8') as file:
    txtcontent_2 = file.read()
    txtcontent_2 = txtcontent_2.replace("-", "")
with open(the_file_3, 'r', encoding='utf-8') as file:
    txtcontent_3 = file.read()
    txtcontent_3 = txtcontent_3.replace("-", "")
with open(the_file_4, 'r', encoding='utf-8') as file:
    txtcontent_4 = file.read()
    txtcontent_4 = txtcontent_4.replace("-", "")
with open(the_file_5, 'r', encoding='utf-8') as file:
    txtcontent_5 = file.read()
    txtcontent_5 = txtcontent_5.replace("-", "")
with open(the_file_6, 'r', encoding='utf-8') as file:
    txtcontent_6 = file.read()
    txtcontent_6 = txtcontent_6.replace("-", "")
with open(the_file_7, 'r', encoding='utf-8') as file:
    txtcontent_7 = file.read()
    txtcontent_7 = txtcontent_7.replace("-", "")
with open(the_file_8, 'r', encoding='utf-8') as file:
    txtcontent_8 = file.read()
    txtcontent_8 = txtcontent_8.replace("-", "")

print("--------------------------")
print("multi_news_sampling_30 OLP simu =", similarity(txtcontent_1, txtcontent_2))
print("multi_news_sampling_30 dir simu =", similarity(txtcontent_1, txtcontent_3))
print("multi_news_sampling_30 COT simu =", similarity(txtcontent_1, txtcontent_4))
print("multi_news_sampling_30 Reflex simu =", similarity(txtcontent_1, txtcontent_5))
print("multi_news_sampling_30 COA simu =", similarity(txtcontent_1, txtcontent_6))
print("multi_news_sampling_30 LA simu =", similarity(txtcontent_1, txtcontent_7))
print("multi_news_sampling_30 OLP+Role-RL simu =", similarity(txtcontent_1, txtcontent_8))


