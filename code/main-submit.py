# coding=gbk
# -*- coding: utf-8 -*-
import re
import time
import ast
import json
import copy
import sys
import numpy as np
from openai import OpenAI

import os

# 打开文件
file_path = '1376_en435_test.txt'
English = True
# file_path = '1089zh.txt'

free = 1
t = 0
if English:
    sentence_size = 60
else:
    sentence_size = 120

overlapping = 0
least_no_of_sentence = 3



with open(file_path, 'r', encoding='utf-8') as file:
    txtcontent = file.read()
    txtcontent = txtcontent.replace("\\", "")
    if txtcontent[0] == '"':
        txtcontent = txtcontent[1:]
    if txtcontent[len(txtcontent)-1] == '"':
        txtcontent = txtcontent[:-1]


client6 = OpenAI(
    api_key="sk-",
    base_url="https://api.moonshot.cn/v1",
)
# https://platform.moonshot.cn/console/account

####################


if free == 1:
    time.sleep(t)

# sys.exit()   #####

final_txt = "{"


def iou(box1, box2):
    try:
        a1, b1 = box1
        a2, b2 = box2

        # 计算交集的坐标
        inter_left = max(a1, a2)
        inter_right = min(b1, b2)

        # 计算交集的面积
        inter_area = max(0, inter_right - inter_left)

        # 计算并集的面积
        union_area = (b1 - a1) + (b2 - a2) - inter_area

        # 计算IOU
        IOU = inter_area / union_area if union_area != 0 else 0

    except Exception as e:
        print("iou not succesfull")
        print(box1)
        print(box2)
        IOU = 0


    return copy.deepcopy(IOU)

def to_jason(item_name, data):
    data = data.replace("'", '"')

    if English:
        class_1 = "Opening"
        class_2 = "Order Urging"
        class_3 = "Price"
        class_4 = "Product Description"
    else:
        class_1 = "商品开场白"
        class_2 = "催促下单"
        class_3 = "商品价格"
        class_4 = "商品介绍和卖点描述"


    if class_4 not in data:
        data += class_4 + ": - []"
    if class_3 not in data:
        insert_string = class_3 + ": - []"
        index = data.find(class_4)
        data = data[:index] + insert_string + data[index:]
    if class_2 not in data:
        insert_string = class_2 + ": - []"
        index = data.find(class_3)
        data = data[:index] + insert_string + data[index:]
    if class_1 not in data:
        insert_string = class_1 + ": - []"
        data = insert_string + data

    KaiChang = data[data.find(class_1):data.find(class_2)]
    YinDao = data[data.find(class_2):data.find(class_3)]
    JiaGe = data[data.find(class_3):data.find(class_4)]
    Miaoshu = data[data.find(class_4):]

    # print("KaiChang =", KaiChang)

    # print("KaiChang =", KaiChang)
    # print("YinDao =", YinDao)
    # print("JiaGe =", JiaGe)
    # print("Miaoshu =", Miaoshu)

    # print("JiaGe =", JiaGe)

    # 正则表达式匹配项
    pattern = re.compile(r'\[(\d+), "(.*?)"\]')


    # 解析文本
    items_KaiChang = pattern.findall(KaiChang)
    items_YinDao = pattern.findall(YinDao)
    items_Miaoshu = pattern.findall(Miaoshu)
    items_JiaGe = pattern.findall(JiaGe)

    # print("items_KaiChang =", items_KaiChang)

    Items_tasks = [["商品开场", items_KaiChang], ["购买引导", items_YinDao], ["商品介绍和卖点描述", items_Miaoshu],
                   ["商品价格", items_JiaGe]]

    # 构建字典
    parsed_data = {
        "商品名称": item_name,
        "商品开场": [],
        "购买引导": [],
        "商品介绍和卖点描述": [],
        "商品价格": []
    }

    # 填充数据
    for task, my_item in Items_tasks:
        for item in my_item:
            id, content = item

            parsed_data[task].append({"id": id, "content": content})

    json_data = json.dumps(parsed_data, ensure_ascii=False, indent=4)
    json_data = json_data[1:-1]
    print("json_data1 =\n", json_data)

    if items_KaiChang == [] and items_YinDao == [] and items_Miaoshu == [] and items_JiaGe == []:
        json_data = ""
    return json_data


def relationship_checker(a):
    # 函数体开始
    if a == []:
        return copy.deepcopy(a)

    a = sorted(a, key=lambda x: x[1][1])
    print("sorted =", a)

    delete_index = []

    for item_index, the_item in enumerate(a[:-1]):

        index = the_item[1]
        this_name = the_item[0]
        this_start = index[0]
        this_end = index[1]

        next_item = a[item_index + 1]
        index = next_item[1]
        next_name = next_item[0]
        next_start = index[0]
        next_end = index[1]

        if this_end > next_start:
            if free == 1:
                time.sleep(t)

            if English:
                question = "Are '" + this_name + "' and '" + next_name + "' the same product on eBay？Please answer by 'Yes, they are' or 'No, they aren't'. Do not output any other content."
            else:
                question = "“" + this_name + "”和“" + next_name + "”在淘宝中属于同一商品吗？请用“是。”或者“否。”回答"



            completion = client6.chat.completions.create(
                model="moonshot-v1-8k",
                messages=[
                    {"role": "user",
                     "content": question}
                ],
                temperature=0.0,
            )

            result = completion.choices[0].message.content
            print("result 5 =", result)

            if "否" in result or "No" in result and iou([this_start, this_end], [next_start, next_end]) < 0.5:
                print("不属于")
                if this_end - this_start < least_no_of_sentence:
                    delete_index.append(item_index)
            else:
                print("属于同一类")
                if free == 1:
                    time.sleep(t)

                if English:
                    question = "Please combine '" + this_name + "' and '" + next_name + "' into one item, remember not to output any other content"
                else:
                    question = "请把“" + this_name + "”和“" + next_name + '”合并为一个商品并输出成["{合并后的限定性商品名（越具体越好）}"]，注意不要输出其他内容'


                completion = client6.chat.completions.create(
                    model="moonshot-v1-8k",
                    messages=[
                        {"role": "user",
                         "content": question}
                    ],
                    temperature=0.0,
                )

                result = completion.choices[0].message.content
                # result = ast.literal_eval(result)
                result = result.replace("[", "").replace("]", "").replace('"', "").replace("'", "")
                print("合并后的名称=", result)

                a[item_index + 1] = [result, [min(this_start, next_start), max(this_end, next_end)]]
                print("a1=", a)
                delete_index.append(item_index)
                print("item_index=", item_index)
                print("delete_index=", delete_index)
        elif this_end - this_start < least_no_of_sentence:
            delete_index.append(item_index)

    print("a2=", a)
    if delete_index:
        delete_index = list(set(delete_index))
        print("00")
        delete_index.sort(reverse=True)  # 将索引排序，从大到小
        for i in delete_index:
            print("11")
            del a[i]

    last_index = a[-1][1]
    last_start = last_index[0]
    last_end = last_index[1]

    if last_end - last_start < least_no_of_sentence:
        a = a[:-1]

    print("a3=", a)
    return copy.deepcopy(a)


list_content = ast.literal_eval(txtcontent)
Sentence_list = []
Item_list = []
while 1:

    # list_content = ast.literal_eval(txtcontent)
    # print(list_content)
    slice_list = list_content[:sentence_size]

    last_num = slice_list[-1][0]
    # print(slice_list)

    slice_txt = json.dumps(slice_list, ensure_ascii=False)
    # print(slice_txt)

    print("slice_txt=", slice_txt)

    start_num = list_content[0][0]
    # print(start_num)
    start_num = int(start_num)

    if free == 1:
        time.sleep(t)

    if English:
        question = "Please read sentence by sentence and determine which items are being sold in the following sentences" + "'" + slice_txt + "'" + 'Only select the items with price, and answer in the format of ["item 1: price 1 (if exist)", "item 2: price 2 (if exist)", "item 3: price 3 (if exist)"] in the order of appearance.'
    else:
        question = "请判断以下段落在销售哪些商品，从头到尾一句一句地阅读" + "“" + slice_txt + "”" + '只将有价格的商品名称按照出现顺序输出成一个["限定性商品名1（越具体越好）：商品1的价格", "限定性商品名2（越具体越好）：商品2的价格", "限定性商品名3（越具体越好）：商品3的价格"]列表，而删除未出现价格的商品'

    completion = client6.chat.completions.create(
        model="moonshot-v1-8k",
        messages=[
            {"role": "user",
             "content": question}
        ],
        temperature=0.0,
    )

    result = completion.choices[0].message.content
    print("result=", result)

    contains_bracket = '[' in result

    try:
        print("is a list")

        # 找到第一个左方括号和最后一个右方括号的位置
        first_bracket_index = result.find('[')
        last_bracket_index = result.rfind(']')

        # 如果找到了这两个括号，提取它们之间的内容
        if first_bracket_index != -1 and last_bracket_index != -1:
            # 确保最后一个右方括号在第一个左方括号之后
            if last_bracket_index > first_bracket_index:
                retained_content = result[first_bracket_index:last_bracket_index + 1]
                print(retained_content)  # 输出: 需要保留的内容
            else:
                print("未找到有效的方括号配对")
        else:
            print("未找到方括号")

        item_list = ast.literal_eval(retained_content)
        item_list = [re.split(r'[：:]', item)[0] for item in item_list]
        print(item_list)

    except:
        print("is not a list")
        temp = re.split(r'[：:-]', result)
        print(temp)
        # 使用列表推导式保留奇数索引的元素
        item_list = []
        for index, element in enumerate(temp):
            if index % 2 == 0 and index > 0:
                item_list.append(element)

    seen = set()
    item_list = [x for x in item_list if not (x in seen or seen.add(x))]
    item_list = [item for item in item_list if 'product not specified' not in item and "unspecified item" not in item]

    ##############

    Index_list = []
    print("item_list=", item_list)

    for the_item in item_list:

        if free == 1:
            time.sleep(t)

        completion = client6.chat.completions.create(
            model="moonshot-v1-8k",
            messages=[
                {"role": "user",
                 "content": "请从前往后一句一句地读并摘抄出以下段落中介绍" + the_item + "的第一句的序号，以及从后往前一句一句地读并摘抄出以下段落中介绍" + the_item + "的最后一句的序号" + "“" + slice_txt + "”" + "请从前往后一句一句地读并摘抄出以上段落中介绍" + the_item + "的第一句的序号，以及从后往前一句一句地读并摘抄出以上段落中介绍" + the_item + "的最后一句的序号" + "并用[第一句的序号, 最后一句的序号]来回答，注意不要输出序号以外的其他内容"}
            ],
            temperature=0.0,
        )

        result = completion.choices[0].message.content
        result = ast.literal_eval(result)
        temp = [the_item]
        temp.append(result)
        print("temp=", temp)
        # '''
        Index_list.append(temp)
        print("Index_list=", Index_list)
        # '''

    # Index_list=[[' 小米布童鞋', '[21974, 21986]'], [' 腰带', '[22027, 22033]'], [' 罗宾汉的老爹鞋', '[22031, 22040]'], [' 帆布包', '[22101, 22204]'], [' 洞洞鞋', '[22128, 22145]'], [' 渔夫帽', '[22168, 22208]']]


    End_index_list = []
    ####################

    #####
    Index_list = relationship_checker(Index_list)
    print("Index_list =", Index_list)
    print("relationship checked")

    for item_index, the_item in enumerate(Index_list[:]):

        print("the_item[0]=", the_item[0])
        # final_txt += "商品名称：" + the_item[0] + "--------------------------------\n"

        index = the_item[1]  #######################
        start = index[0]
        print('start=')
        print(start)

        end = index[1]
        End_index_list.append(end)

        # print("txtcontent=", txtcontent)
        print("Index_list=", Index_list)
        print("item_index + 1=", item_index + 1)

        # if item_index != 0:
        #     prev_item = Index_list[item_index - 1]
        #     index = prev_item[1]
        #     end_prev=index[1]
        #
        #     if end_prev > start:
        #         start = end_prev+1

        if item_index != len(Index_list) - 1:

            next_item = Index_list[item_index + 1]
            index = next_item[1]
            start_next = index[0]

        else:
            start_next = 9999
            print("00000")

        # end = max(end + overlapping, start_next - 1 + overlapping)                                #################
        if end < start_next:
            # end = int((end + overlapping + start_next - 1 + overlapping)/2)
            end = max(end + overlapping, start_next - 1 + overlapping)

        print('start=', start)

        print('end=', end)

        break1 = 0
        break2 = 0
        for my_index, my_item in enumerate(list_content):  #
            if start in my_item and break1 == 0:
                print(f"开始元素 {start} 出现在第 {my_index} 个元素中")
                start_index = my_index
                break1 = 1

            if end in my_item and break2 == 0:
                print(f"结束元素 {end} 出现在第 {my_index} 个元素中")
                end_index = my_index
                break2 = 1

            if break1 == 1 and break2 == 1:
                break
        else:
            if break1 == 0:
                start_index = 0
            if break2 == 0:
                end_index = len(list_content) - 1

            print(f"开始元素 {start}")
            print(f"结束元素 {end} ")
            print("元素在列表中不存在")
            # time.sleep(1)

        if start_index - 2 >= 0:
            sentence = list_content[start_index - 2: end_index + 1 + 2]
        else:
            sentence = list_content[0: end_index + 1 + 2]

        print("the_item=", the_item)


        print("sentence =", sentence)
        print("item =", the_item[0])
        Sentence_list.append(sentence)
        Item_list.append(the_item[0])

###

        print("item_index=", item_index)
        print(len(Index_list) - 1 - 1)

        if len(list_content) > sentence_size and item_index == len(Index_list) - 1 - 1:
            print("are you here?")
            break  ### for loop
        else:
            print("qqqqqqqqqqqqqqqqqqqqqqqqqqqqqq")

    print("------------------------here------------------------")
    if len(list_content) > sentence_size:
        print("进来了")
        # print("txtcontent=", txtcontent)

        # if start_next < end:
        #     start_next = end+1
        if item_list != []:
            End_index_list.append(start_next)
            start_next = max(End_index_list)
        else:
            start_next = last_num

        print("start_next=", start_next)

        for i, j in enumerate(list_content):
            if j[0] == start_next:
                index = i

        print("index=", index)

        if index >= 0:
            list_content = list_content[index:]  # 从找到的位置开始截取到字符串末尾

            # print("txtcontent=", txtcontent)
        else:
            print("未找到子字符串")

    else:
        break  ### while loop



unique_Item_list = []
unique_Sentence_list = []
for index, element in enumerate(Item_list):
    if element not in unique_Item_list:
        unique_Item_list.append(element)
        unique_Sentence_list.append(Sentence_list[index])
    else:
        print(element, "repeated")
        find_index = unique_Item_list.index(element)
        unique_Sentence_list[find_index].append(Sentence_list[index])
        print(unique_Sentence_list[find_index])

Item_list = copy.deepcopy(unique_Item_list)
Sentence_list = copy.deepcopy(unique_Sentence_list)

for my_index, my_item in enumerate(Item_list):

    if free == 1:
        time.sleep(t)

    if English:
        question = "Please extract the sentences related to " + my_item + " from the following document through 4 categories: (1) Opening, (2) Order Urging, (3) Price, (4) Product Description." + str(Sentence_list[my_index]) + "Remember to extract the sentences related to " + \
                   my_item + "and place them into 4 categories: (1) Opening:, (2) Order Urging:, (3) Price:, (4) Product Description:, and answer in the format of - [sentence index, sentence]"
    else:
        question = "请通过4个类别：**（1）商品开场白**、**（2）催促下单**、**（3）商品价格**、**（4）商品介绍和卖点描述**，对以下文档与" + my_item + "有关的部分进行摘抄" + "“" + str(Sentence_list[my_index]) + "”" + '，注意通过4个类别：**（1）商品开场白**、**（2）催促下单**、**（3）商品价格**、**（4）商品介绍和卖点描述**，对以上文档与' + my_item + '有关的部分进行摘抄，以- [序号, 句子]的格式逐行归类并摘抄在4个类别下'

    print("question =", question)
    completion = client6.chat.completions.create(
        model="moonshot-v1-8k",
        messages=[
            {"role": "user",
             "content": question}
        ],
        temperature=0.0,
    )

    result = completion.choices[0].message.content

    star_index = result.find('*')
    bracket_index = result.rfind(']')

    # 检查星号和右方括号是否存在
    if star_index != -1 and bracket_index != -1:
        # 截取从星号开始到右方括号结束的子字符串
        result = result[star_index: bracket_index + 1]
    else:
        print("星号或右方括号不存在，或星号在右方括号之后")

    print("sentence = ", str(Sentence_list[my_index]))
    print("result_txt = \n", result)
    final_txt += to_jason(my_item, result)


final_txt += "}"

print("final_txt=", final_txt)

# 使用 open() 函数打开文件，'w' 表示写入模式
with open('result_1376_en_0801.txt', 'w', encoding='utf-8') as file:
    # 使用 write() 方法写入字符串
    file.write(final_txt)



