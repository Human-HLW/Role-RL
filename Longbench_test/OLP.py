# coding=gbk
# -*- coding: utf-8 -*-
import re
import time
import ast
import json
import copy
import traceback
from openai import OpenAI
from nltk.tokenize import sent_tokenize
import requests

url = "https://api.ainewserver.com/v1/chat/completions"

headers = {
    "Authorization": "Bearer sk-xpqaolSdwUgRVyCV094cBc08A78142CaB060291f5231Ee85",
    "content-type": "application/json"
}

file_path = 'lsht_sampling_30.txt'

start_id = 21981

English = False
News = True

LLM_name = "claude-3-opus-20240229"
LLM_content_organizer = "claude-3-opus-20240229"

kimi = False
client6 = OpenAI(
    api_key="sk-VeqeY6XTu1X5l6aQJXmmt3JxcCWO5bw8sXi52XPnAHF2MTuZ",
    base_url="https://api.moonshot.cn/v1",
)

free = 1
t = 0
if English:
    sentence_size = 85
else:
    sentence_size = 45

overlapping = 0
least_no_of_sentence = 0

with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
    txt = file.read()
    print(txt)
    txt = txt[:-2]

    # 将文本分割成句子列表，这里使用句号作为分句的依据
    if English:
        txt = txt.replace("\\n", "").replace("Passage", "").replace("\\", "").replace('”', "'").replace('“', "'").replace('’', "'").replace('‘', "'").replace('"', "'").replace('NEWLINE_CHAR ', "")
        sentences = sent_tokenize(txt)

    else:
        txt = txt.replace("\\n", "").replace(" ", "")
        sentences = re.split(r'[。！？]', txt)

    result = []

    # 遍历句子列表并格式化
    for i, sentence in enumerate(sentences, start=start_id):
        trimmed_sentence = sentence.strip()  # 去除句子前后的空白字符
        # 将句子添加到结果列表中
        if English:
            result.append([i, trimmed_sentence])
        else:
            result.append([i, trimmed_sentence + "。"])

    txtcontent = str(result)
    print(result)

print("txt=", txt)
with open(file_path, 'w', encoding='utf-8', errors='ignore') as file:
    file.write(txt)

with open(file_path[:-4]+"convert"+".txt", 'w', encoding='utf-8') as file:
   file.write(txtcontent)

print("saved")
time.sleep(10)

if free == 1:
    time.sleep(t)

final_txt = ""

def iou(box1, box2):
    try:
        a1, b1 = box1
        a2, b2 = box2

        inter_left = max(a1, a2)
        inter_right = min(b1, b2)

        inter_area = max(0, inter_right - inter_left + 1)
        union_area = (b1 - a1 + 1) + (b2 - a2 + 1) - inter_area

        IOU = inter_area / union_area

    except Exception as e:
        print("iou not succesfull")
        print(box1)
        print(box2)
        IOU = 0

    return copy.deepcopy(IOU)

# Function of Relationship Checker
def relationship_checker(a):
    if a == []:
        return copy.deepcopy(a)

    a = sorted(a, key=lambda x: x[1][1])
    print("sorted =", a)

    delete_index = []

    while True:
        belong = 0
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

            if True:
                if free == 1:
                    time.sleep(t)

                if English:
                    if News:
                        question = "Are '" + this_name + "' and '" + next_name + "' related topics? Please answer by 'Yes, they are' or 'No, they aren't'. Do not output any other content."
                    else:
                        question = "Are '" + this_name + "' and '" + next_name + "' the same product on eBay？Please answer by 'Yes, they are' or 'No, they aren't'. Do not output any other content."
                else:
                    question = "“" + this_name + "”和“" + next_name + "”适合整合为同一新闻标题吗？请用“是。”或者“否。”回答，不要输出其他内容"

                if True:
                    while True:
                        try:
                            invoke = {"messages": [{"role": "user", "content": question, }], "model": LLM_name,
                                      "temperature": 0, }

                            response = requests.post(url, headers=headers, json=invoke).text
                            response = json.loads(response)

                            result = response["choices"][0]["message"]['content']
                            break
                        except Exception as e:
                            print(f"An error occurred: {e}")

                print("result 5 =", result)
                print("4 number =", this_start, next_start, this_end, next_end)

                # judge whether belong to the same item
                if ("否" in result or "No" in result) and iou([this_start, this_end], [next_start, next_end]) < 0.3:
                    print("不属于")
                    if this_end - this_start < least_no_of_sentence:
                        delete_index.append(item_index)
                        break
                else:
                    print("属于同一类")
                    belong = 1
                    if free == 1:
                        time.sleep(t)

                    # combine into one item
                    if English:
                        if News:
                            question = 'Please combine "' + this_name + '" and "' + next_name + '" into one short title and output ["(fill in the combined short title)"], remember not to output any other content'
                        else:
                            question = "Please combine '" + this_name + "' and '" + next_name + "' into one item, remember not to output any other content"
                    else:
                        question = "请把“" + this_name + "”和“" + next_name + "”合并为一个新闻事件并输出成['填入合并后的事件名']，注意不要输出其他内容"

                    if True:
                        while True:
                            try:
                                invoke = {"messages": [{"role": "user", "content": question, }], "model": LLM_name,
                                          "temperature": 0, }

                                response = requests.post(url, headers=headers, json=invoke).text
                                response = json.loads(response)

                                result = response["choices"][0]["message"]['content']
                                break
                            except Exception as e:
                                print(f"An error occurred: {e}")

                    result = result.replace("[", "").replace("]", "").replace('"', "").replace("'", "")
                    print("合并后的名称=", result)

                    print("4 number =", this_start, next_start, this_end, next_end)
                    a[item_index + 1] = [result, [min(this_start, next_start), max(this_end, next_end)]]
                    print("a1=", a)
                    delete_index.append(item_index)
                    print("item_index=", item_index)
                    print("delete_index=", delete_index)
                    break


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

        if belong == 0:
            break

    return copy.deepcopy(a)


list_content = ast.literal_eval(txtcontent)
Sentence_list = []
Item_list = []

while 1:

    slice_list = list_content[:sentence_size]

    last_num = slice_list[-1][0]

    slice_txt = json.dumps(slice_list, ensure_ascii=False)

    print("slice_txt=", slice_txt)

    start_num = list_content[0][0]

    start_num = int(start_num)

    if free == 1:
        time.sleep(t)

    # Topic Finder
    if English:
        if News:
            question = "Please determine the news topics in the following paragraph, read each sentence from beginning to end, and when several topics have an inclusive relationship, only output the parent category topic " + "'" + slice_txt + "'" + '\nRead from beginning to end, sentence by sentence, and in the context of the surrounding text, determine what news topics are present in the above paragraph. There are often clear jumps between different topics. Then answer in the format of ["(fill in short title 1)", "(fill in short title 2)", "(fill in short title 3)"]. When several topics have an inclusive relationship, only output the parent topic. Remember to output in the list format and do not output any content other than the topic list.'
        else:
            question = "Please read sentence by sentence and determine which items are being sold in the following sentences" + "'" + slice_txt + "'" + '. Only select the items with price, and answer in the format of ["item 1: price 1 (if exist)", "item 2: price 2 (if exist)", "item 3: price 3 (if exist)"] in the order of appearance.'
    else:
        question = "请判断以下段落有哪些新闻主题，从头到尾一句一句地阅读，当几个标题具有包含关系时，只输出父类标题" + "“" + slice_txt + "”" + "从头到尾一句一句地阅读，并联系上下文，判断以上段落有哪些新闻主题，不同主题之间往往有明显的跳跃，并将主题输出成['填入主题1'，'填入主题2'，'填入主题3']的列表形式，当几个标题具有包含关系时，只输出父类标题，注意输出列表形式，并且不要输出除标题列表外的其它任何内容"

    if True:
        while True:
            try:
                invoke = {"messages": [{"role": "user", "content": question, }], "model": LLM_name,
                          "temperature": 0, }

                response = requests.post(url, headers=headers, json=invoke).text
                response = json.loads(response)

                result = response["choices"][0]["message"]['content']
                if "[" in result:
                    break
            except Exception as e:
                print(f"An error occurred: {e}")

    print("result=", result)

    contains_bracket = '[' in result

    try:
        print("is a list")

        first_bracket_index = result.find('[')
        last_bracket_index = result.rfind(']')

        if first_bracket_index != -1 and last_bracket_index != -1:

            if last_bracket_index > first_bracket_index:
                retained_content = result[first_bracket_index:last_bracket_index + 1]
                retained_content.replace("\n", "").replace("\r", "")
                print(retained_content)
            else:
                print("未找到有效的方括号配对")
        else:
            print("未找到方括号")

        result = retained_content.strip()

        item_list = ast.literal_eval(str(result))

        print(item_list)

    except Exception as e:

        print(f"捕获到异常：{e}")
        print(f"异常类型：{type(e)}")
        print(f"堆栈跟踪：{traceback.format_exc()}")

        temp = re.split(r'[：:-]', result)
        print(temp)

        item_list = []
        for index, element in enumerate(temp):
            if index % 2 == 0 and index > 0:
                item_list.append(element)

    #  item list
    seen = set()
    item_list = [x for x in item_list if not (x in seen or seen.add(x))]
    item_list = [item for item in item_list if 'product not specified' not in item and "unspecified item" not in item]

    Index_list = []
    print("item_list=", item_list)

    for the_item in item_list:

        if free == 1:
            time.sleep(t)

        # Topic Locator
        question = "请从前往后一句一句地读并摘抄出以下段落中介绍" + the_item + "及其子类的第一句的序号，以及从后往前一句一句地读并摘抄出以下段落中介绍" + the_item + "及其子类的最后一句的序号，" + "“" + slice_txt + "”" + "请从前往后一句一句地读并摘抄出以上段落中介绍" + the_item + "及其子类的第一句的序号，以及从后往前一句一句地读并摘抄出以上段落中介绍" + the_item + "及其子类的最后一句的序号" + "说出你的思路，最后用一个列表[第一句的序号, 最后一句的序号]来回答"

        if True:
            while True:
                try:
                    invoke = {"messages": [{"role": "user", "content": question, }], "model": LLM_name,
                              "temperature": 0, }

                    response = requests.post(url, headers=headers, json=invoke).text
                    response = json.loads(response)

                    result = response["choices"][0]["message"]['content']
                    break
                except Exception as e:
                    print(f"An error occurred: {e}")

        result = result.strip()
        matches = re.findall(r'\[(.*?)\]', result)

        if matches:
            result = f"[{matches[-1]}]"
            print(result)
        else:
            print("No matches found.")

        result = ast.literal_eval(result)
        if result[0] > result[1]:
            result.reverse()

        temp = [the_item]
        temp.append(result)
        print("temp=", temp)
        #
        Index_list.append(temp)
        print("Index_list=", Index_list)

    End_index_list = []

    Index_list = relationship_checker(Index_list)
    print("Index_list =", Index_list)
    print("relationship checked")

    for item_index, the_item in enumerate(Index_list[:]):

        print("the_item[0]=", the_item[0])

        index = the_item[1]  ##
        start = index[0]
        print('start=')
        print(start)

        end = index[1]
        End_index_list.append(end)

        print("Index_list=", Index_list)
        print("item_index + 1=", item_index + 1)

        if item_index != len(Index_list) - 1:
            next_item = Index_list[item_index + 1]
            index = next_item[1]
            start_next = index[0]
        else:
            end = slice_list[-1][0]
            print("end =", end)
            start_next = end
            print("99999999")

        if item_index != 0:
            last_item = Index_list[item_index - 1]
            index = last_item[1]
            end_last = index[1]
        else:
            end_last = 0
            print("00000")

        if end < start_next:
            end = max(end + overlapping, start_next - 1 + overlapping)

        if start > end_last:
            start = min(start - overlapping, end_last + 1 - overlapping)

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

        overlap = 0

        if start_index - overlap >= 0:
            sentence = list_content[start_index - overlap: end_index + 1 + overlap]
        else:
            sentence = list_content[0: end_index + 1 + overlap]

        print("the_item=", the_item)

        print("sentence =", sentence)
        print("item =", the_item[0])

        Sentence_list.append(sentence)  # Sentence_list
        Item_list.append(the_item[0])

        print("item_index=", item_index)
        print(len(Index_list) - 1 - 1)

        if len(list_content) > sentence_size and item_index == len(Index_list) - 1 - 1:
            print("are you here?")
            break  ### for loop
        else:
            print("qqq")

    print("------------------------here------------------------")
    if len(list_content) > sentence_size:
        print("进来了")

        if item_list != []:
            End_index_list.append(start_next)
            start_next = max(End_index_list)
        else:
            start_next = last_num

        print("start_next=", start_next)

        print("list_content =", list_content)
        for i, j in enumerate(list_content):
            if j[0] == start_next:
                index = i

        print("index=", index)

        if index >= 0:
            list_content = list_content[index:]

        else:
            print("未找到子字符串")

    else:
        print("haha")
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

    # Content Organizer
    if English:
        if News:
            question = "Please place EACH AND EVERY sentence from the first to the last sentences in any subtle relationship to '" + my_item + "' and things associated with it into the most appropriate ONE of the four entries in UNCHANGED sequence: '(1) Future Plans: (2) Assumptions: (3) Opinions: (4) Facts: ', from the following document " + '"' + str(Sentence_list[my_index]) + '"' + " Remember to place EACH AND EVERY sentence from the first to the last sentences in any subtle relationship to '" + my_item + "' and things associated with it into the most appropriate ONE of the 4 entries in UNCHANGED sequence: '(1) Future Plans: (2) Assumptions: (3) Opinions: (4) Facts: ', in the original list format of [sentence index, sentence] WITHOUT ANY REPETITION OR OMISSION"
        else:
            question = "Please extract the sentences related to '" + my_item + "' from the following document through 4 categories: (1) Opening, (2) Order Urging, (3) Price, (4) Product Description." + str(
                Sentence_list[my_index]) + "Remember to extract the sentences related to '" + \
                       my_item + "' and place them into 4 categories: (1) Opening:, (2) Order Urging:, (3) Price:, (4) Product Description:, and answer in the format of [sentence index, sentence]"
    else:
        question = "请通过”(1) 未来计划”、“(2) 猜测”、“(3) 观点”、“(4) 事实”4个类别，对以下文档与" + my_item + "有关的部分进行摘抄" + "“" + str(Sentence_list[my_index]) + "”" + '，注意通过”(1) 未来计划”、“(2) 猜测”、“(3) 观点”、“(4) 事实”4个类别，对以上文档与' + my_item + '有关的部分进行逐句摘抄（仅仅摘抄而不要输出其他内容），注意以[序号, 句子]的格式，不重复、不遗漏地归类并摘抄在4个类别下'

    print("question =", question)

    if True:
        while True:
            try:
                invoke = {"messages": [{"role": "user", "content": question, }], "model": LLM_name,
                          "temperature": 0, }

                response = requests.post(url, headers=headers, json=invoke).text
                response = json.loads(response)

                result = response["choices"][0]["message"]['content']
                break
            except Exception as e:
                print(f"An error occurred: {e}")

    star_index = result.find('(')
    bracket_index = result.rfind(']')

    if star_index != -1 and bracket_index != -1:
        result = result[star_index: bracket_index + 1]
    else:
        print("星号或右方括号不存在，或星号在右方括号之后")

    print("sentence = ", str(Sentence_list[my_index]))
    print("result_txt = \n", result)
    final_txt += my_item
    final_txt += "\n"
    final_txt += result
    final_txt += "\n"
    final_txt += "\n"

print("final_txt=", final_txt)

with open(file_path[:-11] + "_OLP" + ".txt", 'w', encoding='utf-8', errors='ignore') as file:
    file.write(final_txt)
