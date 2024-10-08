# coding=gbk
# -*- coding: utf-8 -*-
import Levenshtein as lev
import requests
import matplotlib.pyplot as plt
import random
import re
import time
import ast
import json
import copy
from datetime import datetime
import numpy as np
import pandas as pd

url = "https://api.ainewserver.com/v1/chat/completions"

headers = {
    "Authorization": "Bearer sk-xpqaolSdwUgRVyCV094cBc08A78142CaB060291f5231Ee85",
    "content-type": "application/json"
}

output_dict = {}
score_dict = {}
score_dict_old = {}
cost_dict = {}
cost_dict_old = {}
reward_dict = {}
reward_dict_old = {}

small = 1e-8
free = 1
t = 1
v = 10
k_t = 1 / 10
k_c = 1 / 10

true_value = 1
false_value = -10

sentence_size = 120

iters = 50000
change_hardness_last = 500
prob_change_hardness = 0.03  #
print_limit = 500
Hardness = []

epsilon = 0.03  # ----------
epsilon = float(epsilon)
epsilon_test_1 = 0.03
epsilon_test_2 = random.uniform(0.0, 0.3)
epsilon_test_3 = random.uniform(0.0, 0.3)
epsilon_test_4 = random.uniform(0.0, 0.3)

alpha = 0.1
alpha_test_1 = alpha
alpha_test_2 = alpha
alpha_test_3 = alpha

# ["gpt-4o-mini", 0.15, 0.6]
LLMs = [["llama3-8b-8192", 0.08, 0.08], ["mixtral-8x7b-32768", 0.7, 0.7], ["command-r", 0.5, 1.5],
        ["gpt-4o-2024-05-13", 5, 15], ["gemini-1.5-pro", 7, 21], ["claude-3-opus-20240229", 15, 75]]

# LLM_judger = "claude-3-opus-20240229"
# c1_judger = 15 / 1000000
# c2_judger = 75 / 1000000
# weight_list = [0, 0, 0, 0, 0, 1]     ###

LLM_judger = "gemini-1.5-pro"
c1_judger = 7 / 1000000
c2_judger = 21 / 1000000
weight_list = [0, 0, 0, 0, 1, 0]     ###

length_input = 25000
length_output = 25000

File_path = ['Task2.txt', 'Task3.txt', 'Task1.txt']
English = False

parameter_file = 'parameters.txt'

gamma = 0
n_LLMs = len(LLMs)
n_states = n_LLMs * 2
epochs = 1

try:
    with open('output_dict_judged_by_Gem.txt', 'r') as file:
        output_dict = json.load(file)
        print("reading succeed")
except:
    print("reading failed")

try:
    with open('output_dict_judged_by_Cla.txt', 'r') as file:
        output_dict_Claude = json.load(file)
        print("reading succeed")
except:
    print("reading failed")

for key in output_dict:
    score_dict.update({key: np.zeros((n_LLMs, 4))})
    cost_dict.update({key: np.zeros((n_LLMs, 4))})
    reward_dict.update({key: np.zeros((n_LLMs, 4))})

LLMs_topic_finder = copy.deepcopy(LLMs)
LLMs_topic_locator = copy.deepcopy(LLMs)
LLMs_relationship_checker = copy.deepcopy(LLMs)
LLMs_content_organizer = copy.deepcopy(LLMs)
LLM_table = [LLMs_topic_finder, LLMs_topic_locator, LLMs_relationship_checker, LLMs_content_organizer]

Cost_ref_topic_finder = []
Cost_ref_topic_locator = []
Cost_ref_relationship_checker = []
Cost_ref_content_organizer = []

Cost_min_topic_finder = []
Cost_min_topic_locator = []
Cost_min_relationship_checker = []
Cost_min_content_organizer = []

Reward_topic_finder = []
Reward_topic_locator = []
Reward_relationship_checker = []
Reward_content_organizer = []

Score_success = np.zeros((len(LLMs), 4))
Score_failure = np.zeros((len(LLMs), 4))

Action_hist = []
win = 1
key = 0
compare = 0

LLM_topic_finder = 0
LLM_topic_locator = 0
LLM_relationship_checker = 0
LLM_content_organizer = 0
LLMs_current = [LLM_topic_finder, LLM_topic_locator, LLM_relationship_checker, LLM_content_organizer]

Q_table_topic_finder = np.zeros((n_states, n_LLMs))
Q_table_topic_locator = np.zeros((n_states, n_LLMs))
Q_table_relationship_checker = np.zeros((n_states, n_LLMs))
Q_table_content_organizer = np.zeros((n_states, n_LLMs))

Q_cost_table = np.zeros((n_states, n_LLMs))
for rr in range(n_states):
    for cc in range(6):
        Q_cost_table[rr][cc] = (LLMs[cc][1] + LLMs[cc][2]) / 40

state_topic_finder = 0
state_topic_locator = 0
state_relationship_checker = 0
state_content_organizer = 0

LLM_TOPic_finder = []
LLM_TOPic_locator = []
LLM_RELationship_checker = []
LLM_CONtent_organizer = []

Markov_end = []

def iou(box1, box2):
    try:
        a1, b1 = box1
        a2, b2 = box2

        inter_left = max(a1, a2)
        inter_right = min(b1, b2)

        inter_area = max(0, inter_right - inter_left)

        union_area = (b1 - a1) + (b2 - a2) - inter_area

        IOU = inter_area / union_area if union_area != 0 else 0

    except Exception as e:
        IOU = 0

    return copy.deepcopy(IOU)


def similarity(str1, str2):

    distance = lev.distance(str1, str2)

    similarity = 1 - (distance / max(len(str1), len(str2)))

    return copy.deepcopy(similarity)


def random_min_values_indices(matrix, num_values=1):

    num_cols = len(matrix[0])
    num_rows = len(matrix)

    indices = []

    for col in range(num_cols):

        column_values = [row[col] for row in matrix]

        min_value = min(column_values)

        min_indices = [index for index, value in enumerate(column_values) if value == min_value]

        min_indices_new = [x for x in min_indices if x != LLMs_current[col]]

        try:
            selected_indices = random.sample(min_indices_new, num_values)
        except:
            selected_indices = random.sample(min_indices, num_values)

        indices.extend(selected_indices)

    return copy.deepcopy(indices)


def markov(table):
    odd_indices = np.arange(1, table.shape[0], 2)

    table = table[odd_indices]

    markov_end = -1
    markov_start = 0
    for qq in range(10):
       markov_end = np.argmax(table[markov_start])
       if markov_end == markov_start or all(elem == 0 for elem in table[markov_end]):
            break
       else:
           markov_start = markov_end
    else:
        markov_end = -1

    return markov_end


def weight_update(markov_index, weight_list, weight_step=0.01):
    non_zero_elements_with_id = [(index, elem) for index, elem in enumerate(weight_list) if elem != 0 and index != markov_index]
    count_non_zero = len(non_zero_elements_with_id)

    if 1 - weight_list[markov_index] > weight_step:
        adjust = weight_step
        weight_list[markov_index] += adjust

    else:
        adjust = 1 - weight_list[markov_index]
        weight_list[markov_index] += adjust

    sorted_indices = sorted(enumerate(weight_list), key=lambda pair: pair[1], reverse=False)

    for index, value in sorted_indices:

        if index != markov_index and value != 0:

            if weight_list[index] > adjust / count_non_zero:
                weight_list[index] -= adjust / count_non_zero
            else:
                residual = weight_list[index]
                weight_list[index] = 0
                adjust -= residual
                count_non_zero -= 1

    return copy.deepcopy(weight_list)


def ask(LLM_name, question):
    start_time = time.time()  #

    response = ""
    again = 1
    while again > 0:

        try:
            invoke = {"messages": [{"role": "user", "content": question, }], "model": LLM_name, "temperature": 0, }

            response = requests.post(url, headers=headers, json=invoke).text

            # print("response here=", response) if iter > (iters - print_limit) or iter < print_limit else None

            response = json.loads(response)


            response = response["choices"][0]["message"]['content']
            again = 0

            if "Hello!" in response or response.count('\n') > 30:
                print("Hello or repeatation in response:", response)
                again = 1

        except Exception as e:
            # print("response =", response) if iter > (iters - print_limit) or iter < print_limit else None
            if "rate limit" in str(response) or response == "":
                again = 1

            else:
                again = again - 0.00001
                time.sleep(1)
            print(again, f"An unexpected error occurred: {e}") if iter > (
                        iters - print_limit) or iter < print_limit else None
            response = "I'm not able to answer"


    print("answer of " + LLM_name + ":", response)

    end_time = time.time()
    run_time = end_time - start_time

    return copy.deepcopy(response), run_time


def judge(LLM_name, question, answer, hint, hint2):
    start_time = time.time()  #
    response = ""

    if English:
        txt = " - Question: " + question + "\n - Answer: " + answer + "\nDo you think the content and the format of the answer are correct?" + hint2 + "If they are correct, please answer 'Both content and format are correct!'; If not, please answer 'No, they are wrong', and" + hint + " remember to put the correct answer behind 'Correct answer: '"
    else:
        txt = " - 问题：" + question + "\n - 答案：" + answer + "\n你认为这个答案的内容和格式都正确吗？" + hint2 + "如果正确，请回答“内容和格式都正确！”；如果错误，请回答“错误！”并" + hint + "注意将正确答案写在“正确答案：”之后"

    again = 1
    while again > 0:

        try:
            invoke = {"messages": [{"role": "user", "content": txt, }], "model": LLM_name, "temperature": 0, }

            response = json.loads(requests.post(url, headers=headers, json=invoke).text)

            response = response["choices"][0]["message"]['content']
            again = 0

        except Exception as e:
            # print("response =", response) if iter > (iters - print_limit) or iter < print_limit else None
            if "rate limit" in str(response) or response == "":
                again = 1
            else:
                again = again - 0.00001
                time.sleep(1)
            print(again, f"An unexpected error occurred: {e}") if iter > (
                        iters - print_limit) or iter < print_limit else None
            response = "I'm not able to answer"

    # print("check by " + LLM_name + ":", response) if iter > (iters - print_limit) or iter < print_limit else None

    if English:
        if "are correct!" not in response and "Correct answer:" in response:
            true_or_false = false_value
            start = response.find('Correct answer:')
            answer_corrected = response[start + 15:]
        else:
            true_or_false = true_value
            answer_corrected = ""
    else:
        if "正确！" not in response and "正确答案" in response:
            true_or_false = false_value
            start = response.find('正确答案')
            answer_corrected = response[start + 5:]
        else:
            true_or_false = true_value
            answer_corrected = ""

    end_time = time.time()
    run_time = end_time - start_time

    return copy.deepcopy(true_or_false), copy.deepcopy(answer_corrected)


def relationship_checker(LLM_index, a):
    global ttt
    cal_times = 0
    LLM_name = LLMs_relationship_checker[LLM_index][0]
    c = (LLMs_relationship_checker[LLM_index][1] + LLMs_relationship_checker[LLM_index][2]) / 2
    c1 = LLMs_relationship_checker[LLM_index][1] / 1000000
    c2 = LLMs_relationship_checker[LLM_index][2] / 1000000
    cost_ref = 0
    cost_min = 0
    reward = 0

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

        # if this_end > next_start:
        if True:
            if English:
                question = "Are '" + this_name + "' and '" + next_name + "' the same product on eBay? Please answer by 'Yes, they are' or 'No, they aren't'. Do not output any other content."
            else:
                question = "“" + this_name + "”和“" + next_name + "”在淘宝上属于同一商品吗？请用“是。”或者“否。”回答，不要输出其他内容。"

            hint = ""
            hint2 = ""

            result, run_time = ask(LLM_name, question)

            # length_input = len(question)
            # length_output = len(result)

            # print("the answer of ", LLM_name, "result=", result) if iter > (iters - print_limit) or iter < print_limit else None

            true_or_false, answer_corrected = judge(LLM_judger, question, result, hint, hint2)

            if true_or_false != true_value:
                f1 = -1 if "No" in result or "否" in result else 1
                f2 = -1 if "No" in answer_corrected or "否" in answer_corrected else 1

                true_or_false = true_value if f1 * f2 == 1 else false_value

            cost_min += (c1 * length_input + c2 * length_output)
            cost_ref += c1_judger * length_input + c2_judger * length_output
            reward = reward + true_or_false * v - k_t * t - (
                        c1 * length_input + c2 * length_output)
            cal_times += 1
            # print("relationship_checker reward =", reward, "true_or_false =", true_or_false, "cost_min =", cost_min,
            #       "LLM =", LLM_name) if iter > (iters - print_limit) or iter < print_limit else None
            ttt = str(ttt) + "file_name = " + str(the_file) + " relationship_checker reward = " + str(
                reward) + " true_or_false = " + str(true_or_false) + " cost_min = " + str(
                cost_min) + " LLM = " + LLM_name + "\n"
            if true_or_false != true_value:
                result = answer_corrected

            if "No" in result or "否" in result:
                # print("No need to combine.") if iter > (iters - print_limit) or iter < print_limit else None
                pass
            else:
                if English:
                    question = "Please combine'" + this_name + "' and '" + next_name + "' into one item and answer with '[{merged item}]', and remember not to output other content."
                else:
                    question = "请把“" + this_name + "”和“" + next_name + "”合并为一个类别并输出成['合并后的类别']，注意不要输出其他内容。"

                hint = ""
                hint2 = ""

                result, run_time = ask(LLM_name, question)

                # length_input = len(question)
                # length_output = len(result)

                start = result.find('[')
                end = result.find(']')

                result = result[start + 2:end - 1]

                # print("merged product =", result) if iter > (iters - print_limit) or iter < print_limit else None

                # print("the answer of ", LLM_name) if iter > (iters - print_limit) or iter < print_limit else None
                true_or_false, answer_corrected = judge(LLM_judger, question, result, hint, hint2)

                cost_min += (c1 * length_input + c2 * length_output)
                cost_ref += c1_judger * length_input + c2_judger * length_output
                reward = reward + true_or_false * v - k_t * t - (c1 * length_input + c2 * length_output)
                cal_times += 1

                # print("relationship_checker reward =", reward, "true_or_false =", true_or_false, "cost_min =", cost_min,
                #       "LLM =", LLM_name) if iter > (iters - print_limit) or iter < print_limit else None
                ttt = str(ttt) + "file_name = " + str(the_file) + " relationship_checker reward = " + str(
                    reward) + " true_or_false = " + str(true_or_false) + " cost_min = " + str(
                    cost_min) + " LLM = " + LLM_name + "\n"
                if true_or_false != true_value:
                    result = answer_corrected

                    start = result.find('[')
                    end = result.find(']')

                    result = result[start + 2:end - 1]

                # print("Checked item =", result) if iter > (iters - print_limit) or iter < print_limit else None

                a[item_index] = [result, [min(this_start, next_start), max(this_end, next_end)]]

                delete_index.append(item_index + 1)

    if delete_index:
        delete_index.sort(reverse=True)  #
        for i in delete_index:
            del a[i]

    if cal_times != 0:
        reward = reward / cal_times
        cost_min = cost_min / cal_times
    return {"Index_list": copy.deepcopy(a), "reward": reward, "cost": [cost_min, cost_ref]}


def topic_locator(LLM_index, text, item_list):
    global ttt
    cal_times = 0
    LLM_name = LLMs_topic_locator[LLM_index][0]
    c = (LLMs_topic_locator[LLM_index][1] + LLMs_topic_locator[LLM_index][2]) / 2
    c1 = LLMs_topic_locator[LLM_index][1] / 1000000
    c2 = LLMs_topic_locator[LLM_index][2] / 1000000
    cost_ref = 0
    cost_min = 0
    reward = 0

    Index_list = []


    for the_item in item_list:
        if English:
            question = "Please read sentence by sentence and extract the first index related to " + the_item + " and the last index related to " + the_item + " from the following text '" + text + "'" + " and then answer by [first index, last index], remember not to output any other content."
            hint = "Please read sentence by sentence and extract the first index related to " + the_item + " and the last index related to " + the_item + " from the above text, and then answer by [first index, last index], remember not to output any other content."
            hint2 = ""
        else:
            question = "请一句一句地读并摘抄出以下段落中介绍" + the_item + "的第一句的号码和介绍" + the_item + "的最后一句的号码" + "“" + text + "”" + "并用[第一句的号码, 最后一句的号码]来回答，注意不要输出号码以外的其他内容"
            hint = "一句一句地读并摘抄出以上段落中介绍" + the_item + "的第一句的号码和介绍" + the_item + "的最后一句的号码" + "再用[第一句的号码, 最后一句的号码]来回答，注意不要输出号码以外的其他内容"
            hint2 = ""

        result, run_time = ask(LLM_name, question)

        # length_input = len(question)
        # length_output = len(result)

        start = result.rfind('[')
        end = result.rfind(']')

        result = result[start:end + 1]
        result = result.replace('”', '"').replace('“', '"')
        # print("the answer of ", LLM_name) if iter > (iters - print_limit) or iter < print_limit else None
        true_or_false, answer_corrected = judge(LLM_judger, question, result, hint, hint2)

        # print("result =", result) if iter > (iters - print_limit) or iter < print_limit else None
        try:
            result = ast.literal_eval(result)
        except Exception as e:
            print("not successful")
            result = [0, 1]

        start = answer_corrected.rfind('[')
        end = answer_corrected.rfind(']')

        answer_corrected = answer_corrected[start:end + 1]
        answer_corrected = answer_corrected.replace('”', '"').replace('“', '"')
        # print("answer_corrected =", answer_corrected) if iter > (iters - print_limit) or iter < print_limit else None
        try:
            answer_corrected = ast.literal_eval(answer_corrected)
        except Exception as e:
            # print("not successful")
            answer_corrected = [2, 3]

        if true_or_false != true_value:
            # print("result =", result) if iter > (iters - print_limit) or iter < print_limit else None
            # print("answer_corrected =", answer_corrected) if iter > (
            #             iters - print_limit) or iter < print_limit else None
            IOU = iou(result, answer_corrected)
            result = answer_corrected
            k = (IOU * (true_value + true_value) - true_value)
        else:
            k = true_value
            IOU = 1

        cost_min += (c1 * length_input + c2 * length_output)
        cost_ref += c1_judger * length_input + c2_judger * length_output
        reward = reward + k * v - k_t * t - (
                    c1 * length_input + c2 * length_output)
        cal_times += 1
        print("topic_locator reward =", reward, "IOU =", IOU, "cost_min =", cost_min, "LLM =", LLM_name) if iter > (
                    iters - print_limit) or iter < print_limit else None
        ttt = str(ttt) + "file_name = " + str(the_file) + " topic_locator reward = " + str(
            reward) + " IOU = " + str(IOU) + " cost_min = " + str(cost_min) + " LLM = " + LLM_name + "\n" if iter > (
                    iters - print_limit) or iter < print_limit else None
        # print("result =", result) if iter > (iters - print_limit) or iter < print_limit else None
        try:
            result = ast.literal_eval(result)
        except Exception as e:
            print("not successful")

        temp = [the_item, result]
        Index_list.append(copy.deepcopy(temp))

    if cal_times != 0:
        reward = reward / cal_times
        cost_min = cost_min / cal_times
    return {"Index_list": copy.deepcopy(Index_list), "reward": reward, "cost": [cost_min, cost_ref]}


def topic_finder(LLM_index, text):
    global ttt
    LLM_name = LLMs_topic_finder[LLM_index][0]
    c = (LLMs_topic_finder[LLM_index][1] + LLMs_topic_finder[LLM_index][2]) / 2
    c1 = LLMs_topic_finder[LLM_index][1] / 1000000
    c2 = LLMs_topic_finder[LLM_index][2] / 1000000
    cost_ref = 0
    cost_min = 0
    reward = 0
    if English:
        question = "Please read sentence by sentence and determine which items are being sold in the following sentences" + "'" + slice_txt + "'" + 'Only select the items with price, and answer in the format of ["item 1: price 1 (if exist)", "item 2: price 2 (if exist)", "item 3: price 3 (if exist)"] in the order of appearance.'
        hint = "read sentence by sentence and determine which items are being sold in the above sentences" + "'" + slice_txt + "'" + 'Only select the items with price, and answer in the format of ["item 1: price 1 (if exist)", "item 2: price 2 (if exist)", "item 3: price 3 (if exist)"] in the order of appearance.'
        hint2 = "Please note not to output items whose prices are not mentioned,"
    else:
        question = "请判断以下段落在销售哪些商品，从头到尾一句一句地阅读" + "“" + text + "”" + "只将有价格的商品名称按出现顺序输出成一个['商品1的名称:商品1的价格', '商品2的名称:商品2的价格', '商品3的名称:商品3的价格']列表"
        hint = "判断以上段落在销售哪些商品，从头到尾一句一句地阅读，只将有价格的商品名称按出现顺序输出成一个['商品1的名称:商品1的价格', '商品2的名称:商品2的价格', '商品3的名称:商品3的价格']列表"
        hint2 = "注意不要输出价格未提及的商品，"

    result, run_time = ask(LLM_name, question)

    # length_input = len(question)
    # length_output = len(result)

    # print("result=", result) if iter > (iters - print_limit) or iter < print_limit else None

    try:
        first_bracket_index = result.rfind('[')
        last_bracket_index = result.rfind(']')

        result = result[first_bracket_index:last_bracket_index + 1]
        result = str(result)
    except Exception as e:
        print("not successful")

    # print("the answer of ", LLM_name) if iter > (iters - print_limit) or iter < print_limit else None
    true_or_false, answer_corrected = judge(LLM_judger, question, result, hint, hint2)

    answer_corrected = answer_corrected.replace("\n", "")
    result = result.replace("\n", "")

    # print("answer_corrected 5=", answer_corrected) if iter > (iters - print_limit) or iter < print_limit else None
    # print("result 5=", result) if iter > (iters - print_limit) or iter < print_limit else None

    if true_or_false != true_value:
        if result.replace(" ", "") == answer_corrected.replace(" ", ""):
            true_or_false = true_value
        result = answer_corrected

    cost_min += (c1 * length_input + c2 * length_output)
    cost_ref += c1_judger * length_input + c2_judger * length_output
    reward = true_or_false * v - k_t * t - (
                c1 * length_input + c2 * length_output)

    # print("topic_finder reward =", reward, "true_or_false =", true_or_false, "cost_min =", cost_min, "LLM =",
    #       LLM_name) if iter > (iters - print_limit) or iter < print_limit else None
    ttt = str(ttt) + "file_name = " + str(the_file) + " topic_finder reward = " + str(
        reward) + " true_or_false = " + str(true_or_false) + " cost_min = " + str(
        cost_min) + " LLM = " + LLM_name + "\n"

    contains_bracket = '[' in result

    if contains_bracket:
        first_bracket_index = result.rfind('[')
        last_bracket_index = result.rfind(']')

        if first_bracket_index != -1 and last_bracket_index != -1:
            if last_bracket_index > first_bracket_index:
                retained_content = result[first_bracket_index:last_bracket_index + 1]
            else:
                retained_content = result
        else:
            retained_content = result

        if "-" in retained_content:
            retained_content = retained_content.replace("-", ",")

        retained_content = retained_content.replace('”', '"').replace('“', '"').replace('：', ':').replace('，', ',')
        print("retained_content =", retained_content)
        try:
            item_list = ast.literal_eval(retained_content)
        except Exception as e:
            print("not successful")

        try:
            item_list = [re.split(r'[：:]', item)[0] for item in item_list]
        except Exception as e:
            print("not successful")

    else:
        temp = re.split(r'[：:-]', result)
        item_list = []
        for index, element in enumerate(temp):
            if index % 2 == 0 and index > 0:
                item_list.append(copy.deepcopy(element))

    seen = set()
    item_list = [x for x in item_list if not (x in seen or seen.add(x))]

    return {"item_list": copy.deepcopy(item_list), "reward": reward, "cost": [cost_min, cost_ref]}


def content_organizer(LLM_index, Text, The_item):
    global ttt
    LLM_name = LLMs_content_organizer[LLM_index][0]
    c = (LLMs_content_organizer[LLM_index][1] + LLMs_content_organizer[LLM_index][2]) / 2
    c1 = LLMs_content_organizer[LLM_index][1] / 1000000
    c2 = LLMs_content_organizer[LLM_index][2] / 1000000
    cost_ref = 0
    cost_min = 0
    reward = 0
    cal_times = 0
    Result = ""

    for ii in range(len(The_item)):
        text = Text[ii]
        the_item = The_item[ii]

        if English:
            question = "Please extract the sentences related to " + my_item + " from the following document through 4 categories: (1) Opening, (2) Order Urging, (3) Price, (4) Product Description." + "'" + text + "'" + '. Remember to answer in the format of - [sentence index, sentence]'
            hint = "Please extract the sentences related to " + my_item + ' from the above document through 4 categories: (1) Opening, (2) Order Urging, (3) Price, (4) Product Description. Remember to answer in the format of - [sentence index, sentence]'
            hint2 = ""
        else:
            question = "请通过4个类别：**（1）商品开场**、**（2）商品价格**、**（3）引导购买**、**（4）商品介绍和卖点描述**，对以下文档与" + the_item + "有关的部分进行摘抄" + "“" + text + "”" + '，注意以- [编号, "句子"]的格式逐行摘抄在4个类别下'
            hint = "通过4个类别：**（1）商品开场**、**（2）商品价格**、**（3）引导购买**、**（4）商品介绍和卖点描述**，对以上文档与" + the_item + "有关的部分进行摘抄" + '，注意以- [编号, "句子"]的格式逐行摘抄在4个类别下'
            hint2 = ""

        result, run_time = ask(LLM_name, question)

        # length_input = len(question)
        # length_output = len(result)

        star_index = result.find('*')
        bracket_index = result.rfind(']')

        if star_index != -1 and bracket_index != -1:
            result = result[star_index: bracket_index + 1]

        true_or_false, answer_corrected = judge(LLM_judger, question, result, hint, hint2)

        star_index = answer_corrected.find('*')
        bracket_index = answer_corrected.rfind(']')

        if star_index != -1 and bracket_index != -1:
            answer_corrected = answer_corrected[star_index: bracket_index + 1]

        if true_or_false != true_value:
            # print("result =", result) if iter > (iters - print_limit) or iter < print_limit else None
            # print("answer_corrected =", answer_corrected) if iter > (
            #             iters - print_limit) or iter < print_limit else None
            simi = similarity(result, answer_corrected)
            result = answer_corrected
            f_value = -10 * true_value
            k = (simi * (true_value - f_value) + f_value)
        else:
            k = true_value
            simi = 1
        Result += the_item + "\n"
        Result += result + "\n"

        cost_min += (c1 * length_input + c2 * length_output)
        cost_ref += c1_judger * length_input + c2_judger * length_output
        reward = reward + k * v - k_t * t - (c1 * length_input + c2 * length_output)
        cal_times += 1
        # print("content_organizer reward =", reward, "simi =", simi, "cost_min =", cost_min, "LLM =",
        #       LLM_name) if iter > (iters - print_limit) or iter < print_limit else None
        ttt = str(ttt) + "file_name = " + str(the_file) + " content_organizer reward = " + str(
            reward) + " simi = " + str(
            simi) + " cost_min = " + str(cost_min) + " LLM = " + LLM_name + "\n"

    if cal_times != 0:
        reward = reward / cal_times
        cost_min = cost_min / cal_times

    return {"result": copy.deepcopy(Result), "reward": reward, "cost": [cost_min, cost_ref]}


def update(task, *args):
    global LLM_topic_finder, LLM_topic_locator, LLM_relationship_checker, LLM_content_organizer, state_topic_finder, state_topic_locator, state_relationship_checker, state_content_organizer, Q_table_topic_finder, Q_table_topic_locator, Q_table_relationship_checker, Q_table_content_organizer, output_dict, score_dict, cost_dict, reward_dict

    if task == "topic_finder":
        my_fun = topic_finder
        my_LLM = LLM_topic_finder
        my_state = state_topic_finder
        my_Qtable = Q_table_topic_finder
        task_index = 0

    elif task == "topic_locator":
        my_fun = topic_locator
        my_LLM = LLM_topic_locator
        my_state = state_topic_locator
        my_Qtable = Q_table_topic_locator
        task_index = 1

    elif task == "relationship_checker":
        my_fun = relationship_checker
        my_LLM = LLM_relationship_checker
        my_state = state_relationship_checker
        my_Qtable = Q_table_relationship_checker
        task_index = 2

    elif task == "content_organizer":
        my_fun = content_organizer
        my_LLM = LLM_content_organizer
        my_state = state_content_organizer
        my_Qtable = Q_table_content_organizer
        task_index = 3

    if the_file not in output_dict.keys():
        output_dict.update({the_file: [[{} for _ in range(4)] for _ in range(n_LLMs)]})
        score_dict.update({the_file: np.zeros((n_LLMs, 4))})
        cost_dict.update({the_file: np.zeros((n_LLMs, 4))})
        reward_dict.update({the_file: np.zeros((n_LLMs, 4))})

    if np.all(my_Qtable == 0):
        first_alpha = alpha
        Output = [0] * n_LLMs
        Reward = [0] * n_LLMs
        Cost = [0] * n_LLMs

        for i in range(n_LLMs - 1, -1, -1):
            args = (i, *args[1:])

            if output_dict[the_file][i][task_index] == {}:
                output_temp = my_fun(*args)  #
                output_dict[the_file][i][task_index] = copy.deepcopy(output_temp)

                reward_temp = output_temp['reward']
            else:
                output_temp = output_dict[the_file][i][task_index]
                output_temp_Claude = output_dict_Claude[the_file][i][task_index]
                reward_temp = output_temp['reward']
                reward_temp_Claude = output_temp_Claude['reward']

                reward_temp = reward_temp * weight_list[4] + reward_temp_Claude * weight_list[5]

            cost_temp = output_temp['cost']

            score_dict[the_file][i][task_index] = 0 if abs(reward_temp) < small else 1 if reward_temp > 0 else -1
            reward_dict[the_file][i][task_index] = reward_temp
            cost_dict[the_file][i][task_index] = cost_temp[0]

            Reward[i] = reward_temp
            Output[i] = output_temp
            Cost[i] = cost_temp

            if reward_temp > 0:
                Score_success[i, task_index] += 1
            else:
                Score_failure[i, task_index] += 1

        for LLM_from in range(0, n_LLMs):
            for LLM_to in range(0, n_LLMs):
                if Reward[LLM_from] != 0:
                    if Reward[LLM_from] > 0:
                        my_state_temp = LLM_from * 2
                    else:
                        my_state_temp = LLM_from * 2 + 1
                    my_Qtable[my_state_temp, LLM_to] += first_alpha * (
                                Reward[LLM_to] - my_Qtable[my_state_temp, LLM_to])

        my_LLM_new = np.argmax(Reward)

        output = Output[my_LLM_new]
        reward = Reward[my_LLM_new]
        cost = Cost[my_LLM_new]

        if reward > 0:
            my_next_state = my_LLM_new * 2
        elif reward == 0:
            my_next_state = my_state
        else:
            my_next_state = my_LLM_new * 2 + 1

    else:
        if np.random.rand() <= float(epsilon):
            my_LLM_new = np.random.choice(n_LLMs)

        else:
            my_LLM_new = np.argmax(my_Qtable[my_state])  #

        # if my_LLM_new == my_LLM or test == 1:
        if my_LLM_new == my_LLM:

            if output_dict[the_file][my_LLM_new][task_index] == {}:
                output = my_fun(*args)  #
                output_dict[the_file][my_LLM_new][task_index] = copy.deepcopy(output)
                reward = output['reward']
            else:
                output = output_dict[the_file][my_LLM_new][task_index]
                output_Claude = output_dict_Claude[the_file][my_LLM_new][task_index]

                reward = output['reward']
                reward_Claude = output_Claude['reward']
                reward = reward * weight_list[4] + reward_Claude * weight_list[5]

            cost = output['cost']
            score_dict[the_file][my_LLM_new][task_index] = 0 if abs(reward) < small else 1 if reward > 0 else -1
            reward_dict[the_file][my_LLM_new][task_index] = reward
            cost_dict[the_file][my_LLM_new][task_index] = cost[0]

            if reward > 0:
                my_next_state = my_LLM_new * 2
            else:
                my_next_state = my_LLM_new * 2 + 1

            if reward != 0:
                my_Qtable[my_state, my_LLM_new] += alpha * (reward - my_Qtable[my_state, my_LLM_new])
            else:
                my_next_state = my_state

        else:
            if output_dict[the_file][my_LLM][task_index] == {}:
                output1 = my_fun(*args)  #
                output_dict[the_file][my_LLM][task_index] = copy.deepcopy(output1)
                reward1 = output1['reward']
            else:
                output1 = output_dict[the_file][my_LLM][task_index]
                output1_Claude = output_dict_Claude[the_file][my_LLM][task_index]

                reward1 = output1['reward']
                reward1_Claude = output1_Claude['reward']
                reward1 = reward1 * weight_list[4] + reward1_Claude * weight_list[5]

            cost1 = output1['cost']
            score_dict[the_file][my_LLM][task_index] = 0 if abs(reward1) < small else 1 if reward1 > 0 else -1
            reward_dict[the_file][my_LLM][task_index] = reward1
            cost_dict[the_file][my_LLM][task_index] = cost1[0]

            args = (my_LLM_new, *args[1:])  #

            if output_dict[the_file][my_LLM_new][task_index] == {}:
                output2 = my_fun(*args)  #
                output_dict[the_file][my_LLM_new][task_index] = copy.deepcopy(output2)
                reward2 = output2['reward']
            else:
                output2 = output_dict[the_file][my_LLM_new][task_index]
                output2_Claude = output_dict_Claude[the_file][my_LLM_new][task_index]
                reward2 = output2['reward']
                reward2_Claude = output2_Claude['reward']
                reward2 = reward2 * weight_list[4] + reward2_Claude * weight_list[5]

            cost2 = output2['cost']
            score_dict[the_file][my_LLM_new][task_index] = 0 if abs(reward2) < small else 1 if reward2 > 0 else -1
            reward_dict[the_file][my_LLM_new][task_index] = reward2
            cost_dict[the_file][my_LLM_new][task_index] = cost2[0]

            if reward1 != 0:
                my_Qtable[my_state, my_LLM] += alpha * (reward1 - my_Qtable[my_state, my_LLM])
                # my_LLM ---> my_LLM_new
                if reward1 > 0:
                    my_state_temp = my_LLM * 2
                else:
                    my_state_temp = my_LLM * 2 + 1
                my_Qtable[my_state_temp, my_LLM_new] += alpha * (reward2 - my_Qtable[my_state_temp, my_LLM_new])

            if reward2 != 0:
                my_Qtable[my_state, my_LLM_new] += alpha * (reward2 - my_Qtable[my_state, my_LLM_new])
                # my_LLM_new ---> my_LLM
                if reward2 > 0:
                    my_state_temp = my_LLM_new * 2
                else:
                    my_state_temp = my_LLM_new * 2 + 1
                my_Qtable[my_state_temp, my_LLM] += alpha * (reward1 - my_Qtable[my_state_temp, my_LLM])

            if reward2 > reward1:
                output = output2
                reward = reward2
                cost = cost2
            else:
                my_LLM_new = my_LLM
                output = output1
                reward = reward1
                cost = cost1

            if reward > 0:
                my_next_state = my_LLM_new * 2
            elif reward == 0:
                my_next_state = my_state
            else:
                my_next_state = my_LLM_new * 2 + 1

    if reward < -10:
        reward = -10

    if task == "topic_finder":
        LLM_TOPic_finder.append(my_LLM_new)
        Reward_topic_finder.append(reward)
        Cost_min_topic_finder.append(cost[0])
        Cost_ref_topic_finder.append(cost[1])
        LLM_topic_finder = my_LLM_new
        state_topic_finder = my_next_state
        Q_table_topic_finder = my_Qtable

    elif task == "topic_locator":
        LLM_TOPic_locator.append(my_LLM_new)
        Reward_topic_locator.append(reward)
        Cost_min_topic_locator.append(cost[0])
        Cost_ref_topic_locator.append(cost[1])
        LLM_topic_locator = my_LLM_new
        state_topic_locator = my_next_state
        Q_table_topic_locator = my_Qtable

    elif task == "relationship_checker":
        LLM_RELationship_checker.append(my_LLM_new)
        Reward_relationship_checker.append(reward)
        Cost_min_relationship_checker.append(cost[0])
        Cost_ref_relationship_checker.append(cost[1])
        LLM_relationship_checker = my_LLM_new
        state_relationship_checker = my_next_state
        Q_table_relationship_checker = my_Qtable

    else:
        LLM_CONtent_organizer.append(my_LLM_new)
        Reward_content_organizer.append(reward)
        Cost_min_content_organizer.append(cost[0])
        Cost_ref_content_organizer.append(cost[1])
        LLM_content_organizer = my_LLM_new
        state_content_organizer = my_next_state
        Q_table_content_organizer = my_Qtable

    return copy.deepcopy(output)

###
final_txt = ""

Weight = []
Weight.append(copy.deepcopy(weight_list))
the_file = File_path[1]
Epsilon = []
random_min_indices = [0, 0, 0, 0]
Score = np.zeros((n_LLMs, 4))
first_time = True
tough = False
recording = 0
ttt = ""

for iter in range(iters):
    # print("the_file =", the_file)
    if iter % 10 == 0:
        print("iter-------------------------------------------------------------------------------", iter, "/", iters)
        try:
            with open(parameter_file, 'r') as file:
                for line in file:
                    line = line.strip()
                    if line.startswith('print_limit ='):
                        print_limit = float(line.split('=')[1].strip())
        except Exception as e:
            print(f"{e}")

    if iter <= 500:
        epsilon = 0.03

    elif iter <= iters - change_hardness_last * 6:  # ----------------- training
        epsilon = epsilon_test_1
        if np.random.rand() < prob_change_hardness:
            random_number = random.randint(0, len(File_path))
            if random_number == len(File_path):
                tough = True
                the_file = random.choice([File_path[len(File_path) - 2], File_path[len(File_path) - 1]])
            else:
                tough = False
                the_file = File_path[random_number]
        elif tough:
            the_file = random.choice([File_path[len(File_path) - 2], File_path[len(File_path) - 1]])
        Epsilon.append(epsilon)

    elif iter <= iters - change_hardness_last * 5:  # ----------------- preparing tough

        the_file = random.choice([File_path[len(File_path) - 2], File_path[len(File_path) - 1]])

        Epsilon.append(epsilon)

    elif iter <= iters - change_hardness_last * 4 - 10:  # -----------------preparing easy
        alpha = 0
        epsilon = 1
        if iter > iters - change_hardness_last * 4.5:
            alpha = 0.1
            epsilon = epsilon_test_1
        the_file = File_path[0]
        Epsilon.append(epsilon)

    elif iter <= iters - change_hardness_last * 3:  # -----------------tough
        Hardness.append(3)
        Epsilon.append(epsilon)
        the_file = File_path[len(File_path) - 1]
        recording = 1

    elif iter <= iters - change_hardness_last * 2:  # ----------------easy
        epsilon = - epsilon_test_2 + (iter - (iters - change_hardness_last * 3)) * 2 / change_hardness_last
        Hardness.append(0)
        Epsilon.append(epsilon)
        the_file = File_path[0]

    elif iter <= iters - change_hardness_last * 1 + 10:  # ------------------medium
        epsilon = - epsilon_test_3 + (iter - (iters - change_hardness_last * 2)) * 2 / change_hardness_last
        Epsilon.append(epsilon)
        Hardness.append(2)
        the_file = File_path[len(File_path) - 2]

    else:  # ------------------------------------------------gpt-4o mini
        epsilon = - epsilon_test_4 + (iter - (iters - change_hardness_last * 1 + 10)) * 2 / change_hardness_last
        if first_time:
            with open('output_dict_judge_by_Gem and Claude.txt', 'w') as file:
                json.dump(output_dict, file)
            try:
                score_dict_old = copy.deepcopy(score_dict)
                cost_dict_old = copy.deepcopy(cost_dict)
                reward_dict_old = copy.deepcopy(reward_dict)
            except Exception as e:
                print("not successful")
            for key in score_dict:
                Score += score_dict[key]

            random_min_indices = random_min_values_indices(score_dict_old[the_file], 1)

            first_time = False

            ###

            # LLMs_topic_finder[random_min_indices[0]] = ["gpt-4o-mini", 0.15, 0.6]
            # LLMs_topic_locator[random_min_indices[1]] = ["gpt-4o-mini", 0.15, 0.6]
            # LLMs_relationship_checker[random_min_indices[2]] = ["gpt-4o-mini", 0.15, 0.6]
            # LLMs_content_organizer[random_min_indices[3]] = ["gpt-4o-mini", 0.15, 0.6]
            #
            # for filefile in File_path:
            #     for jj in range(4):
            #         output_dict[filefile][random_min_indices[jj]][jj] = {}
            #         score_dict[filefile][random_min_indices[jj]][jj] = 0
            #         cost_dict[filefile][random_min_indices[jj]][jj] = 0
            #         reward_dict[filefile][random_min_indices[jj]][jj] = 0
            #
            # for ii, mm in enumerate([Q_table_topic_finder, Q_table_topic_locator, Q_table_relationship_checker,
            #                          Q_table_content_organizer]):
            #     mm[:, random_min_indices[ii]] = 0
            #     mm[random_min_indices[ii] * 2, :] = 0
            #     mm[random_min_indices[ii] * 2 + 1, :] = 0

        Hardness.append(2)
        Epsilon.append(epsilon)

    with open(the_file, 'r', encoding='utf-8') as file:
        txtcontent = file.read()

    while 1:
        list_content = ast.literal_eval(txtcontent)
        slice_list = list_content[:sentence_size]

        slice_txt = json.dumps(slice_list, ensure_ascii=False)

        start_num = list_content[0][0]
        start_num = int(start_num)

        output_1 = update("topic_finder", LLM_topic_finder, slice_txt)  # -------------------------
        item_list = output_1["item_list"]
        reward = output_1["reward"]

        if iter > 100:
            markov_end = markov(Q_table_topic_finder + Q_table_topic_locator + Q_table_relationship_checker + Q_table_content_organizer + Q_cost_table * 4)
            time.sleep(0.1) if iter < 500 else None
            Markov_end.append(markov_end)

            if markov_end != -1:
                weight_list = weight_update(markov_end, weight_list)

        if len(Weight) < 3000:
            print("weight_list =", weight_list)
            Weight.append(weight_list)

        output_2 = update("topic_locator", LLM_topic_locator, slice_txt, item_list)  # -------------------------
        Index_list = output_2["Index_list"]
        reward = output_2["reward"]

        output_3 = update("relationship_checker", LLM_relationship_checker, Index_list)  # -------------------------

        Index_list = output_3["Index_list"]
        reward = output_3["reward"]

        Sentence_list = []
        Sentence_txt = []
        The_item = []

        for item_index, the_item in enumerate(Index_list):
            index = the_item[1]
            start = index[0]
            end = index[1]

            if item_index != len(Index_list) - 1:
                next_item = Index_list[item_index + 1]
                index = next_item[1]
                start_next = index[0]

            else:
                start_next = list_content[len(list_content) - 1][0]

            end = max(end, start_next)

            break1 = 0
            break2 = 0
            for my_index, my_item in enumerate(list_content):  #
                if start in my_item and break1 == 0:
                    start_index = my_index
                    break1 = 1

                if end in my_item and break2 == 0:
                    end_index = my_index
                    break2 = 1

                if break1 == 1 and break2 == 1:
                    break
            else:
                if break1 == 0:
                    start_index = 0
                if break2 == 0:
                    end_index = len(list_content) - 1

            sentence = list_content[start_index: end_index + 1]
            sentence = sentence[:300]
            sentence_txt = json.dumps(sentence, ensure_ascii=False)

            if len(list_content) > sentence_size and item_index == len(Index_list) - 1 - 1:
                break  ### for loop

            Sentence_txt.append(sentence_txt)
            The_item.append(the_item[0])

        output_4 = update("content_organizer", LLM_content_organizer, Sentence_txt, The_item)
        result = output_4["result"]
        reward = output_4["reward"]

        if len(list_content) > sentence_size:
            start_next = str(start_next)
            index = txtcontent.find(start_next) - 1

            if index != -1:
                txtcontent = txtcontent[index:]
                txtcontent = "[" + txtcontent

        else:
            break  ### while loop

print("ttt =\n", ttt)

ff_txt = ""

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
file_name = f"History_{timestamp}.xlsx"

Df = []
Pic_index = []
Plot_name = ["topic_finder", "topic_locator", "relationship_checker", "content_organizer"]

for pic_index, To_draw in enumerate(
        [[LLM_TOPic_finder, Reward_topic_finder, Cost_min_topic_finder, Cost_ref_topic_finder],
         [LLM_TOPic_locator, Reward_topic_locator, Cost_min_topic_locator, Cost_ref_topic_locator],
         [LLM_RELationship_checker, Reward_relationship_checker, Cost_min_relationship_checker, Cost_ref_relationship_checker],
         [LLM_CONtent_organizer, Reward_content_organizer, Cost_min_content_organizer, Cost_ref_content_organizer]]):

    for pp in range(4):
        To_draw[pp] = To_draw[pp][-5 * change_hardness_last:]
        Epsilon = Epsilon[-5 * change_hardness_last:]

    ff_txt += "pic_index =" + str(pic_index) + "\n"
    ff_txt += "LLM =" + str(To_draw[0]) + "\n"
    ff_txt += "Reward =" + str(To_draw[1]) + "\n"

    data = {
        'Iteration': [i for i in range(0, len(To_draw[0]))],
        'LLM': To_draw[0],
        'Reward': To_draw[1],
        'Cost_min': To_draw[2],
        'Cost_ref': To_draw[3]
    }
    df = pd.DataFrame(data)
    df = df.transpose()
    Df.append(df)
    Pic_index.append(f'history_{pic_index}')

    plt.figure(pic_index)  #
    X1 = [i for i in range(0, len(To_draw[0]))]
    plt.subplot(3, 1, 1)  #
    plt.plot(X1, To_draw[0])
    plt.ylim(-0.5, n_LLMs + 0.5)
    plt.title(Plot_name[pic_index])
    plt.xticks([])
    plt.ylabel('LLM')

    X1 = [i for i in range(0, len(To_draw[1]))]
    plt.subplot(3, 1, 2)  #
    plt.plot(X1, To_draw[1])
    plt.ylim(-12, 12)
    # plt.title('Reward')
    plt.xticks([])
    plt.ylabel('Reward')

    X1 = [i for i in range(0, len(To_draw[2]))]
    plt.subplot(3, 1, 3)  #
    plt.plot(X1, To_draw[2])
    # plt.title('Cost')
    plt.xticks([])
    plt.ylabel('Cost')

with open(f'History_{timestamp}.txt', 'w', encoding='utf-8') as file:
    file.write(ff_txt)

try:
    df = pd.DataFrame([score_dict_old])
    df = df.transpose()
    Df.append(df)
    Pic_index.append('score_dict_old')

    df = pd.DataFrame([score_dict])
    df = df.transpose()
    Df.append(df)
    Pic_index.append('score_dict')

    df = pd.DataFrame([reward_dict_old])
    df = df.transpose()
    Df.append(df)
    Pic_index.append('reward_dict_old')

    df = pd.DataFrame([reward_dict])
    df = df.transpose()
    Df.append(df)
    Pic_index.append('reward_dict')

    df = pd.DataFrame([cost_dict_old])
    df = df.transpose()
    Df.append(df)
    Pic_index.append('cost_dict_old')

    df = pd.DataFrame([cost_dict])
    df = df.transpose()
    Df.append(df)
    Pic_index.append('cost_dict')

    df = pd.DataFrame([LLM_table])
    df = df.transpose()
    Df.append(df)
    Pic_index.append('LLM_table')

    df = pd.DataFrame(Q_table_topic_finder)
    Df.append(df)
    Pic_index.append('Q_table_topic_finder')

    df = pd.DataFrame(Q_table_topic_locator)
    Df.append(df)
    Pic_index.append('Q_table_topic_locator')

    df = pd.DataFrame(Q_table_relationship_checker)
    Df.append(df)
    Pic_index.append('Q_table_relationship_checker')

    df = pd.DataFrame(Q_table_content_organizer)
    Df.append(df)
    Pic_index.append('Q_table_content_organizer')

    df = pd.DataFrame(Weight[0:1000])
    df = df.transpose()
    Df.append(df)
    Pic_index.append('Weight')

    df = pd.DataFrame([ttt])
    df = df.transpose()
    Df.append(df)
    Pic_index.append('ttt')

except Exception as e:
    print(f"An unexpected error occurred: {e}")

with pd.ExcelWriter(file_name, engine='openpyxl') as writer:
    for qq in range(len(Df)):
        try:
            Df[qq].to_excel(writer, index=True, sheet_name=Pic_index[qq])
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

plt.show()
