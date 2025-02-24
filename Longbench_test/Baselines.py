# coding=gbk
# -*- coding: utf-8 -*-
import time
import requests
import json

url = "https://api.ainewserver.com/v1/chat/completions"

headers = {
    "Authorization": "Bearer sk-xpqaolSdwUgRVyCV094cBc08A78142CaB060291f5231Ee85",
    "content-type": "application/json"
}

LLM_name = "claude-3-5-sonnet-20240620"

file_list = ["lsht_sampling_10convert.txt"]

Chinese = True
total = 10
step = 2
Heads = [number for number in range(1, total+1) if number % step != 0]
print(Heads)

final_Direct = ""
final_COT = ""
final_Reflex = ""
final_LA = ""
final_COA = ""



for file_name in file_list:

    with open(file_name, 'r', encoding='utf-8', errors='ignore') as file:
        txt = file.read()

    for H in Heads:
        print("H =", H)


        ### Direct
        if Chinese:
            question = "以下文档有" + str(total) + "条新闻片段，每条新闻片段包含若干句话，首先判断以下文档有哪些新闻标题，接着通过第" + str(H) + "-" + str(H+step-1) + "个新闻标题的4个方面”(1) 未来计划”、“(2) 猜测”、“(3) 观点”、“(4) 事实”，对以下文档进行摘抄" + "“" + txt + "”" + "，注意判断以上文档有哪些新闻标题并通过第" + str(H) + "-" + str(H+step-1) + "个新闻标题的4个方面”(1) 未来计划”、“(2) 猜测”、“(3) 观点”、“(4) 事实”，对以上文档进行逐句摘抄（仅仅摘抄而不要输出其他内容），并且以[原序号, 句子]的格式，不重复、不遗漏地归类并摘抄在第" + str(H) + "-" + str(H+step-1) + "个新闻标题的4个方面下"
        else:
            question = "There are " + str(total) + " pieces of news in the following document, and each piece of news headline includes multiple sentences. Please first identify the news headlines in the following document, and place each and every sentence into the most appropriate one of the four entries in unchanged sequence: '(1) Future Plans: , (2) Assumptions: , (3) Opinions: , (4) Facts: ' of each headline in Headlines " + str(H) + "-" + str(H+step-1) + " \n" + txt + " \nRemember to identify the news headlines first and then place each and every sentence into the most appropriate one of the 4 entries in unchanged sequence: '(1) Future Plans: , (2) Assumptions: , (3) Opinions: , (4) Facts: ' of each headline in Headlines " + str(H) + "-" + str(H+step-1) + " in the original list format of [sentence index, sentence]. Remember to include ALL the sentences and not to repeat or omit any possible sentence for the Headlines " + str(H) + "-" + str(H+step-1)
        print("question =", question)
        while True:
            try:
                print("I'm here")
                invoke = {"messages": [{"role": "user", "content": question, }], "model": LLM_name,
                          "temperature": 0, }
                print(invoke)
                response = requests.post(url, headers=headers, json=invoke).text
                response = json.loads(response)

                result = response["choices"][0]["message"]['content']

                result_Direct = result.strip()
                print("result_Direct =", result_Direct)
                final_Direct += result_Direct

                if H == Heads[-1]:
                    with open(file_name[:-11] + "_Direct" + ".txt", 'w', encoding='utf-8', errors='ignore') as file:
                        file.write(final_Direct)

                break
            except Exception as e:
                print(response)
                print(f"An error occurred: {e}")
        time.sleep(10)




        ### COT
        if Chinese:
            question_COT = question + "，请一步步解释你的思路，然后给出最终答案"
        else:
            question_COT = question + ". Please explain your thought process step by step and then give the final answer"

        while True:
            try:
                invoke = {"messages": [{"role": "user", "content": question_COT, }], "model": LLM_name,
                          "temperature": 0, }

                response = requests.post(url, headers=headers, json=invoke).text
                response = json.loads(response)

                result = response["choices"][0]["message"]['content']

                result_COT = result.strip()
                print("result_COT =", result_COT)
                final_COT += result_COT

                if H == Heads[-1]:
                    with open(file_name[:-11] + "_COT" + ".txt", 'w', encoding='utf-8', errors='ignore') as file:
                        file.write(final_COT)
                break
            except Exception as e:
                print(f"An error occurred: {e}")
                print(response)




        ### Reflex
        while True:
            if Chinese:
                hint = "请仔细回顾你刚才的回答。考虑一下你是否可能犯了错误，或者是否有改进的地方。然后重新给出答案"
            else:
                hint = "Please carefully review your previous response. Consider whether you might have made a mistake, or if there is room for improvement. Then provide the complete answer again."
            try:
                invoke = {"messages": [{"role": "user", "content": question_COT, }, {"role": "assistant", "content": result_COT, }, {"role": "user", "content": hint, }, ], "model": LLM_name,
                          "temperature": 0, }

                response = requests.post(url, headers=headers, json=invoke).text
                response = json.loads(response)

                result = response["choices"][0]["message"]['content']

                result_Reflex = result.strip()
                print("result_Reflex =", result_Reflex)
                final_Reflex += result_Reflex

                if H == Heads[-1]:
                    with open(file_name[:-11] + "_Reflex" + ".txt", 'w', encoding='utf-8', errors='ignore') as file:
                        file.write(final_Reflex)

                break
            except Exception as e:
                print(f"An error occurred: {e}")
                print(response)




        ### LA
        while True:

            if Chinese:
                hint = "以上是另一个人的答案，请结合你们的答案并给出最终答案"
            else:
                hint = "The above is another person's answer; please combine your answers and provide the final answer."

            try:
                invoke = {"messages": [{"role": "user", "content": question_COT, }, {"role": "assistant", "content": result_COT, }, {"role": "user", "content": result_Direct + hint, }, ], "model": LLM_name,
                          "temperature": 0, }

                response = requests.post(url, headers=headers, json=invoke).text
                response = json.loads(response)

                result = response["choices"][0]["message"]['content']

                result_LA = result.strip()
                print("result_LA =", result_LA)
                final_LA += result_LA

                if H == Heads[-1]:
                    with open(file_name[:-11] + "_LA" + ".txt", 'w', encoding='utf-8', errors='ignore') as file:
                        file.write(final_LA)

                break
            except Exception as e:
                print(f"An error occurred: {e}")
                print(response)




    ### COA
    def find_nth_element_comma(s, n):
        if n <= 0 or n > len(s):
            return -1

        index = n - 1
        comma_index = s.rfind(',', 0, index)

        return comma_index

    cut1 = find_nth_element_comma(txt, int(len(txt)/3)) + 1
    cut2 = find_nth_element_comma(txt, int(len(txt) * 2 / 3)) + 1

    txt1 = txt[:cut1]
    txt2 = txt[cut1:cut2]
    txt3 = txt[cut2:]

    if Chinese:
        densify = "请精炼以上内容，选择你认为重要的句子按照原格式输出，保持列表的格式并保留句子前面的序号："
    else:
        densify = "Please refine the above content, select the sentences you consider important, and output them in the original format, maintaining the list structure and retaining the numbering before each sentence:"

    while True:
        try:
            invoke = {"messages": [{"role": "user", "content": txt1 + densify, }, ], "model": LLM_name,
                      "temperature": 0, }

            response = requests.post(url, headers=headers, json=invoke).text
            response = json.loads(response)

            result = response["choices"][0]["message"]['content']

            break
        except Exception as e:
            print(f"An error occurred: {e}")

    dense1 = result.strip()
    print("dense1 =", dense1)

    while True:
        try:
            invoke = {"messages": [{"role": "user", "content": dense1 + txt2 + densify, }, ], "model": LLM_name,
                      "temperature": 0, }

            response = requests.post(url, headers=headers, json=invoke).text
            response = json.loads(response)

            result = response["choices"][0]["message"]['content']

            break
        except Exception as e:
            print(f"An error occurred: {e}")

    dense2 = result.strip()
    print("dense2 =", dense2)

    while True:
        try:
            invoke = {"messages": [{"role": "user", "content": dense2 + txt3 + densify, }, ], "model": LLM_name,
                      "temperature": 0, }

            response = requests.post(url, headers=headers, json=invoke).text
            response = json.loads(response)

            result = response["choices"][0]["message"]['content']

            break
        except Exception as e:
            print(f"An error occurred: {e}")

    dense3 = result.strip()
    print("dense2 =", dense3)

    question = "请判断以下文档有哪些新闻标题，通过每个新闻标题的4个方面”(1) 未来计划”、“(2) 猜测”、“(3) 观点”、“(4) 事实”，对以下文档进行摘抄" + "“" + dense3 + "”" + '，注意识别新闻标题并通过每个新闻标题的4个方面”(1) 未来计划”、“(2) 猜测”、“(3) 观点”、“(4) 事实”，对以上文档进行逐句摘抄（仅仅摘抄而不要输出其他内容），并且以[序号, 句子]的格式，不重复、不遗漏地归类并摘抄在每个新闻标题的4个方面下'
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

    result_COA = result.strip()
    print("result_COA =", result_COA)

    with open(file_name[:-11] + "_COA" + ".txt", 'w', encoding='utf-8', errors='ignore') as file:
        file.write(result_COA)