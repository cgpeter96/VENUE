import os
import numpy as np
import argparse

def read_data(filepath):
    dic = {}
    with open(filepath,encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:continue
            items = line.split("\t")
            for item in items:
                dic[item]=items
    return dic


def get_candidate_word(word,data):
    res = data.get(word,None)
    if res is None:
        print("No words in Vocab.")
    else:
        for i in res:
            print(i)

def run(data):
    while True:
        print("-"*50+"\n")
        fmt="""多模态同义词集推断系统
        1. 查询理解
        2. 基于tensorboard的嵌入可视化
        3. 输入q，exit，quit退出"""
        print(fmt)
        print("-"*50+"\n")
        
        mode = input("输入模式：\n")
        if mode=="q" or mode=="exit" or mode=="quit":
            print("退出ing")
            break
        elif mode=="1":
            word = input("输入新词:\n")
            get_candidate_word(word,data)
            print("\n"+"-"*50+"\n")

        elif mode=="2":
            print("\n"+"-"*50+"\n")
            print("启动嵌入可视化,运行在http://0.0.0.0:6006 ")
            print("\n"+"-"*50+"\n")
        else:
            print("错误选项，重新选择")
            print("\n"+"-"*50+"\n")

def main():
    parser = argparse.ArgumentParser(description='Process some params.')
    parser.add_argument('--synset', type=str)
    args = parser.parse_args()
    data = read_data(args.synset)
    run(data)

if __name__ == '__main__':
    main()