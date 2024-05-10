import os
import re

import pandas as pd

FOLDER_BASE = "../datasets-v5/tasks-2-3/train/"
OUTPUT_FILE = "../datasets/propaganda_on_sentence_level.csv"


for file_name in os.listdir(FOLDER_BASE):
    print(file_name)
    if file_name.endswith(".txt"):
        with open(f'{FOLDER_BASE}{file_name}', 'r', encoding='utf-8') as f1:
            file1_data = f1.readlines()
            file1_data = [re.sub(r'[\d\W_]+', ' ', data) for data in file1_data]
            file1_data = [data.strip() for data in file1_data]
        with open(f'{FOLDER_BASE}{file_name.replace(".txt", ".task2.labels")}', 'r', encoding='utf-8') as f2:
            file2_data = f2.readlines()
            file2_data = [line.strip().split('\t') for line in file2_data]
            file2_data_filtered = [line[2] for line in file2_data]

        # Объединение DataFrame в один и сохранение в CSV
        df = pd.DataFrame({'Text': file1_data, 'Class': file2_data_filtered})
        # Удаление строк с пустым текстовым полем
        df = df[df['Text'].str.strip() != '']
        df.to_csv(OUTPUT_FILE, mode="a", header=False, index=False, sep=";")
