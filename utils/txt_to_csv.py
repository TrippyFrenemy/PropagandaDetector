import csv

INPUT_PATH = '../datasets-v5/task-1/task1.train.txt'
OUTPUT_PATH = '../datasets/propaganda_on_article_level.csv'


# Открываем текстовый файл для чтения и CSV файл для записи
with open(INPUT_PATH, 'r', encoding='utf-8') as txt_file, open(OUTPUT_PATH, 'w', encoding='utf-8', newline='') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=";")

    # Читаем каждую строку из текстового файла
    for line in txt_file:
        # Разделяем строку по табуляции и записываем в CSV файл
        csv_writer.writerow(line.strip().split('\t'))

print(f'Преобразование завершено и сохранено в {OUTPUT_PATH}')
