import time

from googletrans import Translator
from fake_useragent import UserAgent
import csv

from tqdm import tqdm

ua = UserAgent()
translator = Translator(
    user_agent=ua.random,
    service_urls=['translate.google.com',]
)


def translate_csv(input_path, output_path):
    # Открываем .csv файл для чтения и новый .csv файл для записи результатов
    with open(input_path, 'r', encoding='utf-8') as input_file, open(output_path, 'w', encoding='utf-8', newline='') as output_file:
        csv_reader = csv.reader(input_file, delimiter=",")
        csv_writer = csv.writer(output_file, delimiter=",")
        # Читаем каждую строку из входного файла
        for row in tqdm(csv_reader):
            try:
                row[2] = translate_text(row[2])
                if row[5]:
                    row[5] = translate_text(row[5])
                csv_writer.writerow(row)
                time.sleep(0.1)

            except Exception as ex:
                print(ex)
                time.sleep(20)
                continue


    print(f'Перевод завершен и сохранен в {output_path}')


def translate_text(text_to_translate, dest="uk"):
    # Переводим текст
    translated_text = translator.translate(text_to_translate, dest=dest).text
    return translated_text


def translate_corpus(corpus_to_translate, dest="en"):
    corpus = [translate_text(text_to_translate, dest) for text_to_translate in tqdm(corpus_to_translate)]
    print(f'Перевод завершен и возвращен')
    return corpus


def check_lang(text, lang="uk"):
    return True if translator.detect(text).lang == lang else False


def check_lang_corpus(corpus_to_translate, lang="uk"):
    return True if translator.detect(" ".join(corpus_to_translate)).lang == lang else False


if __name__ == '__main__':
    INPUT_PATH = '../datasets/tasks-2-3/combined_dataset.csv'
    OUTPUT_PATH = '../datasets/tasks-2-3/combined_dataset_ua.csv'

    translate_csv(INPUT_PATH, OUTPUT_PATH)
