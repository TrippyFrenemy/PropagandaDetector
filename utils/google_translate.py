from pygoogletranslation import Translator
import csv

from tqdm import tqdm

translator = Translator()


def translate_csv(input_path, output_path):
    # Открываем .csv файл для чтения и новый .csv файл для записи результатов
    with open(input_path, 'r', encoding='Windows-1251') as input_file, open(output_path, 'w', encoding='Windows-1251', newline='') as output_file:
        csv_reader = csv.reader(input_file, delimiter=";")
        csv_writer = csv.writer(output_file, delimiter=";")
        # Читаем каждую строку из входного файла
        for row in tqdm(csv_reader):
            # Текст для перевода находится в первом столбце каждой строки
            text_to_translate = str(row[0])

            # Переводим текст
            translated_text = translator.translate(text_to_translate, dest="uk").text

            # Записываем переведенный текст в выходной файл
            csv_writer.writerow([translated_text.replace("\u200B", ""), row[1]])

    print(f'Перевод завершен и сохранен в {output_path}')


def translate_text(text_to_translate):
    # Переводим текст
    translated_text = translator.translate(text_to_translate, dest="en").text
    return translated_text


def translate_corpus(corpus_to_translate):
    corpus = [translate_text(text_to_translate) for text_to_translate in tqdm(corpus_to_translate)]
    print(f'Перевод завершен и возвращен')
    return corpus


if __name__ == '__main__':
    INPUT_PATH = '../datasets/propaganda_on_article_level.csv'
    OUTPUT_PATH = '../datasets/propaganda_on_article_level_ua.csv'

    translate_csv(INPUT_PATH, OUTPUT_PATH)
