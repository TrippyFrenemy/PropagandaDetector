import os
from config import MODEL_PATH, LAST_NAME, PHOTO_PATH


def delete_files_by_pattern(directory, pattern):
    if directory != "models_back":
        # Перебираем все файлы в указанной директории
        for filename in os.listdir(directory):
            # Проверяем, содержит ли имя файла заданный шаблон
            if pattern in filename:
                # Формируем полный путь к файлу
                file_path = os.path.join(directory, filename)
                # Удаляем файл
                os.remove(file_path)
                print(f"Deleted {file_path}")


if __name__ == "__main__":
    delete_files_by_pattern(MODEL_PATH, LAST_NAME)
    delete_files_by_pattern(PHOTO_PATH, LAST_NAME)
