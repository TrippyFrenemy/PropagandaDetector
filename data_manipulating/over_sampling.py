import pandas as pd


def over_sampling(df):
    # Разделение датафрейма на классы
    df_major = df[df['Class'] == 0]
    df_minor = df[df['Class'] == 1]

    # Вычисление необходимого количества копий
    num_major = df_major.shape[0]
    num_minor = df_minor.shape[0]
    num_needed = num_major - num_minor  # сколько не хватает до баланса

    # Дублирование примеров минорного класса
    oversampled_minor = df_minor.sample(num_needed, replace=True)

    # Объединение датафреймов для получения сбалансированного набора данных
    balanced_df = pd.concat([df_major, df_minor, oversampled_minor], axis=0)

    # Перемешивание данных
    balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)

    # Проверка количества примеров по классам
    # print(balanced_df['Class'].value_counts())

    return balanced_df
