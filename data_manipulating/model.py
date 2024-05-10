import pickle
import joblib


def save_model(model, file_name="model"):
    with open(f'{file_name}.pkl', 'wb') as file:
        pickle.dump(model, file)


def load_model(file_name="model"):
    with open(f'{file_name}.pkl', 'rb') as file:
        model = pickle.load(file)
        return model


def save_vectorizer(model, file_name="model"):
    joblib.dump(model, f'{file_name}.joblib')


def load_vectorizer(file_name="model"):
    return joblib.load(f'{file_name}.joblib')
