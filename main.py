import time
from contextlib import asynccontextmanager
from io import BytesIO

import uvicorn
from fastapi import FastAPI, Request, HTTPException, UploadFile, Form, File
from fastapi.responses import HTMLResponse
from sklearn.inspection import permutation_importance
from starlette.templating import Jinja2Templates

from config import FOREST_PATH, TFIDF_PATH, MODEL_PATH, CASCADE_PATH, IMPROVED_CASCADE_PATH
from data_manipulating.manipulate_models import load_model, load_vectorizer
from data_manipulating.preprocessing import Preprocessor
from pipelines.cascade_classification import CascadePropagandaPipeline
from pipelines.improved_cascade import ImprovedCascadePropagandaPipeline
from utils.add_data_to_csv import add_data_to_csv
from utils.google_translate import check_lang_corpus, translate_corpus


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.model = load_model(f"{FOREST_PATH}")
    app.vectorizer = load_vectorizer(f"{TFIDF_PATH}")
    app.threshold = 0.45

    app.cascade_pipeline = CascadePropagandaPipeline(
        model_path=f"{MODEL_PATH}",
        model_name=f"{CASCADE_PATH}"
    )
    app.improved_cascade_pipeline = ImprovedCascadePropagandaPipeline(
        model_path=f"{MODEL_PATH}",
        model_name=f"{IMPROVED_CASCADE_PATH}"
    )

    print("Model parameters:     ", app.model.get_params())
    print("Vectorizer parameters: ", app.vectorizer.get_params())
    print("Threshold value: ", app.threshold)

    print()
    print("Cascade parameters: ", app.cascade_pipeline.get_params(detailed=True))
    print("Improved Cascade parameters: ", app.improved_cascade_pipeline.get_params(detailed=True))
    print()
    yield


app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")


async def check_text(text):
    for char in [';', '!', '?', '\n']:
        text = text.replace(char, '.')

    text = [sentence.strip() for sentence in text.split('.')
            if len(sentence.strip()) > 3]

    if text[-1] == "":
        text = text[:-1]

    # Check the length of each element
    for item in text:
        if len(item) > 500:
            raise HTTPException(status_code=400, detail="Each element in the text should not exceed 500 characters.")

    return text


@app.get("/", response_class=HTMLResponse)
async def classification(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})


@app.post("/", response_class=HTMLResponse)
async def classification(request: Request, text: str = Form(...)):
    try:
        start = time.time()

        text = await check_text(text)

        if not check_lang_corpus(text, "en"):
            text = translate_corpus(text)

        preprocessing = Preprocessor()
        preprocessed_text = preprocessing.preprocess_corpus(text, lemma=True)
        vectorized_text = app.vectorizer.transform(preprocessed_text)
        vectorized_text_dense = vectorized_text.toarray()

        predicted_forest = (app.model.predict_proba(vectorized_text_dense)[:, 1] >= app.threshold).astype(int)
        result_list_forest = ["non-propaganda" if x == 0 else "propaganda" for x in predicted_forest]

        # Extracting feature names and TF-IDF values
        feature_names = app.vectorizer.get_feature_names_out()
        tfidf_values = vectorized_text_dense

        # Feature importances from the Random Forest model
        # feature_importances = app.model.feature_importances_

        # Пермутационная важность признаков
        # perm_importance = permutation_importance(app.model, vectorized_text_dense, predicted_forest, n_repeats=10, random_state=0)
        # perm_importance_values = perm_importance.importances_mean

        # Pack results with filtered TF-IDF values and feature importances based on preprocessed text
        results = []
        for original, preprocessed, tfidf, prediction, percent in zip(text, preprocessed_text, tfidf_values, result_list_forest, app.model.predict_proba(vectorized_text_dense)[:, 1]):
            words = preprocessed.split()
            word_tfidf = [(word, tfidf[feature_names.tolist().index(word)]) for word in words if word in feature_names]
            # word_importance = [(word, feature_importances[feature_names.tolist().index(word)]) for word in words if word in feature_names]
            # word_perm_importance = [(word, perm_importance_values[feature_names.tolist().index(word)]) for word in words if
            #                         word in feature_names]
            # avg_importance = sum([imp for _, imp in word_importance]) / len(word_importance) if word_importance else 0
            results.append({
                "original": original,
                "preprocessed": preprocessed,
                "word_tfidf": word_tfidf,
                # "word_importance": word_importance,
                # "word_perm_importance": word_perm_importance,
                # "avg_importance": avg_importance,
                "percent": percent,
                "status": prediction
            })

        end = time.time()
        length = end - start

        print("It took", length, "seconds!")

        return templates.TemplateResponse("result.html", {"request": request, "results": results})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/improved_classification", response_class=HTMLResponse)
async def improved_classification(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})


@app.post("/improved_classification", response_class=HTMLResponse)
async def improved_classification(request: Request, text: str = Form(...)):
    try:
        start = time.time()

        text = await check_text(text)

        results, formatted = app.improved_cascade_pipeline.predict(text, True)

        end = time.time()
        length = end - start

        print("It took", length, "seconds!")

        return templates.TemplateResponse("result.html", {"request": request, "results": results, "formatted": formatted})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/add", response_class=HTMLResponse)
async def get_data_to_csv(request: Request):
    return templates.TemplateResponse("add_data_to_csv.html", {"request": request})


@app.post("/add")
async def get_data_to_csv(request: Request, text: str = Form(None), files: list[UploadFile] = File(...)):

    if text:
        data = text.strip()
    else:
        data = ""
        for file in files:
            try:
                # Считываем содержимое файла в переменную
                contents = await file.read()
                file_contents = BytesIO(contents)

                # Пример обработки содержимого файла
                file_contents.seek(0)
                data += file_contents.read().decode()  # или другой подходящий метод для обработки

                if not data.endswith('\r\n'):
                    data += '\r\n'
            except Exception as e:
                return {"message": f"There was an error uploading the file: {e}"}
            finally:
                await file.close()

    try:
        add_data_to_csv(data)
        return {"message": f"Your data was successfully added to the dataset"}
    except Exception as e:
        return {"message": f"There was an error uploading the dataset: {e}"}


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return templates.TemplateResponse("error.html", {"request": request, "detail": exc.detail}, status_code=exc.status_code)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
