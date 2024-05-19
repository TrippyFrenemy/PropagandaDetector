from contextlib import asynccontextmanager
import time

from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import HTMLResponse
from matplotlib import pyplot as plt
from sklearn.inspection import permutation_importance
from starlette.templating import Jinja2Templates
import uvicorn

from data_manipulating.model import load_model, load_vectorizer
from data_manipulating.preprocessing import Preprocessor
from config import FOREST_PATH, TFIDF_PATH


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.model = load_model(f"{FOREST_PATH}")
    app.vectorizer = load_vectorizer(f"{TFIDF_PATH}")
    app.threshold = 0.45

    print("Model parameters:     ", app.model.get_params())
    print("Vectorizer parameters: ", app.vectorizer.get_params())
    print("Threshold value: ", app.threshold)
    yield


app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})


@app.post("/", response_class=HTMLResponse)
async def handle_text(request: Request, text: str = Form(...)):
    start = time.time()
    text = text.split(";")

    if text[-1] == "":
        text = text[:-1]

    # Check the length of each element
    for item in text:
        if (len(item) > 500):
            raise HTTPException(status_code=400, detail="Each element in the text should not exceed 500 characters.")

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
    feature_importances = app.model.feature_importances_

    # Пермутационная важность признаков
    perm_importance = permutation_importance(app.model, vectorized_text_dense, predicted_forest, n_repeats=2, random_state=0)
    perm_importance_values = perm_importance.importances_mean

    # Pack results with filtered TF-IDF values and feature importances based on preprocessed text
    results = []
    for original, preprocessed, tfidf, prediction in zip(text, preprocessed_text, tfidf_values, result_list_forest):
        words = preprocessed.split()
        word_tfidf = [(word, tfidf[feature_names.tolist().index(word)]) for word in words if word in feature_names]
        word_importance = [(word, feature_importances[feature_names.tolist().index(word)]) for word in words if word in feature_names]
        word_perm_importance = [(word, perm_importance_values[feature_names.tolist().index(word)]) for word in words if
                                word in feature_names]
        avg_importance = sum([imp for _, imp in word_importance]) / len(word_importance) if word_importance else 0
        results.append({
            "original": original,
            "preprocessed": preprocessed,
            "word_tfidf": word_tfidf,
            "word_importance": word_importance,
            "word_perm_importance": word_perm_importance,
            "avg_importance": avg_importance,
            "status": prediction
        })

    end = time.time()
    length = end - start

    # Show the results : this can be altered however you like
    print("It took", length, "seconds!")

    return templates.TemplateResponse("result.html", {"request": request, "results": results})


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return templates.TemplateResponse("error.html", {"request": request, "detail": exc.detail}, status_code=exc.status_code)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
