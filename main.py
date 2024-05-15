from contextlib import asynccontextmanager

from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import HTMLResponse
from starlette.templating import Jinja2Templates
import uvicorn

from data_manipulating.model import load_model, load_vectorizer
from data_manipulating.preprocessing import preprocess_corpus
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
    text = text.split(";")

    # Проверка на размер каждого элемента
    for item in text:
        if len(item) > 500:
            raise HTTPException(status_code=400, detail="Each element in the text should not exceed 500 characters.")

    preprocessed_text = preprocess_corpus(text, lemma=True)
    vectorized_text = app.vectorizer.transform(preprocessed_text)
    predicted_forest = (app.model.predict_proba(vectorized_text)[:, 1] >= app.threshold).astype(int)
    result_list_forest = ["non-propaganda" if x == 0 else "propaganda" for x in predicted_forest]
    results = zip(text, result_list_forest)

    return templates.TemplateResponse("result.html", {"request": request, "results": list(results)})


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return templates.TemplateResponse("error.html", {"request": request, "detail": exc.detail}, status_code=exc.status_code)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
