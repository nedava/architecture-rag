from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from retrieval_qa import init_qdrant, build_chain, ask_question
from text_classifier import RussianTextClassifier


@asynccontextmanager
async def lifespan(app: FastAPI):
    vectorstore = init_qdrant()
    chain = build_chain(vectorstore, llm_model_name='llama3.1:8b-insecure')
    app.state.chain = chain
    app.state.classifier = RussianTextClassifier()
    yield

app = FastAPI(lifespan=lifespan)


class Question(BaseModel):
    text: str


class Answer(BaseModel):
    text: str

@app.post("/ask")
def ask(data: Question) -> Answer:
    text_answer, _ = ask_question(chain=app.state.chain, question=data.text)
    classifier_result = app.state.classifier.classify_toxicity(text_answer)
    if classifier_result['success'] is True:
        if classifier_result['category'] in ['TOXIC', 'SPAM', 'SENSITIVE']:
            error_msg = f"Не могу предоставить результат с меткой {classifier_result['category']}"
            return Answer(text=error_msg)

    return Answer(text=text_answer)
