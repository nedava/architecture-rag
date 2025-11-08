from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from retrieval_qa import init_qdrant, build_chain, ask_question


@asynccontextmanager
async def lifespan(app: FastAPI):
    vectorstore = init_qdrant()
    chain = build_chain(vectorstore)
    app.state.chain = chain
    yield

app = FastAPI(lifespan=lifespan)


class Question(BaseModel):
    text: str


class Answer(BaseModel):
    text: str

@app.post("/ask")
def ask(data: Question) -> Answer:
    text_answer, _ = ask_question(chain=app.state.chain, question=data.text)

    return Answer(text=text_answer)
