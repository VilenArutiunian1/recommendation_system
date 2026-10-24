import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime

import pandas as pd
from catboost import CatBoostClassifier
from fastapi import FastAPI, Query, Request
from pydantic import BaseModel
from database import postgres_engine

from recommender import Recommender

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get('MODEL_PATH', 'model.cbm')


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info('Creating database engine')
    # The engine is lazy: no connection is opened until the first query
    engine = postgres_engine()

    logger.info('Loading model from %s', MODEL_PATH)
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)

    logger.info('Loading candidates')
    app.state.recommender = Recommender(model, model.feature_names_, engine)
    app.state.posts = pd.read_sql('SELECT id, text, topic FROM public.post', engine).set_index('id')

    logger.info('Service is ready')
    yield
    engine.dispose()


app = FastAPI(lifespan=lifespan)


class PostGet(BaseModel):
    id: int
    text: str
    topic: str


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.get('/post/recommendations/', response_model=list[PostGet])
def recommended_posts(
    request: Request,
    user_id: int = Query(alias='id'),
    time: datetime = Query(),
    limit: int = Query(10, ge=1, le=100),
):
    state = request.app.state
    top = state.recommender.top_k_posts(user_id, k=limit, timestamp=pd.Timestamp(time))
    return state.posts.loc[top['post_id']].reset_index().to_dict('records')