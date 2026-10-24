import os
import pandas as pd
from sqlalchemy import URL, create_engine
from sqlalchemy import text
from dotenv import load_dotenv
from functools import lru_cache

@lru_cache
def postgres_engine():
    load_dotenv()
    url = URL.create(
        drivername='postgresql+psycopg2',
        username=os.environ['DB_USER'],
        password=os.environ['DB_PASSWORD'],
        host=os.environ['DB_HOST'],
        port=int(os.environ.get('DB_PORT', 5432)),
        database=os.environ['DB_NAME'],
    )
    return create_engine(url, pool_size=5, max_overflow=10, pool_pre_ping=True)


FEED_CHUNK_ROWS = 1_000_000


def load_tables(engine):
    with engine.connect() as connection:
        users = pd.read_sql(
            text("SELECT id AS user_id, gender, age, country, city, exp_group, os, source FROM public.user"),
            connection,
        )
        posts = pd.read_sql(text("SELECT id AS post_id, text, topic FROM public.post"), connection)

    # 65 млн строк одним pd.read_sql в память не влезут: читаем потоком кусками и сразу ужимаем типы
    action_type = pd.CategoricalDtype(['view', 'like'])
    chunks = []
    with engine.connect().execution_options(stream_results=True) as connection:
        for chunk in pd.read_sql(
            text("SELECT user_id, post_id, action, time AS timestamp FROM public.feed_action"),
            connection,
            chunksize=FEED_CHUNK_ROWS,
        ):
            chunks.append(chunk.astype({'user_id': 'int32', 'post_id': 'int32', 'action': action_type}))
    feed = pd.concat(chunks, ignore_index=True)

    return users, posts, feed
