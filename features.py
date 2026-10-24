import pandas as pd
from pandas import DataFrame
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
import numpy as np
from ctr import add_user_ctr, add_post_ctr


def build_dataset(users: DataFrame, posts: DataFrame, feed: DataFrame, split: bool = True, train_rows: int = 5_000_000) -> tuple[DataFrame, DataFrame | None]:
    # only the post topic is needed for counters: adding one column instead of two merges
    feed['post_topic'] = feed['post_id'].map(posts.set_index('post_id')['topic']).astype('category')

    feed['is_like'] = (feed['action'] == 'like').astype('int8')
    feed = feed.sort_values(['timestamp', 'is_like'])
    feed['is_view'] = (feed['action'] == 'view').astype('int8')

    # счётчики считаем по всей истории; int32 вместо int64 — вдвое меньше памяти на 65 млн строк
    by_user_topic = feed.groupby(['user_id', 'post_topic'], observed=True)
    feed['user_topic_views'] = (by_user_topic['is_view'].cumsum() - feed['is_view']).astype('int32')
    feed['user_topic_likes'] = (by_user_topic['is_like'].cumsum() - feed['is_like']).astype('int32')

    by_user = feed.groupby('user_id')
    feed['user_views'] = (by_user['is_view'].cumsum() - feed['is_view']).astype('int32')
    feed['user_likes'] = (by_user['is_like'].cumsum() - feed['is_like']).astype('int32')

    by_topic = feed.groupby('post_topic', observed=True)
    feed['topic_views'] = (by_topic['is_view'].cumsum() - feed['is_view']).astype('int32')
    feed['topic_likes'] = (by_topic['is_like'].cumsum() - feed['is_like']).astype('int32')

    by_post = feed.groupby('post_id')
    feed['post_views'] = (by_post['is_view'].cumsum() - feed['is_view']).astype('int32')
    feed['post_likes'] = (by_post['is_like'].cumsum() - feed['is_like']).astype('int32')

    # а обучаемся только на последних train_rows просмотрах
    views = feed[feed['action'] == 'view'].tail(train_rows)
    liked = feed.loc[feed['is_like'] == 1, ['user_id', 'post_id']].drop_duplicates()
    views = views.merge(liked.assign(target=1), on=['user_id', 'post_id'], how='left')
    views['target'] = views['target'].fillna(0).astype('int8')
    views['post_topic'] = views['post_topic'].astype(str)

    views['weekday'] = views['timestamp'].dt.dayofweek
    views['hour'] = views['timestamp'].dt.hour

    split_date = views['timestamp'].quantile(0.8)
    history_columns = ['user_id', 'is_view', 'is_like']
    if split:
        train = views[views['timestamp'] < split_date]
        test = views[views['timestamp'] >= split_date]
        history = feed.loc[feed['timestamp'] < split_date, history_columns]
    else:
        train = views
        test = None
        history = feed[history_columns]

    global_ctr = history['is_like'].sum() / history['is_view'].sum()
    train_by_user = history.groupby('user_id')
    view_count = train_by_user['is_view'].sum()
    view_count = view_count[view_count > 0]
    user_ctr_prior = (train_by_user['is_like'].sum() / view_count).mean()

    new_column_names = {
        'gender': 'user_gender',
        'age': 'user_age',
        'country': 'user_country',
        'city': 'user_city',
        'exp_group': 'user_exp_group',
        'os': 'user_os',
        'source': 'user_source',
        'text': 'post_text',
    }
    
    posts_info = posts.drop(columns='topic')

    def add_info(df: DataFrame) -> DataFrame:
        df = df.merge(users, on='user_id').merge(posts_info, on='post_id')
        return df.rename(columns=new_column_names)

    train = train.copy().reset_index(drop=True)
    train = add_info(train)
    if split:
        test = test.copy().reset_index(drop=True)
        test = add_info(test)

    add_post_ctr(train, global_ctr)
    add_user_ctr(train, user_ctr_prior)

    if split:
        add_post_ctr(test, global_ctr)
        add_user_ctr(test, user_ctr_prior)

    columns_to_drop = [
        'user_topic_likes',
        'topic_likes',
        'topic_views',
        'post_likes',
        'post_views',
        'user_likes',
        'user_views',
        'user_os',
        'user_source',
        'is_view',
        'is_like',
        'action'
    ]
    train = train.drop(columns=columns_to_drop)
    if split:
        test = test.drop(columns=columns_to_drop)

    return train, test