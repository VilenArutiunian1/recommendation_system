import argparse
from catboost import CatBoostClassifier, CatBoostRanker
from database import load_tables
from features import build_dataset
from sklearn.metrics import roc_auc_score

CAT_FEATURES = [
    'user_gender', 'user_country', 'user_city', 'user_exp_group', 'post_topic'
]
NUM_FEATURES = ['user_age', 'topic_ctr', 'user_topic_ctr', 'user_topic_views', 'post_ctr', 'user_ctr', 'hour', 'weekday']
FEATURES = CAT_FEATURES + NUM_FEATURES

CATBOOST_CLASSIFIER_ITERATIONS=500
CATBOOST_CLASSIFIER_LEARNING_RATE=0.1
CATBOOST_CLASSIFIER_DEPTH=6
CATBOOST_RANKER_ITERATIONS=500
CATBOOST_RANKER_LEARNING_RATE=0.1
CATBOOST_RANKER_DEPTH=6


def fit_classifier(df):
    model = (
        CatBoostClassifier(
            iterations=CATBOOST_CLASSIFIER_ITERATIONS,
            learning_rate=CATBOOST_CLASSIFIER_LEARNING_RATE,
            depth=CATBOOST_CLASSIFIER_DEPTH,
            random_seed=42,
            verbose=100,
            thread_count=-1
        )
    )

    model.fit(df[FEATURES], df['target'], cat_features=CAT_FEATURES)
    return model

def fit_ranker(df):
    model = (
        CatBoostRanker(
            iterations=CATBOOST_RANKER_ITERATIONS,
            learning_rate=CATBOOST_RANKER_LEARNING_RATE,
            depth=CATBOOST_RANKER_DEPTH,
            random_seed=42,
            verbose=100,
            thread_count=-1
        )
    )

    model.fit(df[FEATURES], df['target'], cat_features=CAT_FEATURES)
    return model


def group_auc(df, group_col='user_id', target_col='target', score_col='p'):
    # AUC внутри группы через ранги: (сумма рангов позитивов - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    ranks = df.groupby(group_col)[score_col].rank(method='average')
    is_pos = df[target_col] == 1

    stats = (
        df.assign(pos_rank=ranks.where(is_pos, 0), is_pos=is_pos.astype('int64'))
        .groupby(group_col)
        .agg(n=('is_pos', 'size'), n_pos=('is_pos', 'sum'), pos_rank_sum=('pos_rank', 'sum'))
    )
    stats['n_neg'] = stats['n'] - stats['n_pos']
    # в группах с одним классом AUC не определён
    stats = stats[(stats['n_pos'] > 0) & (stats['n_neg'] > 0)]

    auc = (stats['pos_rank_sum'] - stats['n_pos'] * (stats['n_pos'] + 1) / 2) / (stats['n_pos'] * stats['n_neg'])
    return (auc * stats['n']).sum() / stats['n'].sum()


def validate_classifier(users, posts, feed):
    train_df, test_df = build_dataset(users, posts, feed)
    model = fit_classifier(train_df)

    y_train_proba_pred = model.predict_proba(train_df[FEATURES])[:, 1]
    y_test_proba_pred = model.predict_proba(test_df[FEATURES])[:, 1]

    train_df = train_df.assign(p=y_train_proba_pred)
    test_df = test_df.assign(p=y_test_proba_pred)

    train_roc_auc = roc_auc_score(train_df['target'], y_train_proba_pred)
    test_roc_auc = roc_auc_score(test_df['target'], y_test_proba_pred)
    train_group_auc = group_auc(train_df)
    test_group_auc = group_auc(test_df)
    metrics = {
        'train_roc_auc': train_roc_auc,
        'test_roc_auc': test_roc_auc,
        'train_group_auc': train_group_auc,
        'test_group_auc': test_group_auc
    }

    return (
        model,
        metrics,
        train_df,
        test_df
    )


def train_final_classifier(users, posts, feed):
    full_df, _ = build_dataset(users, posts, feed, split=False)
    model = fit_classifier(full_df)
    # cbm - catboost model
    model.save_model('model.cbm')
    return model

