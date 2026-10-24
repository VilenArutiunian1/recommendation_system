import pandas as pd
from sqlalchemy import text
from ctr import K, add_post_ctr, add_user_ctr

class Recommender:
    REFRESH_SECONDS = 24 * 60 * 60

    def __init__(self, model, features, engine):
        self.engine = engine
        self.model = model
        self.features = features
        self.candidates: pd.DataFrame | None = None
        self.user_ctr_prior: float | None = None
        self._load_user_ctr_prior()
        self.global_ctr: float | None = None
        self.candidates = self._load_candidates()


    def _load_user_ctr_prior(self):
        query = """
            WITH users_stats AS (
                SELECT 
                    user_id,
                    COUNT(CASE WHEN fa.action = 'view' THEN 1 END) as user_views,
                    COUNT(CASE WHEN fa.action = 'like' THEN 1 END) as user_likes
                FROM public.feed_action fa
                GROUP BY fa.user_id
            )

            SELECT 
                AVG(CAST(user_likes AS FLOAT) / user_views) AS user_ctr_prior 
            FROM users_stats
            WHERE user_views > 0
        """
        user_ctr_prior = pd.read_sql(text(query), self.engine)['user_ctr_prior'].iloc[0]
        self.user_ctr_prior = user_ctr_prior


    def _load_candidates(self):
        query = """
            WITH post_stats AS (
                SELECT 
                    p.id as post_id, 
                    p.topic as post_topic, 
                    COUNT(CASE WHEN fa.action = 'view' THEN 1 END) as post_views,
                    COUNT(CASE WHEN fa.action = 'like' THEN 1 END) as post_likes
                FROM public.post p
                LEFT JOIN public.feed_action fa on fa.post_id = p.id
                GROUP BY p.id, p.topic
            )

            SELECT 
                post_id, 
                post_topic, 
                post_views, 
                post_likes,
                SUM(ps.post_views) OVER (PARTITION BY ps.post_topic) as topic_views,
                SUM(ps.post_likes) OVER (PARTITION BY ps.post_topic) as topic_likes
            FROM post_stats ps
        """

        stats = pd.read_sql(text(query), self.engine)
        self.global_ctr = stats['post_likes'].sum() / stats['post_views'].sum()
        return stats


    def _build_features(self, user_id, timestamp):
        with self.engine.connect() as connection:
            query = """
                SELECT 
                    u.id as user_id,
                    u.gender AS user_gender,
                    u.age AS user_age,
                    u.country AS user_country,
                    u.city AS user_city,
                    u.exp_group AS user_exp_group,
                    u.os AS user_os,
                    u.source AS user_source,
                    COUNT(CASE WHEN fa.action = 'view' THEN 1 END) AS user_views,
                    COUNT(CASE WHEN fa.action = 'like' THEN 1 END) AS user_likes
                FROM public.user u
                LEFT JOIN public.feed_action fa ON fa.user_id = u.id
                WHERE u.id = :id
                GROUP BY u.id, u.gender, u.age, u.country, u.city, 
                u.exp_group, u.os, u.source
            """

            user_stats = pd.read_sql(text(query), connection, params={ 'id': user_id })

            if len(user_stats) == 0:
                return None

            query = """
                SELECT
                    p.topic AS post_topic,
                    COUNT(CASE WHEN fa.action = 'view' THEN 1 END) AS user_topic_views,
                    COUNT(CASE WHEN fa.action = 'like' THEN 1 END) AS user_topic_likes 
                FROM public.feed_action fa
                JOIN public.post p ON fa.post_id = p.id
                WHERE fa.user_id = :id
                GROUP BY fa.user_id, p.topic
            """

            user_topic_stats = pd.read_sql(text(query), connection, params={ 'id': user_id })

            df = self.candidates.merge(user_topic_stats, on='post_topic', how='left').merge(user_stats, how='cross')

            cols = ['user_topic_views', 'user_topic_likes']
            df[cols] = df[cols].fillna(0)

            features = add_post_ctr(df, self.global_ctr)
            features = add_user_ctr(df, self.user_ctr_prior)
        
            features['hour'] = timestamp.hour
            features['weekday'] = timestamp.dayofweek

            return features


    def _seen_posts(self, user_id):
        query = """
            SELECT fa.post_id
            FROM public.feed_action fa
            WHERE fa.user_id = :id
            GROUP BY fa.post_id
        """

        return pd.read_sql(text(query), self.engine, params={'id': user_id})['post_id']


    def _popular(self, k):
        df = add_post_ctr(self.candidates.copy(), self.global_ctr)
        
        return (df.nlargest(k, 'post_ctr')[['post_id', 'post_topic']]
            .assign(p=None)
            .reset_index(drop=True))


    def top_k_posts(self, user_id, k, timestamp=None):
        timestamp = timestamp if timestamp is not None else pd.Timestamp.now()
        features = self._build_features(user_id, timestamp=timestamp)

        if features is None:
            return self._popular(k)
    
        seen = self._seen_posts(user_id)
        features = features[~features['post_id'].isin(seen)].copy()
        if features.empty:
            return self._popular(k)
    
        features['p'] = self.model.predict_proba(features[self.features])[:, 1]
        return features.nlargest(k, 'p')[['post_id', 'post_topic', 'p']]
        

