smooth_user_topic = 25
smooth_topic = 500
smooth_post = 100
smooth_user = 30
K = 5

def smoothed_ctr(likes, views, prior, smooth):
    return (likes + smooth * prior) / (views + smooth)

def add_post_ctr(df, global_ctr):
    df['topic_ctr'] = smoothed_ctr(df['topic_likes'], df['topic_views'], global_ctr, smooth_topic)
    df['post_ctr'] = smoothed_ctr(df['post_likes'], df['post_views'], df['topic_ctr'], smooth_post)

    return df
    
def add_user_ctr(df, user_ctr_prior):
    df['user_topic_ctr'] = smoothed_ctr(df['user_topic_likes'], df['user_topic_views'], df['topic_ctr'], smooth_user_topic)
    df['user_ctr'] = smoothed_ctr(df['user_likes'], df['user_views'], user_ctr_prior, smooth_user)

    return df