import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from tqdm import tqdm


BERT_MODEL = 'bert-base-multilingual-cased'
BERT_COMPONENTS = 32
BERT_FEATURES = [f'post_bert_{i}' for i in range(BERT_COMPONENTS)]


def post_embeddings(posts, batch_size=32):
    print('Загружаем BERT и токенизатор', flush=True)
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    model = AutoModel.from_pretrained(BERT_MODEL)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    print(f'Считаем эмбеддинги для {len(posts)} постов на {device}', flush=True)

    texts = posts['text'].fillna('').astype(str).tolist()
    embeddings = []
    for start in tqdm(range(0, len(texts), batch_size), desc='BERT', unit='батч'):
        batch = tokenizer(texts[start:start + batch_size], padding=True, truncation=True,
                          max_length=512, return_tensors='pt').to(device)
        with torch.no_grad():
            hidden = model(**batch).last_hidden_state
            mask = batch['attention_mask'].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        embeddings.append(pooled.cpu().numpy())

    return np.concatenate(embeddings).astype('float32')


def add_post_embeddings(posts, pca=None):
    embeddings = post_embeddings(posts)
    if pca is None:
        print(f'Обучаем PCA на {BERT_COMPONENTS} компонент', flush=True)
        pca = PCA(n_components=BERT_COMPONENTS, random_state=42)
        pca.fit(embeddings)
    print('Применяем PCA', flush=True)
    embeddings = pca.transform(embeddings)
    columns = pd.DataFrame(embeddings.astype('float32'), columns=BERT_FEATURES, index=posts.index)
    print('Эмбеддинги готовы', flush=True)
    return pd.concat([posts, columns], axis=1), pca
