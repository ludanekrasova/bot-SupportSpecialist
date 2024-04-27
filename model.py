import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import re
import os

import spacy
from pyaspeller import YandexSpeller

from datasets import load_metric, Dataset
import datasets
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import TrainingArguments, Trainer

#===============================================================================
# Загрузка моделей
#===============================================================================
speller = YandexSpeller()

# варианты: схожесть tf-idf веторов или классификатор bert
# model_type = "similarity"
model_type = "bert"

if model_type == "similarity":
    lemmatizer = spacy.load('ru_core_news_md', disable = ['parser', 'ner'])
    with open("model/stopwords.txt") as f: stopwords_nltk = f.read().splitlines()
    tfidf = joblib.load('model/tfidf.pkl')

#===============================================================================

elif model_type == "bert":
    #tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    tokenizer = BertTokenizer.from_pretrained('model/checkpoint-bert')

    best_model = BertForSequenceClassification.from_pretrained('model/checkpoint-bert', local_files_only=True)

    trainer = Trainer(best_model)

#===============================================================================
# Загрузка данных
#===============================================================================

# классы с ответами
train_answer_class = pd.read_csv('model/answer_class.csv', sep=',', index_col=None)

# классы с эмбедингами вопросов
train = joblib.load('model/train.pkl')

# извлечение эмбедингов и классов
# tfidf_embed = np.load('model/tfidf_embed.npy') если по отдельности
tfidf_embed = train['tfidf_embed']
classes = train['classes']

# словарь перевода классов в группы
label_dist = {0: 'Документы', 
             1: 'Документы',
             2: 'Документы',
             3: 'Документы',
             4: 'Документы',
             5: 'Организация уроков',
             6: 'Организация уроков',
             7: 'Организация уроков',
             8: 'Организация уроков',
             9: 'Организация уроков',
             10: 'Оценки', 
             11: 'Перевод/ запись в группу',
             12: 'Практические работы',
             13: 'Программа обучения', 
             14: 'Портал',
             15: 'Портал',
             16: 'Портал',
             17: 'Портал',
             18: 'Портал',
             19: 'Программа обучения', 
             20: 'Программа обучения', 
             21: 'Расписание',
             22: 'Расписание',
             23: 'Требования ПО',
             24: 'Требования ПО',
             25: 'Трудоустройство',
             26: 'Трудоустройство',
             27: 'Трудоустройство',
             28: 'Документы',
             29: 'Документы',
             30: 'Переключить на оператора',
             }
# предел схожести, при котором отдаем ответ
lim_similarities = 0.2

#===============================================================================
# Работа с текстом
#===============================================================================

# очистка текста
def full_clean(text):
    '''подготовка текста к подаче в модель для каждого текста (через applay)'''
    text = speller.spelled(text)
    text=re.sub(r"[^a-zA-Zа-яА-ЯёЁ0-9#]", " ", text)
    text = text.lower()
    text = re.sub(" +", " ", text) #оставляем только 1 пробел
    text = text.strip()
    # токены для моделей
    tokens = [token.lemma_ for token in lemmatizer(text) if token.lemma_ not in stopwords_nltk]
    # для tfidf на вход текст
    text = " ".join(tokens)
    return text, tokens


def tfidf_embeding(model=None, df=None):   
    '''Преобразование текста в мешок слов'''
    if model==None:
        #загрузить если нет
        model = joblib.load('tfidf.pkl')
    else:
        model=model
    X = model.transform(df)
    
    return X.toarray()

#===============================================================================
# Поиск схожих векторов и предсказание класса ответа
#===============================================================================

def find_similarity(query, embeddings, classes, top_k=1):
    # самые близкие вопросы из трейна
    # возвращает класс, категорию, схожесть
    query_embedding = tfidf_embeding(model=tfidf, df=[full_clean(query)[0]])[0]
    cos_similarities = cosine_similarity(query_embedding.reshape(1, -1), np.array(embeddings))
    sorted_indices = np.argsort(cos_similarities[0])[::-1]
    top_class = [classes[idx] for i, idx in enumerate(sorted_indices[0:top_k])]

    if cos_similarities[0][sorted_indices[0]] < lim_similarities:
        top_class = [30]
    
    return top_class[0], label_dist[top_class[0]], cos_similarities[0][sorted_indices[0]]

#===============================================================================
# Классификация дообученным бертом
#===============================================================================

def predict_text(text):
    # предсказание для одного текста
    text = speller.spelled(text)
    
    max_length = 24
    encoding = tokenizer(text, return_tensors="pt", max_length=max_length)
    encoding = {k: v.to(trainer.model.device) for k,v in encoding.items()}

    outputs = trainer.model(**encoding)
    logits = outputs.logits
    # apply sigmoid + threshold
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(logits.squeeze().to("cpu"))
    predictions = np.zeros(probs.shape)
    scores = max(probs).item()
    predictions[np.where(probs == max(probs))] = 1  #один лейбл
    predicted_labels = [idx for idx, label in enumerate(predictions) if label == 1.0]
    answer = train_answer_class[train_answer_class.answer_class==predicted_labels[0]]['Answer'].values[0]
    
    return predicted_labels[0], label_dist[predicted_labels[0]], answer, round(scores, 2)


#===============================================================================
# Предсказание
#===============================================================================

def predict(query):
    # главная функция
    if model_type == "similarity":
        answer_class, category, similarity = find_similarity(query, tfidf_embed, classes, top_k=3)
        answer = train_answer_class[train_answer_class.answer_class==answer_class]['Answer'].values[0]
    
        return answer_class, category, answer, round(similarity, 2)

    elif model_type == "bert":
        return predict_text(query)
