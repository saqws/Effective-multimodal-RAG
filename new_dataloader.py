import os
import pyarrow.parquet as pq
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import io
import random
import numpy as np
from typing import List, Dict, Tuple, Literal
import json
import logging


TaskType = Literal["I-T", "T-I", "IT-T", "T-IT", "I-IT", "IT-I", "T-T"]

class MultimodalRetrievalDataset(Dataset):
    def __init__(self, 
        dataset_path: str, 
        task_type: TaskType,
        num_negatives: int = 3,
        dataset_percents: int = 1
    ):
        """
        Args:
            dataset_path: Путь к директории с датасетом
            task_type: Тип задачи мультимодального поиска
            num_negatives: Количество негативных примеров
        """
        self.dataset_path = dataset_path
        self.task_type = task_type
        self.num_negatives = num_negatives
        self.dataset_percents = dataset_percents
        
        self._load_data()
        self._build_indices()
    
    def _load_image_from_field(self, field):
        """
        Преобразует поле из parquet в PIL.Image
        Ожидается словарь {'bytes': b'...'}
        """
        if isinstance(field, dict) and 'bytes' in field:
            try:
                return Image.open(io.BytesIO(field['bytes'])).convert('RGB')
            except Exception as e:
                logging.error(f"Невозможно декодировать изображение: {e}")
                return None
        else:
            return None
    
    def _load_data(self):
        """Загружает данные из parquet-файлов"""
        logging.debug("start data loading")
        corpus_files = [f for f in os.listdir(self.dataset_path) if f.startswith('corpus')]
        self.corpus = []
        logging.debug("got corp")
        for file in corpus_files:
            table = pq.read_table(os.path.join(self.dataset_path, file))
            self.corpus.append(table.to_pandas())
        self.corpus = pd.concat(self.corpus, ignore_index=True)
        logging.debug("corp loaded")
        self.queries = pq.read_table(os.path.join(self.dataset_path, 'query.parquet')).to_pandas()
        logging.debug("quer loaded")
        self.qrels = pq.read_table(os.path.join(self.dataset_path, 'qrels.parquet')).to_pandas()
        logging.debug("qrel loaded")
    
    def _build_indices(self):
        """Создает индексы для быстрого поиска"""
        self.corpus_id_to_idx = {corpus_id: idx for idx, corpus_id in enumerate(self.corpus['id'])}
        self.query_id_to_idx = {query_id: idx for idx, query_id in enumerate(self.queries['id'])}
        
    def _get_negatives(self, query_id: str, exclude_corpus_id: str) -> Tuple[List[str], List[Image.Image]]:
        """Оптимизированная версия для больших корпусов"""
        neg_texts = []
        neg_images = []
        selected_indices = set()
    
        # Получаем индекс исключаемого документа (чтобы не проверять corpus_id каждый раз)
        exclude_idx = self.corpus_id_to_idx.get(exclude_corpus_id, -1)
    
        # Размер корпуса
        corpus_size = len(self.corpus)
    
        # Нужно выбрать min(3, corpus_size - 1) уникальных негативов
        num_negs = min(3, corpus_size - 1) if corpus_size > 1 else 0
    
        while len(selected_indices) < num_negs:
            # Выбираем случайный индекс
            random_idx = random.randint(0, corpus_size - 1)
            
            # Пропускаем исключаемый документ и уже выбранные
            if random_idx != exclude_idx and random_idx not in selected_indices:
                selected_indices.add(random_idx)
                row = self.corpus.iloc[random_idx]
                
                # Добавляем текст и/или изображение
                if 'text' in row:
                    neg_texts.append(row['text'])
                if 'image' in row:
                    neg_images.append(self._load_image_from_field(row['image']))
                    
        return neg_texts, neg_images
    
    def __len__(self):
        return int(len(self.qrels) * self.dataset_percents)
    
    def __getitem__(self, idx):
        qrel = self.qrels.iloc[idx]
        query_id = qrel['query-id']
        corpus_id = qrel['corpus-id']
        
        # Получаем запрос
        query_idx = self.query_id_to_idx[query_id]
        query_row = self.queries.iloc[query_idx]
        
        # Получаем релевантный документ
        corpus_idx = self.corpus_id_to_idx[corpus_id]
        doc_row = self.corpus.iloc[corpus_idx]

        pos_image = None
        # Декодирование изображений
        if 'image' in doc_row and pd.notnull(doc_row['image']):
            pos_image = self._load_image_from_field(doc_row['image'])
        
        query_image = None
        if 'image' in query_row and pd.notnull(query_row['image']):
            query_image = self._load_image_from_field(query_row['image'])
        
        # Получаем негативные примеры
        neg_texts, neg_images = self._get_negatives(query_id, corpus_id)
        
        
        # Проверки
        if 'text' in query_row and pd.notnull(query_row['text']):
            input_text = query_row['text']
        else:
            input_text = None
        
        if 'text' in doc_row and pd.notnull(doc_row['text']):
            pos_text = doc_row['text']
        else:
            pos_text = None

        return {
            'task_type' : self.task_type,
            'input_text': input_text,
            'input_image': query_image,
            'pos_text': pos_text,
            'pos_image': pos_image,
            'neg_texts': neg_texts,
            'neg_images': neg_images
        }


# Единая функция для MultimodalRetrievalDataset и CSVMultimodalRetrievalDataset
def create_multimodal_dataloader(
    dataset_path: str,
    task_type: TaskType,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    num_negatives: int = 5,
    multi_captions: bool | None = None,
    dataset_percents: float = 1
) -> DataLoader:
    """Создает DataLoader для мультимодального поиска"""
    if os.path.exists(os.path.join(dataset_path, "captions.csv")) and task_type in ["T-I", "I-T"]:
        if "flickr" in dataset_path:
            multi_captions  = True
        dataset = CSVMultimodalRetrievalDataset(
            dataset_path=dataset_path,
            task_type=task_type,
            num_negatives=num_negatives,
            multi_captions=multi_captions,
            dataset_percents=dataset_percents
        )
    else:
         dataset = MultimodalRetrievalDataset(
            dataset_path=dataset_path,
            task_type=task_type,
            num_negatives=num_negatives,
            dataset_percents=dataset_percents
        )
       
    def collate_fn(batch):
        """Обрабатывает батч с переменным числом негативов"""
        collated = {
            'input_text': [item['input_text'] for item in batch],
            'input_image': [item['input_image'] for item in batch],
            'pos_text': [item['pos_text'] for item in batch],
            'pos_image': [item['pos_image'] for item in batch],
            'neg_texts': [item['neg_texts'] for item in batch],
            'neg_images': [item['neg_images'] for item in batch]
        }
        return collated
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    
class CSVMultimodalRetrievalDataset(Dataset):
    def __init__(self, 
        dataset_path: str, 
        task_type: TaskType,
        num_negatives: int = 3,
        multi_captions: bool = False,
        dataset_percents: int = 1):
        """
        Args:
            dataset_path: Путь к директории с датасетом
            task_type: Тип задачи мультимодального поиска
            num_negatives: Количество негативных примеров
        """
        self.dataset_path = dataset_path
        self.task_type = task_type
        self.num_negatives = num_negatives
        self.multi_captions = multi_captions
        
        self._load_data(dataset_percents)
     
    def _load_data(self, dataset_percents):
        # """Загружает данные из csv-файлов"""
        logging.debug("Start reading csv")
        self.data = pd.read_csv(
            os.path.join(self.dataset_path, "captions.csv"),
            skiprows=lambda i: i > 0 and np.random.rand() > dataset_percents 
        )
        logging.debug("End reading csv")
        self.images = []
        for i in range(len(self.data)):
            self.images.append(self.load_image_from_file(self.data.iloc[i]["file_name"]))
    
    def _get_negative_images(self, idx: int) -> list[Image.Image]:
        # print("Getting neg images")
        negative_indexes = self._generate_negative_indexes(idx)
        return [self.images[negative_index] for negative_index in negative_indexes]
    
    def _get_positive_image(self, idx: int) -> Image.Image:
        # print("Getting pos images")
        return self.images[idx]
    
    def _generate_negative_indexes(self, idx: int) -> list[int]:
        negative_indexes = []
        while len(negative_indexes) < self.num_negatives:
            new_index = random.randint(0, len(self.data) - 1)
            if new_index == idx or new_index in negative_indexes:
                continue
            negative_indexes.append(new_index)
        return negative_indexes

    def _get_negative_captions(self, idx: int) -> list[str]:
        if not self.multi_captions:
            return list(map(lambda x: self.data.iloc[x]["caption"], self._generate_negative_indexes(idx)))

        negative_captions = []
        list_captions = json.loads(self.data.iloc[idx]["caption"])
        for i in range(1, len(list_captions)):
            negative_captions.append(list_captions[i])
        used_indexes = [idx]

        while len(negative_captions) < self.num_negatives:
            new_index = random.randint(0, len(self.data) - 1)
            if new_index in used_indexes:
                continue

            list_captions = json.loads(self.data.iloc[idx]["caption"])
            negative_captions.append(list_captions[random.randint(0, len(list_captions) - 1)])
        return negative_captions

    def _get_positive_caption(self, idx: int) -> list[str]:
        # print("getting caption")
        if not self.multi_captions:
            return self.data.iloc[idx]["caption"]
        return json.loads(self.data.iloc[idx]["caption"])[0]

    def __len__(self):
        return len(self.data)
    
    def load_image_from_file(self, filename: str) ->Image.Image:
        """
        Преобразует поле из parquet в PIL.Image
        """
        filepath = os.path.join(self.dataset_path, "images", filename)
        if not os.path.exists(filepath):
            logging.error(f"Не найдено изображение {filepath}")
            return None
        try:
            return Image.open(filepath).convert('RGB')
        except Exception as e:
            logging.error(f"[Ошибка] Невозможно декодировать изображение: {e}")
            return None

    def __getitem__(self, idx):
        if self.task_type not in ["T-I", "I-T"]:
            raise ValueError(f"Unknown task type: {self.task_type}. It should be I-T or T-I")
        
        if self.task_type == 'T-I':
            # Input: Text, Positive: Image
            return {
                'task_type' : self.task_type,
                'input_text': self._get_positive_caption(idx),
                'input_image': None,
                'pos_text': None,
                'pos_image': self._get_positive_image(idx),
                'neg_texts': [],
                'neg_images': self._get_negative_images(idx)
            }
        elif self.task_type == 'I-T':
            # Input: Image, Positive: Text
            return {
                'task_type' : self.task_type,
                'input_text': None,
                'input_image': self._get_positive_image(idx),
                'pos_text': self._get_positive_caption(idx),
                'pos_image': None,
                'neg_texts': self._get_negative_captions(idx),
                'neg_images': []
            }
    
