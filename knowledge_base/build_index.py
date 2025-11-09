import os
import time

import ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import hashlib


class RAGPreprocessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=40,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            length_function=len,
        )
        self.qdrant_client = QdrantClient("localhost", port=6333)

    def get_embeddings(self, text, model="nomic-embed-text"):
        """Генерирует эмбеддинги через Ollama"""
        try:
            response = ollama.embeddings(model=model, prompt=text)
            return response["embedding"]
        except Exception as e:
            print(f"Ошибка генерации эмбеддингов: {e}")
            return None

    def process_documents_folder(self, folder_path: str) -> list[PointStruct]:
        """Обрабатывает все текстовые файлы в папке"""

        start_time = time.perf_counter()

        all_points = []

        for filename in os.listdir(folder_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(folder_path, filename)

                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()

                # Разделяем текст на чанки
                chunks = self.text_splitter.split_text(text)

                chunks_num = len(chunks)

                # Добавляем метаданные к каждому чанку
                for i, chunk in enumerate(chunks):
                    # Создаем эмбеддинг
                    embedding = self.get_embeddings(chunk)
                    print(f'chunk {i} of {chunks_num} for file {filename} was created')
                    if not embedding:
                        continue

                    # Создаем точку для Qdrant
                    point = PointStruct(
                        id=int(self.create_text_hash(chunk)[:8], 16),  # Хеш как ID
                        vector=embedding,
                        payload={
                            "text": chunk,
                            "source": filename,
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "hash": self.create_text_hash(chunk)
                        }
                    )
                    all_points.append(point)

        end_time = time.perf_counter() - start_time
        print(f"Обработано файлов: {len([f for f in os.listdir(folder_path) if f.endswith('.txt')])}")
        print(f"Всего чанков: {len(all_points)}")
        print(f"Общее время формирования чанков: {end_time}")

        return all_points

    def create_text_hash(self, text: str) -> str:
        """Создает хеш для идентификации чанка"""
        return hashlib.md5(text.encode()).hexdigest()

    def process_and_index_documents(self):
        """Полный пайплайн обработки и индексации"""

        # Создаем коллекцию если не существует
        try:
            self.qdrant_client.create_collection(
                collection_name="documents",
                vectors_config=VectorParams(
                    size=768,
                    distance=Distance.COSINE
                )
            )
        except Exception:
            pass  # Коллекция уже существует

        folder_path = "text_data"

        all_points = self.process_documents_folder(folder_path)

        # Загружаем данные в Qdrant
        self.qdrant_client.upsert(
            collection_name="documents",
            points=all_points
        )


if __name__ == '__main__':
    preprocessor = RAGPreprocessor()
    preprocessor.process_and_index_documents()
