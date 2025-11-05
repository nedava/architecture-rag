from knowledge_base.build_index import RAGPreprocessor


if __name__ == '__main__':
    preprocessor = RAGPreprocessor()

    queries = [
        'Катенька испытывает к Владимиру',
    ]

    for query in queries:
        query_embedding = preprocessor.get_embeddings(query)
        # Ищем в Qdrant
        search_result = preprocessor.qdrant_client.search(
            collection_name='documents',
            query_vector=query_embedding,
            limit=4,
        )
        if search_result:
            print(f"Вопрос: {query}")
            print("Ответы:")
            for i in search_result:
                print(i.payload['text'])
