from langchain_community.chat_models import ChatOllama
from langchain_classic.schema import HumanMessage
from typing import Dict


class RussianTextClassifier:
    def __init__(self):
        self.toxicity_model = ChatOllama(
            model='ru-toxicity-classifier',
            temperature=0.1,
            num_predict=20,
        )

    def classify_toxicity(self, text: str) -> Dict:
        """Базовый метод классификации"""
        try:
            response = self.toxicity_model.invoke([HumanMessage(content=text)])
            category = response.content.strip().upper()
            return {
                'text': text,
                'category': category,
                'success': True
            }
        except Exception as e:
            return {
                'text': text,
                'category': 'ERROR',
                'success': False,
                'error': str(e)
            }


def test_classifier():
    classifier = RussianTextClassifier()

    russian_texts = [
        "Ненавижу этих тупых политиков! Они все воры!",
        "Отличная погода сегодня, настроение прекрасное!",
        "Python - лучший язык программирования для анализа данных",
        "Смотрите акцию: iPhone со скидкой 50%!",
        "Мой логин: admin, пароль: secret123"
    ]

    for text in russian_texts:
        result = classifier.classify_toxicity(text)
        print(f"{text} -> {result['category']}")


if __name__ == '__main__':
    test_classifier()
