import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# Загружаем переменные окружения (если работаем локально)
load_dotenv(verbose=True)

# Загружаем API-ключи через Streamlit secrets (для облачного запуска)
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", None)
USER_AGENT = st.secrets.get("USER_AGENT", None)

# Проверяем, заданы ли API-ключи
if not GROQ_API_KEY:
    st.error("Ошибка: GROQ_API_KEY не задана в переменных окружения.")
    st.stop()
if not USER_AGENT:
    st.error("Ошибка: USER_AGENT не задана в переменных окружения.")
    st.stop()

# Настройка LLM
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.6,
    api_key=GROQ_API_KEY
)

# Настройка эмбеддингов
embeddings_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct")

# Функция загрузки только англоязычного контента
def load_english_pages(urls):
    english_docs = []
    
    for url in urls:
        if not any(lang in url for lang in ["/ru", "/ar", "/es", "/ch"]):  
            loader = WebBaseLoader(url)  # Исправлено: передаем строку
            documents = loader.load()
            english_docs.extend(documents)
    
    return english_docs

# Пример URL, где английские страницы без префиксов
urls = ["https://status.law/about", "https://status.law/ru/about", "https://status.law/ar/contact"]
documents = load_english_pages(urls)

# Разбиваем на фрагменты
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)

# Создание векторного хранилища
vector_store = InMemoryVectorStore.from_documents(chunks, embeddings_model)
retriever = vector_store.as_retriever()

# Промпт для бота
template = """
You are a helpful legal assistant that answers questions based on information from status.law.

Answer accurately and concisely.

Question: {question}

Only use the provided context to answer the question.

Context: {context}
"""

prompt = PromptTemplate.from_template(template)

# История сообщений
def format_history(message_history):
    formatted = ""
    for msg in message_history:
        formatted += f"User: {msg['question']}\nBot: {msg['answer']}\n\n"
    return formatted

message_history = []

# Интерфейс Streamlit
st.set_page_config(page_title="Legal Chatbot", page_icon="🤖")
st.title("🤖 Legal Chatbot")
st.write("Этот бот отвечает на юридические вопросы, используя информацию с сайта status.law.")

# Поле для ввода вопроса
user_input = st.text_input("Введите ваш вопрос:")

if st.button("Отправить"):
    if user_input:
        # Создание цепочки обработки запроса
        chain = (
            RunnableLambda(lambda x: {"context": retriever.get_relevant_documents(x["question"])})  # Исправлено
            | prompt
            | llm
            | StrOutputParser()
        )

        # Запуск цепочки
        response = chain.invoke({"question": user_input})

        # Добавляем в историю сообщений
        message_history.append({"question": user_input, "answer": response})

        # Выводим ответ
        st.write(response)
