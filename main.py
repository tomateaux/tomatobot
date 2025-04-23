from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from langchain_core.runnables.history import RunnableWithMessageHistory
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import uuid
import os
from datetime import datetime
import gradio as gr
import atexit

# セッション内における会話履歴の保持
class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    messages: list[BaseMessage] = Field(default_factory=list)
    
    def add_messages(self, messages: list[BaseMessage]) -> None:
        self.messages.extend(messages)
    
    def clear(self) -> None:
        self.messages = []

# セッション内における会話履歴の管理
store = {}
session_id = str(uuid.uuid4())
def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]

# LLMモデルに渡すプロンプト
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant who is good at {ability}."),
    ("system", "The following are past messages, which you should remember and refer to if relevant:\n{context}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

# LLMモデル
model = "llama3.1"
llm = ChatOllama(model=model)

# FAISSの初期化
FAISS_PATH = "./faiss_index"
embeddings = OllamaEmbeddings(model=model)
if os.path.exists(FAISS_PATH):
    vector_store = FAISS.load_local(FAISS_PATH, embeddings=embeddings, allow_dangerous_deserialization=True)
else:
    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

# プロンプトとモデルを接続
chain = prompt | llm

# プロンプトとモデルと会話履歴を接続
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_by_session_id,
    input_messages_key="question",
    history_messages_key="history",
) 

# メッセージに対して、AIの返答を返す
def respond(message, history):
    # FAISSから会話履歴を検索
    related_messages_user = vector_store.similarity_search(message, k=10, filter={"role": "user"})
    related_messages_ai = vector_store.similarity_search(message, k=10, filter={"role": "ai"})
    context = (
        "User previously said:\n" +
        "\n".join(doc.page_content for doc in related_messages_user) +
        "\n\nYou previously said:\n" +
        "\n".join(doc.page_content for doc in related_messages_ai)
    )
    
    # AIの返答を生成
    result = chain_with_history.invoke(
        {
            "ability": "chatting with humans",
            "question": message,
            "context": context
        },
        config={"configurable": {"session_id": session_id}}
    )
    
    # FAISSに会話履歴を保存
    now = datetime.now().isoformat()
    vector_store.add_documents([
        Document(page_content=message, metadata={
            "role": "user",
            "timestamp": now,
            "session_id": session_id
        }),
        Document(page_content=result.content , metadata={
            "role": "ai",
            "timestamp": now,
            "session_id": session_id
        })
    ])
    
    return result.content

# チャットUIの構築
demo = gr.ChatInterface(fn=respond, type="messages", title="tomatobot")

# アプリ終了時に、FAISSの会話履歴を永続保存
@atexit.register
def save_vector_store():
    vector_store.save_local(FAISS_PATH)

# チャットUIの起動
if __name__ == "__main__":
    demo.launch()