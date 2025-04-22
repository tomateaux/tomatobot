from langchain_ollama import ChatOllama
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from langchain_core.runnables.history import RunnableWithMessageHistory
import uuid
import gradio as gr

# 会話履歴の保持
class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    messages: list[BaseMessage] = Field(default_factory=list)
    
    def add_messages(self, messages: list[BaseMessage]) -> None:
        self.messages.extend(messages)
    
    def clear(self) -> None:
        self.messages = []

# ユーザーごとの会話履歴
store = {}
session_id = str(uuid.uuid4())
def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]

# LLMモデルに渡すプロンプト
prompt = ChatPromptTemplate.from_messages([
    ("system", "You're an assistant who's good at {ability}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

# LLMモデル
llm = ChatOllama(model="llama3.1")

# プロンプトとモデルを接続
chain = prompt | llm

# プロンプトとモデルと会話履歴を接続
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_by_session_id,
    input_messages_key="question",
    history_messages_key="history",
) 

# メッセージに対して、モデルの返答を返す
def respond(message, history):
    result = chain_with_history.invoke(
        {"ability": "chatting with humans", "question": message},
        config={"configurable": {"session_id": session_id}}
    )
    return result.content

# チャットUIの構築
demo = gr.ChatInterface(fn=respond, type="messages", title="tomatobot")

# チャットUIの起動
if __name__ == "__main__":
    demo.launch()