import logging
import uuid

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

from typing import Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from langchain_core.messages import trim_messages
from langchain_openai import ChatOpenAI
from src.Prompt import prompt


class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    question: str
    context: str


def call_model(state: State):
    chain = prompt | model
    trimmed_messages = trimmer.invoke(state["messages"])
    response = chain.invoke(
        {"messages": trimmed_messages, "question": state["question"], "context": state["context"]}
    )
    return {"messages": [response]}

model = ChatOpenAI(model="gpt-4o-mini")

workflow = StateGraph(state_schema=State)

trimmer = trim_messages(
    max_tokens=1000,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
