from langchain import hub
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

from set_env import llm, vector_store

# Use a prompt template from Langchain hub or define your own
prompt = hub.pull("rlm/rag-prompt")  # or use your own PromptTemplate

# Define application state
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Retrieval step
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"], k=20)
    return {"context": retrieved_docs}

# Generation step
def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

# Build and run the graph
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

result = graph.invoke({"question": "is the student activity fee collected from students solely enrolled in the College of General Studies?"})

print(f'Context: {result["context"]}\n\n')
print(f'Answer: {result["answer"]}')
