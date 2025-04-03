import gradio as gr
import lancedb

from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import LanceDB
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI


KNOWLEDGE_BASES= {
    "WIZARD_OF_OZ": "knowledge_bases/wizard_of_oz",
    "ODDYSEY": "knowledge_bases/oddysey",
    "PRAWO_O_RUCHU_DROGOWYM": "knowledge_bases/prawo_o_ruchu_drogowym"
}

def get_vector_store_for_kb(knowledge_base: str):
    """Get vector store for a given knowledge base."""
    if knowledge_base not in KNOWLEDGE_BASES:
        raise ValueError(f"Unknown knowledge base: {knowledge_base}")

    ODDYSEY = KNOWLEDGE_BASES[knowledge_base]

    # Connect to LanceDB
    connection = lancedb.connect(ODDYSEY)
    return LanceDB(
        connection=connection,
        embedding=OpenAIEmbeddings(),
        table_name="knowledge_base"
    )


# Define graph state
class RAGState(TypedDict):
    no_snippets: int
    knowledge_base: str
    question: str
    snippets: list
    answer: str


# Node #1 - retrieval
def retrieve_snippets(state: RAGState):
    no_snippets = state.get("no_snippets", 3)
    vector_store = get_vector_store_for_kb(state["knowledge_base"])
    retriever = vector_store.as_retriever(search_kwargs={"k": no_snippets})
    snippets = retriever.get_relevant_documents(state["question"])
    return {"snippets": snippets}


# Node #2 - generation
def generate_answer(state: RAGState):
    context = "\n\n".join(f"text: {doc.page_content}\n page: {doc.metadata['page_no']}" for doc in state["snippets"])
    prompt = f"Using the context below, answer the question and provide the information which page was used for creating the answer.\n\nQuestion: {state['question']}\nContext: {context}\nAnswer:"
    answer = ChatOpenAI(temperature=0.5, model_name="gpt-4o-mini").invoke(prompt).content
    return {"answer": answer}

# Build the state graph with sequential nodes
graph_builder = StateGraph(RAGState).add_sequence([retrieve_snippets, generate_answer])
graph_builder.add_edge(START, "retrieve_snippets")
graph_builder.add_edge("generate_answer", END)
graph = graph_builder.compile()


def answer_the_question(
    question: str,
    no_snippets: int,
    knowledge_base: str
):
    """Generate answer for a given question."""
    state = {
        "question": question,
        "no_snippets": no_snippets,
        "knowledge_base": knowledge_base
    }
    result = graph.invoke(state)
    return result["answer"]


with gr.Blocks() as application:
    with gr.Row():
        question = gr.TextArea(label="Question", placeholder="Enter your question here", lines=2)
    with gr.Row():
        with gr.Column(scale=2):
            no_snippets = gr.Slider(label="Number of snippets", minimum=1, maximum=10, value=3)
        with gr.Column(scale=2):
            knowledge_base = gr.Dropdown(label="Knowledge Base", choices=list(KNOWLEDGE_BASES.keys()), value="WIZARD_OF_OZ")
    with gr.Row():
        submit = gr.Button("Submit", variant="primary")
    with gr.Row():
        answer = gr.TextArea(label="Answer", placeholder="Answer will be displayed here", lines=2, interactive=False)

    submit.click(answer_the_question, inputs=[question, no_snippets, knowledge_base], outputs=answer)

application.launch()

