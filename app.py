import gradio as gr
from langchain.chains import RetrievalQA
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline

# Load data
loader = CSVLoader(file_path="data/starbucks_sales.csv")
documents = loader.load()

# Initialize embeddings and vector store
embeddings = HuggingFaceEmbeddings()
db = FAISS.from_documents(documents, embeddings)

# Load LLM (Replace with your model path)
llm = HuggingFacePipeline.from_model_id(
    model_id="meta-llama/Llama-2-7b-chat-hf",
    task="text-generation",
    pipeline_kwargs={"max_length": 500},
)

# Build QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(),
)

# Gradio interface
def ask_agent(question):
    return qa_chain.run(question)

iface = gr.Interface(
    fn=ask_agent,
    inputs=gr.Textbox(lines=2, placeholder="Ask about inventory..."),
    outputs="text",
    title="🛒 AI Supply Chain Assistant",
)
iface.launch()