import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

def get_conversational_chain():
    """Create QA chain with Gemini-1.5-pro."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not available in the provided context just say, "answer is not available in the context", don't provide the wrong answer.\n \n
    Context:\n{context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.1)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def process_query(query):
    """Process general queries for research-assistant.html using FAISS and Gemini."""
    try:
        if not query:
            raise Exception("Query is required")
        
        # Load FAISS vector store
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        db_path = 'db'
        if not os.path.exists(db_path):
            return "No research data available. Please ensure the vector store is initialized."
        
        new_db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(query, k=4)
        
        if not docs:
            return "No relevant information found in the research database."

        # Query with QA chain
        chain = get_conversational_chain()
        response = chain(
            {"input_documents": docs, "question": query},
            return_only_outputs=True
        )
        return response["output_text"]
    except Exception as e:
        raise Exception(f"Error processing query: {str(e)}")