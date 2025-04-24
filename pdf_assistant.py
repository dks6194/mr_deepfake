import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Initialize temporary store (in-memory)
temporary_db = None
temporary_texts = []

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

def save_pdf_to_vector_db(file_path, storage_type):
    """Save PDF embeddings to temporary or permanent FAISS store."""
    global temporary_db, temporary_texts
    try:
        # Extract text
        reader = PdfReader(file_path)
        text = ''
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + '\n'
        if not text.strip():
            raise Exception("No text extracted from PDF")

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        if not chunks:
            raise Exception("No text chunks created")

        # Generate embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        if storage_type == 'temporary':
            # Create new in-memory FAISS index
            temporary_texts = chunks
            temporary_db = FAISS.from_texts(chunks, embeddings)
        else:  # permanent
            # Load or create permanent FAISS store
            db_path = 'db'
            if os.path.exists(db_path):
                existing_db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
                # Merge new documents
                new_db = FAISS.from_texts(chunks, embeddings)
                existing_db.merge_from(new_db)
                existing_db.save_local(db_path)
            else:
                # Create new permanent store
                #FAISS.from_texts(chunks, embeddings).save_local(db_path)
                pass

    except Exception as e:
        raise Exception(f"Error saving PDF to {storage_type} vector store: {str(e)}")

def temp_query(query):
    """Query the temporary FAISS store."""
    try:
        global temporary_db, temporary_texts
        if temporary_db is None or not temporary_texts:
            return "No temporary data available. Please upload a PDF and set temporary storage."

        # Perform similarity search
        docs = temporary_db.similarity_search(query, k=4)
        if not docs:
            return "No relevant information found in temporary storage."

        # Query with QA chain
        chain = get_conversational_chain()
        response = chain(
            {"input_documents": docs, "question": query},
            return_only_outputs=True
        )
        return response["output_text"]
    except Exception as e:
        raise Exception(f"Error querying temporary store: {str(e)}")

def permanent_query(query):
    """Query the permanent FAISS store."""
    try:
        db_path = 'db'
        if not os.path.exists(db_path):
            return "No permanent data available. Please upload a PDF and set permanent storage."

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)

        # Perform similarity search
        docs = db.similarity_search(query, k=4)
        if not docs:
            return "No relevant information found in permanent storage."

        # Query with QA chain
        chain = get_conversational_chain()
        response = chain(
            {"input_documents": docs, "question": query},
            return_only_outputs=True
        )
        return response["output_text"]
    except Exception as e:
        raise Exception(f"Error querying permanent store: {str(e)}")