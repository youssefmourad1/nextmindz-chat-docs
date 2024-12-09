import os
import tempfile
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import fitz  # PyMuPDF
import base64
from langchain.schema import Document
from groq import Groq  # Commented out as we are no longer using Groq's Llama Vision
import easyocr
from PIL import Image
import numpy as np
import pytesseract
from io import BytesIO

# Set embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Initialize embeddings
hf_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
# Initialize ChatGroq LLM
llm = ChatGroq(
    api_key = GROQ_API_KEY,
    model="llama-3.1-70b-versatile",
    temperature=0.7,
    max_tokens=8000,
    max_retries=2,
)

# Function to encode the image
def encode_image(image_data):
    return base64.b64encode(image_data).decode('utf-8')

# Function to process document images using OCR
def process_with_ocr(image_data):
    reader = easyocr.Reader(['fr'], gpu=False)
    image = Image.open(BytesIO(image_data))
    image_np = np.array(image)
    result = reader.readtext(image_np, detail=0, paragraph=True)
    text = '\n'.join(result)
    return text

# Function to extract text from PDF using OCR
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        img_bytes = pix.tobytes("png")  # Ensure the image is in PNG format
        try:
            page_text = process_with_ocr(img_bytes)
            text += page_text + "\n"
        except Exception as e:
            st.error(f"Error processing page {page_num + 1}: {e}")
            continue
    return text

# Function to handle file processing and create vector database
def process_files(uploaded_files):
    extracted_texts = []
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.' + file.type.split('/')[-1]) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        try:
            if file.type == "application/pdf":
                text = extract_text_from_pdf(tmp_path)
            else:
                with open(tmp_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            extracted_texts.append(text)
            os.remove(tmp_path)
        except Exception as e:
            st.error(f"Error processing file {file.name}: {e}")
            os.remove(tmp_path)
            continue
    return extracted_texts

# Function to create conversational retrieval chain
def create_conversational_chain(extracted_texts):
    # Convert extracted texts to Document objects
    documents = [Document(page_content=text) for text in extracted_texts]

    splitter = RecursiveCharacterTextSplitter(chunk_size=4500, chunk_overlap=700)
    split_docs = splitter.split_documents(documents)
    vectorstore = FAISS.from_texts([doc.page_content for doc in split_docs], hf_embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return chain

# Main Streamlit app
def main():
    st.title("Document Chat in French with Groq's Llama Models")
    uploaded_files = st.file_uploader(
        "Téléchargez vos documents (PDF ou TXT)",
        type=['pdf', 'txt'],
        accept_multiple_files=True
    )
    if uploaded_files:
        if st.button("Traiter les documents"):
            with st.spinner("Traitement des documents avec OCR et extraction des textes..."):
                extracted_texts = process_files(uploaded_files)
                st.text(extracted_texts)
                if extracted_texts:
                    chain = create_conversational_chain(extracted_texts)
                    st.session_state.chain = chain
                    st.session_state.chat_history = []
                    st.success("Les documents ont été traités avec succès !")
                else:
                    st.error("Aucun texte extrait des documents. Veuillez réessayer.")
    if 'chain' in st.session_state:
        question = st.text_input("Posez une question sur vos documents (en français) :")
        if question:
            with st.spinner("Génération de la réponse..."):
                try:
                    prompt = f"""
                            Vous êtes un agent d'assurance AI, votre role est d'aider les collaborateurs à analyser les données des contrats et des avenants des contrats (différents avenants).
                            Toutes les informations doivent être en ordre chronologique croissant, listant toutes les prestations, dans le contrat ou les avenants.

                Message ou demande de l'utilisateur : {question}

Vous êtes en charge d'extraire des informations précises, comme : 
Capitaux assurés 
Montants des franchises 
Garanties proposées 
Risques assurés et capitaux respectifs 
Sites assurés 
Limite de l'indemnité 
Plafonds pour les contrats maladie 
Taux de prime 
etc ...
                            """
                    result = st.session_state.chain(
                        {"question": prompt, "chat_history": st.session_state.chat_history}
                    )
                    st.session_state.chat_history.append((question, result['answer']))
                    st.write("**Réponse :**", result['answer'])
                    st.write("**Documents sources :**", result['source_documents'])
                except Exception as e:
                    st.error(f"Error generating response: {e}")

if __name__ == "__main__":
    main()
