from langchain_community.vectorstores.chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema.document import Document
from torch import cuda
import os
import pymongo
from datasets import Dataset

# MongoDB connection settings
MONGO_URI = "mongodb://localhost:27017/"  
MONGO_DB = "Metadata_FSTT" 
COLLECTION_NAMES = ["equipes_recherche", "FORMATION_CONTINUE_informations",
                     "recherche_struct",
                     "FORMATION-CONTINUE",
                     "FORMATION-INITIALE",
                     "FORMATION-INITIALE-informations",
                     "espace-entreprise",
                     "espace-etudiant-biblio",
                     "espace-etudiant-clubs",
                     "faculte_contact",
                     "faculte_departements",
                     "faculte_motdoyen",
                     "faculte_presentation",
                     "fstt_service"]  

CHROMA_PATH = "chroma"  

def embedding_function():
    embed_model_id = 'sentence-transformers/distilbert-base-nli-mean-tokens'
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

    embed_model = HuggingFaceEmbeddings(
        model_name=embed_model_id,
        model_kwargs={'device': device},
        encode_kwargs={'device': device, 'batch_size': 32}
    )
    return embed_model

def load_mongodb_documents(collection_name):
    client = pymongo.MongoClient(MONGO_URI)
    db = client[MONGO_DB]
    collection = db[collection_name]
    documents = []
    for doc in collection.find():
        text_data = {k: v for k, v in doc.items() if k not in ["metadata", "id", "url"]}
        text = ""
        for key, value in text_data.items():
            if isinstance(value, (list, dict)):
                text += f"{key}: {value}\n"
            else:
                text += f"{key}: {value} " 

        documents.append(Document(page_content=text, metadata={"source": collection_name}))
    return documents

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=10,
        length_function=len,
        is_separator_regex=False,
    )
    return documents

def add_to_chroma(chunks: list[Document]):
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=embedding_function()
    )
    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[])  
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{source}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id

    return chunks

embeddings = embedding_function()

documents = []
for collection_name in COLLECTION_NAMES:
    mongodb_documents = load_mongodb_documents(collection_name)
    documents.extend(load_mongodb_documents(collection_name))
    split_docs = split_documents(mongodb_documents)
    vectorstore = add_to_chroma(split_docs)
    print(f"Stored documents from '{collection_name}' in ChromaDB at {CHROMA_PATH}")

print(f"Stored documents from all collections in ChromaDB at {CHROMA_PATH}")