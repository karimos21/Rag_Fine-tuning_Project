from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer , pipeline
import torch
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
import requests
#from database import embedding_function , load_mongodb_documents, split_documents, add_to_chroma
#from query import query_rag
from torch import cuda
from langchain_community.embeddings import HuggingFaceEmbeddings



CHROMA_PATH = "chroma"

print("loading model ___________________________")
hf_auth = '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'

MODEL_NAME = "gpt2" 

"""
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2" , token=hf_auth , trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2" , token=hf_auth , trust_remote_code=True)
model = pipeline('text-generation', model=model , tokenizer=tokenizer )""" 
print("model loaded successfully_________________")

def embedding_function():
    embed_model_id = 'sentence-transformers/distilbert-base-nli-mean-tokens'
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

    embed_model = HuggingFaceEmbeddings(
        model_name=embed_model_id,
        model_kwargs={'device': device},
        encode_kwargs={'device': device, 'batch_size': 32}
    )
    return embed_model

prompt = PromptTemplate(
    template="""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 

    Here is the retrieved document: 
    
    {document}
    
    ---
    Here is the user question: 
    
    {question} 
    
    assistant""",
    input_variables=["question", "document"],
)



app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    print(request.form['model_type'])
    if request.form['model_type'] == 'RAG':
        msg = request.form["msg"]
        input = msg
        API_URL1 = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
        db = Chroma(
            persist_directory=CHROMA_PATH, embedding_function=embedding_function()
        )
        retriever = db.as_retriever()
        docs = retriever.invoke(input)
        def format_docs(docs):
                return "\n\n---\n\n".join([doc.page_content for doc in docs])
        prompt_text = prompt.format(document=format_docs(docs), question=input)

        return get_Chat_response2(prompt_text , API_URL1)
    else : 
        msg = request.form["msg"]
        input = msg
        API_URL2 = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
        return get_Chat_response(input , API_URL2)

def get_Chat_response(text , API_URL):
    headers = {"Authorization": f"Bearer {hf_auth}"}
    data = {"inputs": text}
    response = requests.post(API_URL, headers=headers, json=data)
    
    if response.status_code == 200:
        """result = response.json()
        result = result[0]["generated_text"]
        if "assistant" in str(result):
            response = result.split("assistant")[-1].strip()"""

        return str(response.json())
    else:
        print(jsonify({"error": "Erreur de l'API"}))
        return str("Erreur de l'API")
    
def get_Chat_response2(text , API_URL):
    headers = {"Authorization": f"Bearer {hf_auth}"}
    data = {"inputs": text}
    response = requests.post(API_URL, headers=headers, json=data)
    
    if response.status_code == 200:
        result = response.json()
        result = result[0]["generated_text"]
        if "assistant" in str(result):
            response = result.split("assistant")[-1].strip()

        return response
    else:
        print(jsonify({"error": "Erreur de l'API"}))
        return str("Erreur de l'API")

if __name__ == '__main__':
    app.run()
