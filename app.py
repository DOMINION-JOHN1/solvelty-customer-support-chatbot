from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnablePassthrough

# Initialize Flask app
app = Flask(__name__)
CORS(app) 
# Set other environment variables
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
API_KEY = os.getenv("SOLVELTY")

# Initialize model and embeddings
MODEL = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=API_KEY)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=API_KEY)
parser = StrOutputParser()

# Define the prompt template
template ="""
You are a customer support personnel for Solvelty.
Always sound polite and welcome customers warmly when they ask their first questions.
Help customers navigate the platform with ease. Provide very accurate and very concise and precise answers to their queries about the company.
Derive all your answers based on the information in the context provided.
Write out the answer to the question directly .
Even if you don't know the answer immediately, go through the context again and provide the most accurate answer and suggestion.

Context: {context}

Question: {question}
"""
prompt = PromptTemplate.from_template(template)
prompt.format(context="Here is some context", question="Here is a question")

# Create Pinecone Vector Store
index_name = "solvetydb"
pinecone = PineconeVectorStore(index_name=index_name, embedding=embeddings)

# Define the chain
chain = (
    {"context": pinecone.as_retriever(), "question": RunnablePassthrough()}
    | prompt
    | MODEL
    | parser
)

# Function to get response from the bot
def get_response(user_input):
    question = user_input
    response = chain.invoke(question)
    return response

#@app.route("/", methods=["GET"])
#def home():
#    return "Welcome to the Solvelty Customer Support Chatbot API!", 200

@app.route("/", methods=["POST"])
def chatbot():
    data = request.json
    if "question" not in data:
        return jsonify({"error": "No question provided"}), 400
    
    user_input = data["question"]
    response = get_response(user_input)
    
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
