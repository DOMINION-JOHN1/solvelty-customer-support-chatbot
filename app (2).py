from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnablePassthrough

# Initialize Flask app
app = Flask(__name__)

# Set other environment variables
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize model and embeddings
model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=OPENAI_API_KEY)
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
index_name = "sovelty"
pinecone = PineconeVectorStore(index_name=index_name, embedding=embeddings)

# Define the chain
chain = (
    {"context": pinecone.as_retriever(), "question": RunnablePassthrough()}
    | prompt
    | model
    | parser
)

# Function to get response from the bot
def get_response(user_input):
    question = user_input
    response = chain.invoke(question)
    return response

@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.json
    if "question" not in data:
        return jsonify({"error": "No question provided"}), 400
    
    user_input = data["question"]
    response = get_response(user_input)
    
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)