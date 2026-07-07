# chatbot_langchain.py
from flask import Flask, request, jsonify
from langchain_openai import ChatOpenAI   
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Khởi tạo mô hình OpenAI (nhớ set OPENAI_API_KEY trong môi trường)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# Bộ nhớ hội thoại (giúp bot nhớ ngữ cảnh trước đó)
memory = ConversationBufferMemory()

# Chuỗi hội thoại
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# Flask server
app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    response = conversation.run(user_input)
    return jsonify({"reply": response})

if __name__ == "__main__":
    app.run(port=5000)
