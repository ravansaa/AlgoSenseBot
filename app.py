from flask import Flask, render_template, request, jsonify
from chatbot import chatbot_recommendation, classify_question

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    # Terima input dari frontend dan berikan respons
    user_input = request.json.get('message')
    if not user_input:
        return jsonify({'error': 'No input provided'}), 400
    
    # Jawaban dari fungsi rekomendasi (tanpa logika percakapan di sini)
    response = chatbot_recommendation(user_input)
    return jsonify({
        'pertanyaan': response['Pertanyaan'],
        'jawaban': response['Jawaban'],
        'similarity_score': response['Similarity Score']
    })

@app.route('/classify', methods=['POST'])
def classify():
    data = request.json
    user_input = data.get("response", "")

    if not user_input:
        return jsonify({"error": "Input kosong. Mohon ketik sesuatu."}), 400

    classification, reply = classify_question(user_input)
    return jsonify({"classification": classification, "reply": reply})

if __name__ == '__main__':
    app.run(debug=True)
