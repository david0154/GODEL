from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Initialize Flask app
app = Flask(__name__)

# Path to your GODEL model directory (make sure this path is correct)
MODEL_DIR = './GODEL'

# Load the pre-trained GODEL model and tokenizer
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.json['text']  # Get user input from frontend

    # Tokenize and generate a response using the model
    inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    response_ids = model.generate(inputs, max_length=100, pad_token_id=tokenizer.eos_token_id)
    response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)

    # Return the response to the frontend
    return jsonify({"response": response_text})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 for local testing
    app.run(debug=True, host='0.0.0.0', port=port)
