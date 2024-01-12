from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from transformers import DebertaV2Tokenizer, TFDebertaV2ForSequenceClassification
from transformers import RobertaTokenizer, TFRobertaForSequenceClassification

app = Flask(__name__)

# Load Deberta model
deberta_model = TFDebertaV2ForSequenceClassification.from_pretrained("./model_deberta_epoch3")
deberta_tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base", from_pt=True)

# Load Roberta model
roberta_model = TFRobertaForSequenceClassification.from_pretrained("./model_roberta_epoch3")
roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/identify', methods=['POST'])
def identify():
    data = request.json
    model_type = data['model_type']
    text = data['text']

    if model_type == 'deberta':
        tokenizer = deberta_tokenizer
        model = deberta_model
    elif model_type == 'roberta':
        tokenizer = roberta_tokenizer
        model = roberta_model
    else:
        return jsonify({"result": "Invalid model type"})

    # Tokenize input text
    input_ids = tokenizer.encode(text, return_tensors='tf', max_length=512, padding=True, truncation=True)

    # Make prediction
    result = model.predict(input_ids)

    # Get the predicted label
    prediction = int(tf.argmax(result.logits, axis=1))

    return jsonify({"result": prediction})

if __name__ == '__main__':
    app.run(debug=True)
