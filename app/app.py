from flask import Flask, request, render_template, jsonify
from joblib import load
from nltk.corpus import stopwords
import string

def preprocess_text(text):
  tokens = text.split()
  translator = str.maketrans('', '', string.punctuation)
  tokens = [s.translate(translator) for s in tokens]
  tokens = [s for s in tokens if s.isalpha()]
  tokens = [s.lower() for s in tokens]
  filter = set(stopwords.words('english'))
  tokens = [s for s in tokens if not s in filter]
  tokens = [s for s in tokens if len(s) > 2]
  return tokens

def assign_bias(token):
  if token == 0:
    return 'Left'
  elif token == 1:
    return 'Center'
  else:
    return 'Right'
  
app = Flask(__name__, static_folder = 'static')

tokenizer = load('tokenizer.joblib')
model = load('bias_predictor.joblib')

@app.route("/", methods = ['GET', 'POST'])
def main():
  if request.method == 'GET':
    return render_template('index.html')
  else:
    if "heading" not in request.form:
      return jsonify({"Error" : "No heading received"}), 400
    
    if "content" not in request.form:
      return jsonify({"Error" : "No content received"}), 400
    
    input_heading = request.form['heading']
    input_content = request.form['content']

    input = str(input_heading) + ': ' + str(input_content)
    tokens = preprocess_text(input)
    vectorized_tokens = tokenizer.infer_vector(tokens)

    output = model.predict([vectorized_tokens])
    pred = assign_bias(output[0])

    return jsonify({'prediction' : str(pred)}), 200