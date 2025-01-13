import pandas as pd
import string
from nltk.corpus import stopwords

def load_and_process(file_path):
  df = pd.read_csv(file_path, header = 0)
  df['text'] = df['heading'] + ': ' + df['text']
  df = df[['text', 'bias_rating']]
  return df
  
def preprocess_text(text):
  tokens = text.split()
  translator = str.maketrans('', '', string.punctuation)
  tokens = [s.translate(translator) for s in tokens]
  tokens = [s for s in tokens if s.isalpha()]
  filter = set(stopwords.words('english'))
  tokens = [s for s in tokens if not s in filter]
  tokens = [s for s in tokens if len(s) > 2]
  return tokens

def vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    classes, features = zip(*[(doc.tags[0],
      model.infer_vector(doc.words)) for doc in sents])
    return features, classes