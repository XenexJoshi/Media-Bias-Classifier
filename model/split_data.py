import pandas as pd

def split_data():
  print('Opening dataset...')
  data = pd.read_csv('./source_data/bias_data.csv')
  features = data[['heading', 'text', 'bias_rating']]

  print('Retrieving relevant features...')
  features_left = features[features['bias_rating'] == 'left']
  features_left = features_left.sample(n = 4000)

  features_center = features[features['bias_rating'] == 'center']
  features_center = features_center.sample(n = 4000)

  features_right = features[features['bias_rating'] == 'right']
  features_right = features_right.sample(n = 4000)

  print('Storing to .csv...')
  features_left.to_csv('./data/data_left.csv', index = True)
  features_center.to_csv('./data/data_center.csv', index = True)
  features_right.to_csv('./data/data_right.csv', index = True)

if __name__ == '__main__':
  split_data()