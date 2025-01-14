Media Bias Classifier:

<img width="1260" alt="Screenshot 2025-01-14 at 11 32 29 AM" src="https://github.com/user-attachments/assets/b5c71cee-bc76-42c0-884e-35bd8d1d556b" />

This project contains a Flask web app with a Python backend, a trained machine-learning model integrated into the backend, and an HTML/CSS frontend that allows users to input the news article heading and news content to classify the media bias by analyzing the input news article. The backend uses a trained SVM to predict the media bias, by first preprocessing the input text into an encoded vector using a trained tokenizer, which is passed onto the SVM model to generate a prediction based on its learned data. The SVM model was trained, and its hyperparameters were fine-tuned using GridSearchCV(), yielding a 93.1% accuracy on the testing set. The backend is connected to the frontend using rest API, allowing the news article to be passed on from the frontend to the backend, and the prediction, along with the prediction confidence, to be passed on from the backend to the frontend to be accessible by the users.

Required modules:

    flask
    pandas
    numpy
    scikit-learn
    nltk
    gensim
    joblib

To run the program, clone this repository, install the required modules and run the following commands:

    cd app
    flask run

To run the jupyter notebook containing the model, run the following commands before running the jupyter notebook:

    cd model
    python splot_data.py
