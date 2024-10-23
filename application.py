# Used GPT to make this into a basic flask app
from flask import Flask, request
import pickle
from sklearn.feature_extraction.text import CountVectorizer

# Initialize the Flask app
application = Flask(__name__)

###### Model loading #####
loaded_model = None
vectorizer = None

# Load the model and vectorizer
with open("basic_classifier.pkl", "rb") as fid:
    loaded_model = pickle.load(fid)

with open("count_vectorizer.pkl", "rb") as vd:
    vectorizer = pickle.load(vd)


# Home route to display form
@application.route("/")
def home():
    return """
        <h2>Fake News Detection</h2>
        <form action="/predict" method="post">
            <label for="text">Enter text:</label><br><br>
            <textarea name="text" id="text" rows="4" cols="50"></textarea><br><br>
            <input type="submit" value="Predict">
        </form>
    """


# Prediction route to handle form submission and show result
@application.route("/predict", methods=["POST"])
def predict():
    # Get the input text from the form
    input_text = request.form["text"]

    # Use the model to make a prediction
    prediction = loaded_model.predict(vectorizer.transform([input_text]))[0]

    # Return the result
    return f'<h3>Prediction: {prediction}</h3><br><a href="/">Go Back</a>'


if __name__ == "__main__":
    application.run()
