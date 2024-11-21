from flask import Flask, request, render_template
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('house_price_model.pkl')

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')  # HTML form for user input

# Define prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    data = request.form

    # Create a DataFrame for the input data
    sample_data = pd.DataFrame({
        'Province': [data['Province']],
        'City': [data['City']],
        'Type': [data['Type']],
        'SqFt': [float(data['SqFt'])],
        'Bedrooms': [int(data['Bedrooms'])],
        'Bathrooms': [int(data['Bathrooms'])],
        'Garage': [int(data['Garage'])],
        'Year_Built': [int(data['Year_Built'])],
        'Lot_Area': [float(data['Lot_Area'])]
    })

    # Predict using the model
    prediction = model.predict(sample_data)

    # Render the result in a user-friendly format
    return render_template('result.html', price=round(prediction[0], 2))

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
