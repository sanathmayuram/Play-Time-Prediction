from flask import Flask, request, render_template
import pickle

# Initialize Flask
app = Flask(__name__)

# Load trained model
NaiveBays_model = pickle.load(open('GaussianNB.pkl', 'rb'))

# Load LabelEncoders (one per column)
le_outlook = pickle.load(open('le_outlook.pkl', 'rb'))
le_temp = pickle.load(open('le_temp.pkl', 'rb'))
le_humidity = pickle.load(open('le_humidity.pkl', 'rb'))
le_windy = pickle.load(open('le_windy.pkl', 'rb'))

print("Outlook classes:", le_outlook.classes_)
print("Temperature classes:", le_temp.classes_)
print("Humidity classes:", le_humidity.classes_)
print("Windy classes:", le_windy.classes_)

# Home page
@app.route('/', methods=["GET"])
def home():
    return render_template("home.html", results="")

# Prediction route
@app.route('/predict_datapoint', methods=["POST"])
def predict_datapoint():
    # 1️⃣ Get user input from form
    Outlook = request.form.get('Outlook')
    Temperature = request.form.get('Temperature')
    Humidity = request.form.get('Humidity')
    Windy = request.form.get('Windy')

    # 2️⃣ Encode each input using its respective LabelEncoder
    try:
        Outlook_enc = le_outlook.transform([Outlook])[0]
        Temperature_enc = le_temp.transform([Temperature])[0]
        Humidity_enc = le_humidity.transform([Humidity])[0]
        Windy_enc = le_windy.transform([Windy])[0]
    except ValueError:
        return render_template("home.html", results="Error: Invalid input. Use correct categories!")

    # 3️⃣ Combine into 2D array for prediction
    new_data = [[Outlook_enc, Temperature_enc, Humidity_enc, Windy_enc]]

    # 4️⃣ Predict using the model
    result = NaiveBays_model.predict(new_data)[0]

    return render_template("home.html", results="Yes You Can Play" if result == 0 else "No You Can't Play")

# Run the app
if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
