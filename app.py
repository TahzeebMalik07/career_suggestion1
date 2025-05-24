from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Production mein ise secure rakhna

# Load trained models
rf_model = joblib.load('models/random_forest_model.pkl')
ada_model = joblib.load('models/ada_boost_model.pkl')

# Dummy user database (in-memory)
users = {}

# List of all 59 interests (same order as training data)
all_interests = [
    "Drawing", "Dancing", "Singing", "Sports", "Video Game", "Acting", "Travelling", "Gardening",
    "Animals", "Photography", "Teaching", "Exercise", "Coding", "Electricity Components", 
    "Mechanic Parts", "Computer Parts", "Researching", "Architecture", "Historic Collection", 
    "Botany", "Zoology", "Physics", "Accounting", "Economics", "Sociology", "Geography", "Psychology",
    "History", "Science", "Business Education", "Chemistry", "Mathematics", "Biology", "Makeup",
    "Designing", "Content Writing", "Crafting", "Literature", "Reading", "Cartooning", "Debating",
    "Astrology", "Hindi", "French", "English", "Urdu", "Other Language", "Solving Puzzles", 
    "Gymnastics", "Yoga", "Engineering", "Doctor", "Pharmacist", "Cycling", "Knitting", "Director", 
    "Journalism", "Business", "Listening to Music"
]

# Mapping from model class index to course name
course_labels = {
    0: 'Animation, Graphics and Multimedia', 1: 'B.Arch- Bachelor of Architecture',
    2: 'B.Com- Bachelor of Commerce', 3: 'B.Ed.', 4: 'B.Sc- Applied Geology',
    5: 'B.Sc- Nursing', 6: 'B.Sc. Chemistry', 7: 'B.Sc. Mathematics',
    8: 'B.Sc.- Information Technology', 9: 'B.Sc.- Physics', 10: 'B.Tech.-Civil Engineering',
    11: 'B.Tech.-Computer Science and Engineering', 12: 'B.Tech.-Electrical and Electronics Engineering',
    13: 'B.Tech.-Electronics and Communication Engineering', 14: 'B.Tech.-Mechanical Engineering',
    15: 'BA in Economics', 16: 'BA in English', 17: 'BA in Hindi', 18: 'BA in History',
    19: 'BBA- Bachelor of Business Administration', 20: 'BBS- Bachelor of Business Studies',
    21: 'BCA- Bachelor of Computer Applications', 22: 'BDS- Bachelor of Dental Surgery',
    23: 'BEM- Bachelor of Event Management', 24: 'BFD- Bachelor of Fashion Designing',
    25: 'BJMC- Bachelor of Journalism and Mass Communication', 26: 'BPharma- Bachelor of Pharmacy',
    27: 'BTTM- Bachelor of Travel and Tourism Management', 28: 'BVA- Bachelor of Visual Arts',
    29: 'CA- Chartered Accountancy', 30: 'CS- Company Secretary', 31: 'Civil Services',
    32: 'Diploma in Dramatic Arts', 33: 'Integrated Law Course- BA + LL.B', 34: 'MBBS'
}

# Capitalize all interests to match model training case
all_interests_proper_case = [interest.title() for interest in all_interests]

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/signup', methods=['POST'])
def signup():
    username = request.form['username']
    password = request.form['password']
    if username in users:
        return "User already exists. Try logging in."
    users[username] = password
    return redirect(url_for('home'))

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    if username in users and users[username] == password:
        session['username'] = username
        return redirect(url_for('welcome'))
    return "Invalid credentials. Try again."

@app.route('/welcome')
def welcome():
    if 'username' not in session:
        return redirect(url_for('home'))
    return render_template("welcome.html", username=session['username'])

@app.route('/interests', methods=['GET', 'POST'])
def interests():
    if 'username' not in session:
        return redirect(url_for('home'))
    if request.method == 'POST':
        selected = request.form.getlist('interests')
        if not selected:
            return render_template("interests.html", feature_names=all_interests, error_message="Please select at least one interest.")
        session['selected_interests'] = selected
        return redirect(url_for('predict_career'))
    return render_template("interests.html", feature_names=all_interests)

@app.route('/predict_career', methods=['GET'])
def predict_career():
    if 'selected_interests' not in session:
        return redirect(url_for('interests'))

    selected = session['selected_interests']
    
    # One-hot encode input using the correct capitalization
    selected_proper_case = [interest.title() for interest in selected]  # Proper title case for selected interests
    X_input = [1 if interest.title() in selected_proper_case else 0 for interest in all_interests_proper_case]  # Ensure matching titles
    
    # Ensure columns in DataFrame match feature names in training data
    X_input = pd.DataFrame([X_input], columns=all_interests_proper_case)  # Using title-case names for columns

    # Predict probabilities using the trained models
    rf_probs = rf_model.predict_proba(X_input)[0]
    ada_probs = ada_model.predict_proba(X_input)[0]

    # Top 3 indices from each
    rf_top_3 = sorted(enumerate(rf_probs), key=lambda x: x[1], reverse=True)[:3]
    ada_top_3 = sorted(enumerate(ada_probs), key=lambda x: x[1], reverse=True)[:3]

    rf_top_3_labels = [course_labels[i] for i, _ in rf_top_3]
    ada_top_3_labels = [course_labels[i] for i, _ in ada_top_3]

    common_predictions = list(set(rf_top_3_labels) & set(ada_top_3_labels))

    return render_template("results.html", common_predictions=common_predictions,
                           rf_predictions=rf_top_3_labels, ada_predictions=ada_top_3_labels)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)

