import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from imblearn.over_sampling import SMOTE  # Import SMOTE for balancing
import joblib
import os

# Step 1: Read your CSV file
df = pd.read_csv('data/stud.csv')  # <-- stud.csv yahan diya

# Step 2: Features aur Target alag karo
X = df.drop('Courses', axis=1)  # 'Courses' is your target
y = df['Courses']

# Step 3: SMOTE apply karo for balancing the data
smote = SMOTE(random_state=42)  # SMOTE ka object banaya
X_smote, y_smote = smote.fit_resample(X, y)  # Balancing data with SMOTE

# Step 4: Random Forest model train karo on balanced data
rf_model = RandomForestClassifier()
rf_model.fit(X_smote, y_smote)

# Step 5: AdaBoost model train karo on balanced data
ada_model = AdaBoostClassifier()
ada_model.fit(X_smote, y_smote)

# Step 6: Check if 'models' folder exists, agar nahi to banao
os.makedirs('models', exist_ok=True)

# Step 7: Models ko save karo
joblib.dump(rf_model, 'models/random_forest_model.pkl')
joblib.dump(ada_model, 'models/ada_boost_model.pkl')

print(" 'random_forest_model.pkl' aur 'ada_boost_model.pkl' successfully save ho gaye hain.")
