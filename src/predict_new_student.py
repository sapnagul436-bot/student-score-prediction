
import os
import joblib
import pandas as pd

# Project paths
project_root = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

model_path = os.path.join(
    project_root,
    "models",
    "student_score_prediction_pipeline.joblib"
)

# Load the saved model
model = joblib.load(model_path)

# New student information
new_student = {
    "school": "GP",
    "sex": "F",
    "age": 17,
    "address": "U",
    "famsize": "GT3",
    "Pstatus": "T",
    "Medu": 4,
    "Fedu": 3,
    "Mjob": "teacher",
    "Fjob": "services",
    "reason": "course",
    "guardian": "mother",
    "traveltime": 1,
    "studytime": 3,
    "failures": 0,
    "schoolsup": "no",
    "famsup": "yes",
    "paid": "no",
    "activities": "yes",
    "nursery": "yes",
    "higher": "yes",
    "internet": "yes",
    "romantic": "no",
    "famrel": 4,
    "freetime": 3,
    "goout": 2,
    "Dalc": 1,
    "Walc": 2,
    "health": 5,
    "absences": 2,
    "G1": 14,
    "G2": 15
}

# Convert input into a DataFrame
student_df = pd.DataFrame([new_student])

# Generate prediction
prediction = model.predict(student_df)

# Display results
print("NEW STUDENT SCORE PREDICTION")
print("----------------------------")
print("Student Information:")
print(student_df.to_string(index=False))

print("\nPREDICTION RESULT")
print("-----------------")
print("Predicted Final Grade (G3):", round(prediction[0], 2))

print("\nPrediction generated successfully!")
