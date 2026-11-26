

\# Student Score Prediction



A Machine Learning project that predicts students' final academic grades using demographic, social, and academic information.



\## 1. Project Overview



The Student Score Prediction project uses supervised machine learning to estimate a student's final mathematics grade (G3) based on previously recorded student information.



The project applies data preprocessing, exploratory data analysis, data visualization, model training, evaluation, and prediction using Python and Scikit-learn.



\## 2. Problem Statement



Student academic performance can be influenced by several factors, including study time, previous grades, family background, school support, and absences.



The objective of this project is to develop a machine learning model that predicts students' final grades using these features.



\## 3. Dataset Description



\*\*Dataset:\*\* Student Performance — Mathematics (`student-mat.csv`)



\*\*Dataset information:\*\*

\- Total records: 395

\- Total features: 33

\- Target variable: G3

\- Previous grade features: G1 and G2



The dataset contains demographic, family, educational, and academic information about students.



\## 4. Technologies and Libraries



\- Python 3.11.9

\- Pandas

\- NumPy

\- Matplotlib

\- Seaborn

\- Scikit-learn

\- Joblib



\## 5. Project Workflow



The project follows these machine learning stages:



1\. Dataset collection and loading.

2\. Data inspection and understanding.

3\. Data cleaning and preprocessing.

4\. Exploratory Data Analysis (EDA).

5\. Data visualization.

6\. Feature and target separation.

7\. Training and testing data splitting.

8\. Model training and comparison.

9\. Residual analysis.

10\. Final pipeline training and evaluation.

11\. Model saving and loading.

12\. Prediction using new student data.



\## 6. Data Preprocessing



The following preprocessing techniques are implemented:



\*\*Numerical Features\*\*

\- Missing values are handled using median imputation.



\*\*Categorical Features\*\*

\- Missing values are handled using most-frequent imputation.

\- One-Hot Encoding converts categorical variables into numerical representations.

\- Unknown categories are handled using `handle\_unknown="ignore"`.



A Scikit-learn ColumnTransformer combines the numerical and categorical preprocessing operations.



\## 7. Machine Learning Model



\*\*Algorithm:\*\* Linear Regression



Linear Regression is used to predict the continuous final grade (G3) based on the available student features.



The complete machine learning pipeline includes:



\- Numerical preprocessing.

\- Categorical preprocessing.

\- One-Hot Encoding.

\- Linear Regression.



The pipeline is trained and evaluated using an 80:20 train-test split with random\_state=42.



After evaluation, the final pipeline is retrained using the complete dataset and saved using Joblib.



\## 8. Model Evaluation



The final model was evaluated using three performance metrics.



| Metric | Result |

|---|---:|

| Mean Absolute Error (MAE) | 1.646666 |

| Root Mean Squared Error (RMSE) | 2.378370 |

| R² Score | 0.724134 |



\*\*Interpretation:\*\*



\- MAE measures the average absolute prediction error.

\- RMSE gives greater weight to larger prediction errors.

\- R² indicates that the model explains approximately 72.41% of the variation in the holdout test scores.



\## 9. Prediction Example



The saved model was tested using new student information.



\*\*Input:\*\*

\- Age: 17

\- Study Time: 3

\- Previous Grade (G1): 14

\- Previous Grade (G2): 15

\- Absences: 2



\*\*Predicted Final Grade:\*\*



14.52 / 20



The prediction was generated successfully using the saved preprocessing and regression pipeline.



\## 10. Project Structure



```text

Task\_01\_Student\_Score\_Prediction/

│

├── data/

│   └── raw/

│       └── student-mat.csv

│

├── notebooks/

│

├── src/

│   ├── save\_final\_model.py

│   └── predict\_new\_student.py

│

├── models/

│   └── student\_score\_prediction\_pipeline.joblib

│

├── outputs/

│   ├── figures/

│   └── metrics/

│       ├── model\_comparison.csv

│       └── prediction\_errors.csv

│

└── README.md

```



\## 11. Installation and Setup



Clone the repository:



```bash

git clone YOUR\_GITHUB\_REPOSITORY\_URL

```



Navigate to the project directory:



```bash

cd Task\_01\_Student\_Score\_Prediction

```



Install the required libraries:



```bash

pip install pandas numpy matplotlib seaborn scikit-learn joblib

```



Ensure that the dataset is available at:



```text

data/raw/student-mat.csv

```



\## 12. Running the Project



To train and save the final machine learning pipeline:



```bash

python src/save\_final\_model.py

```



To generate predictions for a new student:



```bash

python src/predict\_new\_student.py

```



Both scripts use the saved preprocessing and regression pipeline.



\## 13. Limitations



\- The model is trained on a relatively small dataset.

\- Predictions depend on the quality and representativeness of the available data.

\- Linear Regression may not capture complex relationships between student characteristics and academic performance.

\- Predicted grades are estimates, not guaranteed examination results.



\## 14. Conclusion



This project demonstrates the practical implementation of a complete supervised machine learning workflow.



It covers data analysis, preprocessing, model development, evaluation, model persistence, and prediction using new student information.



The project provides practical experience with Python, Pandas, Scikit-learn, and machine learning pipeline development.



\---



\*\*Project:\*\* Student Score Prediction



\*\*Level:\*\* Level 1 — Machine Learning Internship



\*\*Algorithm:\*\* Linear Regression



\*\*Status:\*\* Completed



