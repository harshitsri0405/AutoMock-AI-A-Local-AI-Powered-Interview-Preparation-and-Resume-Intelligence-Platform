import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib

print("Starting the model training pipeline...")

# 1. Dono dataset files ko load karein
try:
    df1 = pd.read_csv('dataset_part1.csv')
    df2 = pd.read_csv('dataset_part2.csv')
    print(f"Loaded Part 1 ({len(df1)} rows) and Part 2 ({len(df2)} rows) successfully.")
except FileNotFoundError as e:
    print(f"Error: File nahi mili! Kripya check karein ki dono CSV files isi folder me hain. Detail: {e}")
    exit()

# 2. Dono datasets ko ek sath combine (jodna) karein
df = pd.concat([df1, df2], ignore_index=True)
print(f"Combined dataset total rows: {len(df)}")

# 3. Features (X) aur Target (y) select karein
# Hum 'question' text ke basis par 'difficulty' (Easy, Medium, Hard) predict kar rahe hain
X = df['question'].fillna('')  # Kahin khali row ho toh handle karne ke liye fillna('')
y = df['difficulty']

# 4. Data ko Train aur Test set me split karein (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Dataset successfully split into Train and Test sets.")

# 5. Pipeline banayein (Vectorizer + Machine Learning Model)
# Isse text data automatic handle ho jayega aur alag se transformation nahi karni padegi
text_clf = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
    ('clf', RandomForestClassifier(random_state=42, n_estimators=100))
])

# 6. Model ko train karein sirf Training Data par
print("Training the model (this might take a few seconds)...")
text_clf.fit(X_train, y_train)
print("Model training complete!")

# 7. Model ko evaluate (check) karein Test Data par
y_pred = text_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\n" + "="*40)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print("="*40)
print("Classification Report:\n", classification_report(y_test, y_pred))

# 8. Final trained Pipeline/Model ko save karein
model_filename = 'final_combined_model.pkl'
joblib.dump(text_clf, model_filename)
print(f"\nSuccess: Saved the trained model pipeline as '{model_filename}'")