import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("dataset.csv")


symptom_cols = [col for col in df.columns if 'symptom' in col.lower()]

# Flatten all symptom columns, remove NaNs, strip spaces, and lowercase them
all_symptoms = pd.unique(df[symptom_cols].values.ravel('K'))
valid_symptoms = {str(s).strip().lower().replace(" ", "_") for s in all_symptoms if pd.notna(s)}

labels = df['Disease']
df['text'] = df[symptom_cols].fillna('').agg(' '.join, axis=1)

# Train/Test Split & Vectorization 
x_train, x_test, y_train, y_test = train_test_split(
    df['text'], labels, test_size=0.2, random_state=7
)

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)

# Model Training
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

# Prediction Analysis 
y_pred = pac.predict(tfidf_test)
print(f'Model Accuracy: {round(accuracy_score(y_test, y_pred)*100,2)}%')

# Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(15, 12))
sns.heatmap(cm, annot=False, cmap='Blues', fmt='g')
plt.title('Confusion Matrix: Predicted vs. Actual Disease')
plt.xlabel('Predicted Disease')
plt.ylabel('Actual Disease')
plt.tight_layout()
plt.show()

def predict_disease(symptoms_list):
    combined_text = " ".join(symptoms_list)
    vectorized = tfidf_vectorizer.transform([combined_text])
    prediction = pac.predict(vectorized)
    return prediction[0]

# Interactive Loop with Validation
if __name__ == '__main__':
    print("\n--- Disease Prediction System ---")
    print("Enter 5 symptoms. Use underscores for spaces (e.g., 'high_fever').")
    print("Type 'quit' to exit at any time.\n")

    while True:
        user_symptoms = []
        
        while len(user_symptoms) < 5:
            current_num = len(user_symptoms) + 1
            user_input = input(f"Symptom {current_num}: ").strip().lower().replace(" ", "_")

            if user_input in ('quit', 'exit'):
                print("Exiting...")
                exit()

            if not user_input:
                continue

            # Check if the symptom exists in our dataset
            if user_input in valid_symptoms:
                user_symptoms.append(user_input)
            else:
                print(f"  [!] '{user_input}' is not recognized. Please try a different term.")
                

        # Make Prediction
        result = predict_disease(user_symptoms)
        print(f"\n Predicted Disease: {result}\n")
  