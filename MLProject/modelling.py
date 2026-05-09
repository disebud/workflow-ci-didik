import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

# 1. HAPUS/COMMENT Tracking URI & Experiment khusus untuk GitHub Actions
# mlflow.set_tracking_uri("http://127.0.0.1:5000")
# mlflow.set_experiment("Eksperimen_AI_Impact_Basic")

# 2. Memuat Data
df = pd.read_csv("AI_Impact_preprocessed.csv")

# 3. Memisahkan Fitur dan Target
# Kita asumsikan targetnya adalah 'Career_Confidence_Score'
X = df.drop('Career_Confidence_Score', axis=1)
y = df['Career_Confidence_Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Melatih Model dengan Autolog
mlflow.sklearn.autolog() # Mencatat metrik & parameter secara otomatis

with mlflow.start_run(run_name="RandomForest_Basic"):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    print("Model berhasil dilatih dan dicatat di MLflow!")