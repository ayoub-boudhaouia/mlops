import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Charger et préparer les données
df = pd.read_csv("churn_data.csv")
X = pd.get_dummies(df[["age", "salaire", "abonnement", "anciennete", "consommation"]], drop_first=True)
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Créer l'expérience MLflow
mlflow.set_experiment("churn-prediction")

# 3. Lancer un run
with mlflow.start_run(run_name="essai-1"):
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    
    # Enregistrer les paramètres, la métrique et le modèle
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")
    
    print("Accuracy :", acc)