import mlflow
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri("sqlite:///C:/Users/HPP/TP_MLflow/mlflow.db")
mlflow.set_experiment("API_Iris")

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

for n in [50, 100, 200]:
    with mlflow.start_run(run_name=f"RF_{n}"):
        model = RandomForestClassifier(n_estimators=n, random_state=42)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        mlflow.log_param("n_estimators", n)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_artifact("train_model.py")
        print("n_estimators =", n, "| accuracy =", acc)
