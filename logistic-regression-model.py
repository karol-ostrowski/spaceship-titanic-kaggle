import pandas as pd
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score
from scipy.stats import loguniform

if __name__ == "__main__":

    train_data = pd.read_csv("dataset_02_train.csv")

    X = train_data.drop(columns=["Transported"]).values
    y = train_data["Transported"].values

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=9)

    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    mlflow.set_experiment("spaceship-titanic-kaggle")

    model = LogisticRegression(max_iter=5000)

    param_distributions = {
        'penalty': ['l1', 'l2', 'elasticnet', None],
        'C': loguniform(1e-4, 10),
        'l1_ratio': [0.1, 0.5, 0.9],
        'fit_intercept': [True, False],
        'solver': ['liblinear', 'saga', 'lbfgs', 'newton-cg'],
        'class_weight': [None, 'balanced'],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True)
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=1000,
        scoring="accuracy",
        cv=cv,
        verbose=1,
        n_jobs=-1
    )

    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    y_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {test_acc*100:.2f}%")

    with mlflow.start_run():
        mlflow.set_tag("dataset", "02")
        mlflow.set_tag("model", "lr")
        mlflow.log_params(best_params)
        mlflow.log_metric("accuracy", test_acc)
        mlflow.sklearn.log_model(best_model, "logistic-regression")