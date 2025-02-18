import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score
from scipy.stats import randint

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

    model = RandomForestClassifier()
    param_distributions = {
    'n_estimators': randint(50, 400),
    'max_depth': randint(5, 40),
    'min_samples_split': randint(2, 40),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', 0.5, None],
    'criterion': ['gini', 'entropy'],
    'bootstrap': [True, False],
    'class_weight': [None, 'balanced', 'balanced_subsample'],
    'max_leaf_nodes': randint(10, 200),
    'min_impurity_decrease': [0.0, 0.02, 0.04, 0.06, 0.08, 0.1]
}
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    random_search = RandomizedSearchCV(estimator=model,
                                       param_distributions=param_distributions,
                                       n_iter=600,
                                       scoring="accuracy",
                                       cv=cv,
                                       verbose=1,
                                       n_jobs=-1)

    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    y_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {test_acc*100:.2f}%")

    with mlflow.start_run():
        mlflow.set_tag("dataset", "02")
        mlflow.set_tag("model", "rf")
        mlflow.log_params(best_params)
        mlflow.log_metric("accuracy", test_acc)
        mlflow.sklearn.log_model(best_model, "random-forest")