import pandas as pd
import mlflow
from sys import argv
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score
from scipy.stats import randint

if __name__ == "__main__":

    dataset_num = argv[1]
    experiment_name = argv[2]
    try:
        run_name = argv[3]
    except IndexError:
        run_name = None

    data = pd.read_csv(f"dataset_{dataset_num}_train.csv")
    data_for_logging = mlflow.data.from_pandas(df=data,
                                               source=f"dataset_{dataset_num}_train.csv",
                                               name=f"dataset_{dataset_num}_train")

    X = data.drop(columns=["Transported"]).values
    y = data["Transported"].values

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=9)

    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    mlflow.set_experiment(experiment_name)

    model = DecisionTreeClassifier()
    param_distributions = {
        'max_depth': randint(3, 40),
        'min_samples_split': randint(2, 40),
        'max_leaf_nodes': randint(10, 300),
        'splitter': ['best', 'random'],
        'max_features': ["sqrt", "log2", 0.5, None]
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    random_search = RandomizedSearchCV(estimator=model,
                                       param_distributions=param_distributions,
                                       n_iter=10,
                                       scoring="accuracy",
                                       cv=cv,
                                       verbose=1,
                                       n_jobs=-1)

    with mlflow.start_run(run_name=run_name):

        random_search.fit(X_train, y_train)

        best_model = random_search.best_estimator_
        best_params = random_search.best_params_
        y_pred = best_model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {test_acc*100:.2f}%")

        mlflow.log_artifact(f"dataset_{dataset_num}_train.csv")
        mlflow.set_tag("model", "dt")
        mlflow.log_params(best_params)
        mlflow.log_metric("accuracy", test_acc)
        mlflow.log_input(data_for_logging, "training")
        mlflow.sklearn.log_model(best_model, "decision-tree", input_example=X_test[0])