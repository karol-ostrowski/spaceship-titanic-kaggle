import pandas as pd
import mlflow
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score
from scipy.stats import loguniform

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="script creates logistic regression models based on radomized hardcoded parameters, logs the best one found to mlflow server")
    parser.add_argument("dataset_number", type=str, help="the version of the training dataset, file storing the dataset has to adhere to the naming scheme")
    parser.add_argument("experiment_name", type=str, help="mlflow experiment name")
    parser.add_argument("num_iter", type=int, help="number of iterations for the random search")
    parser.add_argument("--run_name", type=str, help="optional argument for setting a custom mlflow run name", required=False)

    args = parser.parse_args()

    dataset_num = args.dataset_number

    data = pd.read_csv(f"dataset_{dataset_num}_train.csv")
    data_for_logging = mlflow.data.from_pandas(df=data,
                                               name=f"dataset_{dataset_num}_train")

    X = data.drop(columns=["Transported"]).values
    y = data["Transported"].values

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=9)

    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    mlflow.set_experiment(args.experiment_name)

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
        n_iter=args.num_iter,
        scoring="accuracy",
        cv=cv,
        verbose=1,
        n_jobs=-1
    )

    with mlflow.start_run(run_name=args.run_name):

        random_search.fit(X_train, y_train)

        best_model = random_search.best_estimator_
        best_params = random_search.best_params_
        y_pred = best_model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {test_acc*100:.2f}%")

        mlflow.log_artifact(f"dataset_{dataset_num}_train.csv")
        mlflow.set_tag("model", "lr")
        mlflow.log_params(best_params)
        mlflow.log_metric("accuracy", test_acc)
        mlflow.log_input(data_for_logging, "training")
        mlflow.sklearn.log_model(best_model, "logistic-regression", input_example=X_test[0].reshape(1, -1))