import pandas as pd
from xgboost import XGBClassifier

if __name__ == "__main__":

    train_data = pd.read_csv("dataset_03_train.csv")
    test_data = pd.read_csv("dataset_03_test.csv")
    test_ids = pd.read_csv("./kaggle-data/test.csv")

    X = train_data.drop(columns=["Transported"]).values
    y = train_data["Transported"].values

    model = XGBClassifier(use_label_encoder=False,
                          eval_metric="error",
                          n_estimators=150,
                          max_depth=35,
                          learning_rate=0.11341643067742249,
                          subsample=0.7675498059971977,
                          colsample_bytree=0.772832835188157,
                          gamma=1.851601104221558,
                          reg_alpha=1.8315009702739427,
                          reg_lambda=14.113396855479602)

    model.fit(X, y)
    pred_col = pd.Series(model.predict(test_data)).astype(bool)
    predicted_df = pd.concat([test_ids["PassengerId"], pred_col], axis=1)
    predicted_df = predicted_df.rename(columns={0 : "Transported"})
    predicted_df.to_csv("kaggle-submission-03.csv", header=True, index=False)