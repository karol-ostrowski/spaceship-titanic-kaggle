import pandas as pd
from xgboost import XGBClassifier

if __name__ == "__main__":

    train_data = pd.read_csv("dataset_02_train.csv")
    test_data = pd.read_csv("dataset_02_test.csv")
    test_ids = pd.read_csv("./kaggle-data/test.csv")

    X = train_data.drop(columns=["Transported"]).values
    y = train_data["Transported"].values

    model = XGBClassifier(use_label_encoder=False,
                          eval_metric="error",
                          n_estimators=265,
                          max_depth=32,
                          learning_rate=0.3809396265380405,
                          subsample=0.7229620629990027,
                          colsample_bytree=0.6111213204870721,
                          gamma=1.046949384674779,
                          reg_alpha=1.9639274983046873,
                          reg_lambda=11.680455723681098)

    model.fit(X, y)
    pred_col = pd.Series(model.predict(test_data)).astype(bool)
    predicted_df = pd.concat([test_ids["PassengerId"], pred_col], axis=1)
    predicted_df = predicted_df.rename(columns={0 : "Transported"})
    predicted_df.to_csv("kaggle-submission-02.csv", header=True, index=False)