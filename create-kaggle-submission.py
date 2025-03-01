import mlflow
import argparse
import pandas as pd
import os
import torch

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="script prepares a kaggle submission using given dataset and model")
    parser.add_argument("dataset_number", type=str, help="the version of the testing dataset, file storing the dataset has to adhere to the naming scheme")
    parser.add_argument("model_uri", type=str, help="uri of a choosen model, e.g. mlflow-artifacts:/123456789012345678/12345678901234567890123456789012/artifacts/model")

    args = parser.parse_args()
    
    test_data = pd.read_csv(f"dataset_{args.dataset_number}_test.csv")
    test_ids = pd.read_csv("./kaggle-data/test.csv")

    mlflow.set_tracking_uri("http://localhost:8080")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = mlflow.pyfunc.load_model(args.model_uri)
    if model.metadata.flavors["python_function"]["loader_module"] == "mlflow.pytorch":
        test_data = test_data.astype({
           "CryoSleep"     : "int8",
           "VIP"           : "int8"
        })
        test_data = test_data.astype('float32')
        test_data = torch.tensor(test_data.values, dtype=torch.float32)

    parent_dir = os.path.dirname(__file__)
    number_of_submissions = len([file for file in os.scandir(parent_dir) \
                                if file.name[:18] == "kaggle-submission-"])

    pred_col = pd.Series(model.predict(test_data).flatten()).astype(bool)
    predicted_df = pd.concat([test_ids["PassengerId"], pred_col], axis=1)
    predicted_df = predicted_df.rename(columns={0 : "Transported"})
    predicted_df.to_csv(f"kaggle-submission-{number_of_submissions + 1}.csv", header=True, index=False)
    print(f"results saved to kaggle-submission-{number_of_submissions + 1}.csv")