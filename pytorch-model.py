import torch
from torch import nn
import pandas as pd
import mlflow
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class binary_classifier(nn.Module):
    def __init__(self, input_size, layer_sizes):
        super().__init__()
        layers = []

        prev_size = input_size
        for size in layer_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size
            
        layers.append(nn.Linear(prev_size, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def train_test(model,
               epochs,
               loss_fn,
               optimizer,
               X_train,
               y_train,
               X_test,
               y_test):
    
    model.to(device)
    X_train, X_test = X_train.to(device), X_test.to(device)
    y_train, y_test = y_train.to(device), y_test.to(device)

    for epoch in range(epochs):
        model.train()
        y_logits = model(X_train).squeeze()
        y_preds = torch.round(torch.sigmoid(y_logits))

        loss = loss_fn(y_logits, y_train)
        acc = accuracy_score(y_train.detach().numpy(), y_preds.detach().numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.inference_mode():
            test_logits = model(X_test).squeeze()
            test_preds = torch.round(torch.sigmoid(test_logits))

            test_loss = loss_fn(test_logits, y_test)
            test_acc = accuracy_score(y_test.detach().numpy(), test_preds.detach().numpy())

        if epoch % 10 == 0:
            print(f"Epoch: {epoch}\nLoss: {loss:.5f}, Acc: {acc*100:.2f}%\n"
                  f"Test loss: {test_loss:.5f}, Test acc: {test_acc*100:.2f}%\n")

    print(f"---- Final score ----\n"
          f"Loss: {loss:.5f}, Acc: {acc*100:.2f}%\nTest loss: {test_loss:.5f}, Test acc: {test_acc*100:.2f}%\n")
    
    return test_acc

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="script creates a pytorch model based on the given parameters, logs it to the mlflow server")
    parser.add_argument("dataset_number", type=str, help="the version of the training dataset, file storing the dataset has to adhere to the naming scheme")
    parser.add_argument("layer_sizes", type=str, help="layer sizes as a string, e.g. 7,7,7 -> 3 layers of sizes equal to 7")
    parser.add_argument("learning_rate", type=float, help="learning rate of the model")
    parser.add_argument("epochs_num", type=int, help="the number of epochs for the training process")
    parser.add_argument("experiment_name", type=str, help="mlflow experiment name")
    parser.add_argument("--run_name", type=str, help="optional argument for setting a custom mlflow run name", required=False)

    args = parser.parse_args()

    dataset_num = args.dataset_number
    experiment_name = args.experiment_name
    layer_sizes = [int(num) for num in args.layer_sizes.split(",")]

    data = pd.read_csv(f"dataset_{dataset_num}_train.csv")
    data_for_logging = mlflow.data.from_pandas(df=data,
                                               name=f"dataset_{dataset_num}_train")
    data = data.astype({
        "CryoSleep"     : "int8",
        "VIP"           : "int8",
        "Transported"   : "int8"
    })

    device = "cuda" if torch.cuda.is_available() else "cpu"

    X = data.drop(columns=["Transported"]).values
    X = torch.tensor(X, dtype=torch.float32)
    y = data["Transported"].values
    y = torch.tensor(y, dtype=torch.float32)

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=9)
    
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    mlflow.set_experiment(args.experiment_name)
    
    model = binary_classifier(X_train.shape[1], layer_sizes)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

    params = {
        "layer_sizes" : layer_sizes,
        "learning_rate" : args.learning_rate,
        "epochs" : args.epochs_num,
        "optimizer" : type(optimizer).__name__,
        "loss_fn" : type(loss_fn).__name__
    }

    with mlflow.start_run(run_name=args.run_name):

        test_acc = train_test(model=model,
                          epochs=args.epochs_num,
                          loss_fn=loss_fn,
                          optimizer=optimizer,
                          X_train=X_train,
                          y_train=y_train,
                          X_test=X_test,
                          y_test=y_test)
        print(f"Model accuracy: {test_acc*100:.2f}%")
        mlflow.log_artifact(f"dataset_{dataset_num}_train.csv")
        mlflow.set_tag("model", "torch")
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", test_acc)
        mlflow.log_input(data_for_logging, "training")
        mlflow.pytorch.log_model(model, "pytorch", input_example=X_test[0].cpu().numpy().astype(np.float32))