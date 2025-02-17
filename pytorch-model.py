import torch
from torch import nn
import pandas as pd
import mlflow
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

def log(params, # layer_sizes, lr, epochs
        test_acc,
        model,
        dataset_version):

    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    mlflow.set_experiment("spaceship-titanic-kaggle")

    with mlflow.start_run():
        mlflow.set_tag("dataset", dataset_version)
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", test_acc)
        mlflow.pytorch.log_model(model, "torch-model")
    
    """
    notes:
    - there is an option to specify the dataset in a better way
    - seems like "with mlflow.start_run()" measures time, so the training
      maybe should be instead done inside it to automatically get the time taken
    """


def train_test_log(X_train,
                   X_test,
                   y_train,
                   y_test,
                   layer_sizes=[10, 10],
                   loss_fn=nn.BCEWithLogitsLoss(),
                   lr=0.01,
                   epochs=10,
                   dataset_version="xx"):

    model = binary_classifier(X_train.shape[1], layer_sizes)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    test_acc = train_test(model=model,
                          epochs=epochs,
                          loss_fn=loss_fn,
                          optimizer=optimizer,
                          X_train=X_train,
                          y_train=y_train,
                          X_test=X_test,
                          y_test=y_test)

    params = {
        "layer_sizes" : layer_sizes,
        "lr" : lr,
        "epochs" : epochs
    }

    log(params=params,
        test_acc=test_acc,
        model=model,
        dataset_version=dataset_version)

if __name__ == "__main__":

    device = "cpu" # sth wrong with my gpu, device hardcoded to cpu

    train_data = pd.read_csv("dataset_02_train.csv")
    train_data = train_data.astype({
        "CryoSleep"     : "int8",
        "VIP"           : "int8",
        "Transported"   : "int8"
    })
    X = train_data.drop(columns=["Transported"]).values
    X = torch.tensor(X, dtype=torch.float32)
    y = train_data["Transported"].values
    y = torch.tensor(y, dtype=torch.float32)

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=9)
    
    
    train_test_log(X_train=X_train,
                   X_test=X_test,
                   y_train=y_train,
                   y_test=y_test,
                   layer_sizes=[10, 10],
                   lr=0.001,
                   epochs=1000,
                   dataset_version="02")