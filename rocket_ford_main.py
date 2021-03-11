import pathlib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeClassifierCV
from ROCKET.rocket_ford_model import RocketModel

DATA_DIR = pathlib.Path("data/")
DATASET = "FordA"

train = pd.read_table(DATA_DIR.joinpath(DATASET, f"{DATASET}_TRAIN.tsv"), header=None)
test = pd.read_table(DATA_DIR.joinpath(DATASET, f"{DATASET}_TEST.tsv"), header=None)

X_train, y_train = torch.tensor(train.to_numpy()[:, 1:]), torch.tensor(train.to_numpy()[:, 0])
X_test, y_test = torch.tensor(test.to_numpy()[:, 1:]), torch.tensor(test.to_numpy()[:, 0])

model = RocketModel(num_kernels=100)

X_train_features = model(X_train.unsqueeze(dim=1)).detach().numpy()

ridge = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
ridge.fit(X_train_features, y_train)

# TEST
X_test_features = model(X_test.unsqueeze(dim=1)).detach().numpy()
print(ridge.score(X_test_features, y_test))
