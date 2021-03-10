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

model = RocketModel(num_kernels=1000)
classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
X_train_transform = model(X_train.unsqueeze(dim=1))
classifier.fit(X_train_transform, y_train)

X_val_transform = model(X_test.unsqueeze(dim=1))
print(classifier.score(X_val_transform, y_test))
