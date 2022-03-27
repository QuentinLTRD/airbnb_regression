import pandas as pd
from sklearn.model_selection import ShuffleSplit
import rampwf as rw


problem_title = "Airbnb price per night regression in Bordeaux"

Predictions = rw.prediction_types.make_regression()
workflow = rw.workflows.Regressor()

score_types = [rw.score_types.RMSE(name="rmse")]


def _get_data(path=".", split="train"):
    df = pd.read_csv(rf"{path}\data\{split}\{split}.csv")
    y = df["prix_nuitee"]
    X = df.drop(columns=["prix_nuitee"])
    return X.to_numpy(), y.to_numpy()


def get_train_data(path="."):
    return _get_data(path, "train")


def get_test_data(path="."):
    return _get_data(path, "test")


def get_cv(X, y):
    cv = ShuffleSplit(n_splits=10, test_size=0.25, random_state=26)
    return cv.split(X, y)
