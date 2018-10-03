import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from datetime import datetime

def get_dataset_pulsi(columns,
                      filename='./utils/test_data/42nights.csv'):
    # read dataset
    dataset = pd.read_csv(filename, delimiter=',')
    dataset["datetime"] = pd.to_datetime(dataset["date_chron"], format="(%y-%m-%d %H:%M:%S)")
    dataset.drop("date_chron", axis=1, inplace=True)

    # reduce number of columns
    dataset_reduced_columns = dataset.filter(np.append(columns, ["datetime", "name"]))

    # standarize (TODO this should be done to test only...)
    scaler = MinMaxScaler()
    dataset_reduced_std = scaler.fit_transform(dataset_reduced_columns.drop(["datetime", "name"],
                                                                            axis=1).values)
    dataset_reduced_std = pd.DataFrame(data=dataset_reduced_std, columns=columns)
    dataset_reduced_std["datetime"] = dataset_reduced_columns["datetime"].values
    dataset_reduced_std["name"] = dataset_reduced_columns["name"].values

    return dataset_reduced_std, dataset
