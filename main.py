import pandas as pd

from bagging import bagging_method
from boosting import boosting_method
from stacking import load_and_preprocess_data, stacking_method


if __name__ == '__main__':
    pd_frames = pd.read_csv("our_Dataset.csv")
    boosting_method(pd_frames)
    # pd_frames.sample(frac=1)
    # pd_frames.to_csv('newDataSet.csv', sep=',', encoding='utf-8')
