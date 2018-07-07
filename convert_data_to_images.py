import pandas as pd
import numpy as np


def convert_to_image(path):
    df = pd.read_csv(path)
    df = df.values
    images = []
    labels = []
    for i in range(df.shape[0]):
        images.append(df[i][1:].reshape(28, 28))
        labels.append(df[i][0])
    images = np.array(images)
    labels = np.array(labels)
    return images, labels
