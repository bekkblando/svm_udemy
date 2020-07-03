import csv
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler
import pandas as pd

def get_minst():
    # Read our data
    data = pd.read_csv('minst_train.csv')

    # Shuffle the data
    data.sample(frac=1)

    # Split between training and test data - split in two
    split_point = int(len(data)/2)
    train = data[:split_point]
    test = data[split_point:]

    # Split Pixels from Label
    Xtrain = train.loc[:,'pixel0':]
    Ytrain = train.loc[:,'label']
    Xtest = test.loc[:,'pixel0':]
    Ytest = test.loc[:,'label']

    # Normalizing The Data
    scaler = StandardScaler()
    Xtrain = scaler.fit_transform(Xtrain)
    Xtest = scaler.transform(Xtest)

    return Xtrain, Ytrain, Xtest, Ytest

# Expecting 738 pixel array
def to_img(pixels):
    data = np.array(pixels).reshape(28,28).astype(np.uint8)
    img = Image.fromarray(data)
    return img


# with open('minst_train.csv') as csvfile:
#     imgs = csv.reader(csvfile)
#     next(imgs)
#     for pixels in imgs:
#         pixels = [int(pixel) for pixel in pixels]
# #         print(', '.join(pixels[1:]))
#         print(len(pixels[1:]))
#         print(pixels[0])
#         to_img(pixels[1:]).show()
#         break
        