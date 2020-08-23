import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time


def showDigit(df, row):
    row = df.iloc[7, :-1]
    row = row.values.reshape(28,28)
    img = Image.fromarray(np.uint8(row) , 'L')
    img.show()



df=pd.read_csv('datasets/mnist_784.csv', sep=',', header=[0])

X = df.iloc[:,:-1].values
y = df.iloc[:,-1:].values.flatten()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
neigh = KNeighborsClassifier(n_neighbors=3)
tic = time.perf_counter()
neigh.fit(X_train, y_train)
toc = time.perf_counter()
print(f"Model was fit in {toc - tic:0.4f} seconds")
tic = time.perf_counter()
y_pred = neigh.predict(X_test)
toc = time.perf_counter()
print(f"Test set predicted in {toc - tic:0.4f} seconds")
score = accuracy_score(y_test, y_pred)
print("Accuracy:"+str(score))