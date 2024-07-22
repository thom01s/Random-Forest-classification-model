from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from seaborn import heatmap
from sklearn import preprocessing

df = pd.read_csv('./WineQT.csv')
X = df.drop(columns=['quality', 'Id'])
y = df.quality

def train_and_predict(X, y, title="Random Forest"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

    rf_classifier = RandomForestClassifier(criterion='entropy', oob_score=True, random_state=1)

    rf_classifier.fit(X_train, y_train)

    y_predict = rf_classifier.predict(X_test)

    rf_classifier.score(X_test, y_test)

    le = preprocessing.LabelEncoder()
    le.fit_transform(np.array(y).flatten())
    labels = list(le.classes_)

    matriz = confusion_matrix(y_test, y_predict)
    df_cm = pd.DataFrame(matriz, index = [i for i in labels], columns = [i for i in labels])
    plt.figure(figsize = (10,7))
    plt.title(title)
    heatmap(df_cm, annot=True, fmt='g')
    plt.xlabel("Previsão")
    plt.ylabel("Verdadeiro")
    plt.show()

    print("*****RF*****")
    print(metrics.classification_report(y_test, y_predict))

train_and_predict(X, y)