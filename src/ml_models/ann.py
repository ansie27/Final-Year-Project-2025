import pandas as pd
import numpy as np
from base_risk_predictor import BaseRiskPredictor
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import config

supplier_train_data = pd.read_csv("processed/supplier_dataset_final.csv")
supplier_features = supplier_train_data.drop(columns=['Supplier_ID', 'Risk_Classification'])

commodity_train_data = pd.read_csv("processed/commodity_dataset_final.csv")
commodity_features = commodity_train_data.drop(columns=['2017 NAICS Code'])


class ANNSupplierModel(BaseRiskPredictor):
    def __init__(self):
        self.model = Sequential()
        self.model.add(Dense(64, input_dim=supplier_features.shape[1]-1, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(3, activation='softmax'))

        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self, X_train, y_train, X_val, y_val):
        self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32)

    def evaluate(self, X_test, y_test):
        y_pred = np.argmax(self.model.predict(X_test), axis=-1)
        print(classification_report(y_test, y_pred))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred)


class ANNCommodityModel(BaseRiskPredictor):
    def __init__(self):
        self.model = Sequential()
        self.model.add(Dense(64, input_dim=commodity_features.shape[1]-1, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(3, activation='softmax'))

        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self, X_train, y_train, X_val, y_val):
        self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32)

    def evaluate(self, X_test, y_test):
        y_pred = np.argmax(self.model.predict(X_test), axis=-1)
        print(classification_report(y_test, y_pred))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred)