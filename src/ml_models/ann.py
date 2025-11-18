import pandas as pd
import numpy as np
from base_risk_predictor import BaseRiskPredictor
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

class ANNRiskPredictor(BaseRiskPredictor):
    """Single unified ANN model for risk prediction (supplier or commodity)"""
    
    def __init__(self, input_dim, num_classes=3):
        """
        Args:
            input_dim: number of input features
            num_classes: number of risk classes (default 3: low/medium/high)
        """
        self.model = Sequential([
            Dense(64, input_dim=input_dim, activation='relu'),
            Dropout(0.3),  # prevents overfitting
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])
        
        self.model.compile(
            loss='sparse_categorical_crossentropy',  # use if y is integers (0,1,2)
            optimizer='adam',
            metrics=['accuracy']
        )
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train with early stopping"""
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1
        )
        return history
    
    def evaluate(self, X_test, y_test):
        """Evaluate and display results"""
        y_pred = np.argmax(self.model.predict(X_test), axis=-1)
        
        print(classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High']))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
        
        return y_pred

if __name__ == "__main__":
    # Load supplier data
    supplier_data = pd.read_csv("processed/supplier_dataset_final.csv")
    X = supplier_data.drop(columns=['Supplier_ID', 'Risk_Classification'])
    y = supplier_data['Risk_Classification']
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Build and train model
    model = ANNRiskPredictor(input_dim=X_train.shape[1])
    model.train(X_train, y_train, X_val, y_val)
    model.evaluate(X_test, y_test)