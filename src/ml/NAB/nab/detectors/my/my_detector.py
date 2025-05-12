import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

from nab.detectors.base import AnomalyDetector

WINDOW_SIZE = 32

class MyDetector(AnomalyDetector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = load_model('../lstm-ae-32-model-bs256.h5')

    def handleRecord(self, input_data):
        preds = self.model.predict(input_data, verbose=1, batch_size=32)
        mse = np.square(input_data - preds)
        mse = mse[:, -1, 0]
        q75 = np.quantile(mse, 0.998)
        q25 = np.quantile(mse, 0.002)
        iqr = q75 - q25

        anomaly_scores = []
        for i in range(len(mse)):
            e = mse[i]
            if e <= q25 or e < 1e-3:
                anomaly_scores.append(0.001)
            elif e >= q75:
                anomaly_scores.append(0.999)
            else:
                anomaly_scores.append((e - q25) / (iqr + 1e-6))

        # anomaly_scores[anomaly_scores > 1] = 1
        # anomaly_scores[anomaly_scores < 1e-6] = 0
        # anomaly_scores = [random.random() for _ in range(len(input_data))]
        return anomaly_scores

    def run(self):
        headers = self.getHeader()
        rows = []

        df = self.dataSet.data
        scaler = StandardScaler()
        normalized = scaler.fit_transform(df[['value']])

        for i in range(WINDOW_SIZE):
            rows.append(list(df.iloc[i]) + [0.001])

        windows = []
        for i in range(WINDOW_SIZE, len(normalized)):
            windows.append(normalized[i - WINDOW_SIZE:i])
        scores = self.handleRecord(np.array(windows))
        
        for i, score in zip(range(WINDOW_SIZE, len(normalized)), scores, strict=True):
            rows.append(list(df.iloc[i]) + [score])

        ans = pd.DataFrame(rows, columns=headers)
        return ans
