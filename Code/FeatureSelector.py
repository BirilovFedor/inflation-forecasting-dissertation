from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

from importlib import reload
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from Data import Dataset
reload(Dataset)
from Data.Dataset import Dataset
import numpy as np
from statsmodels.tsa.api import Holt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings("ignore")
from matplotlib.lines import Line2D
from statsmodels.tsa.api import VAR
from statsmodels.tsa.seasonal import seasonal_decompose
from itertools import combinations
from sklearn.model_selection import TimeSeriesSplit  
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.preprocessing import StandardScaler


class FeatureSelector:

    def __init__(self, target_name) -> None:
        self.target_name = target_name
        self.mi_scores = None

    def compute_mi_scores(self, Data, lag_size=60):
        """
        Compute MI between lagged features and the target.
        Lagging is important — in time series, features at t-1
        predict target at t, not the same timestep.
        """
        df = Data.copy()
        other_cols = [c for c in df.columns if c != self.target_name]

        # Create lagged versions of features
        lagged = {}
        for col in other_cols:
            for lag in range(1, lag_size + 1):
                lagged[f"{col}_lag{lag}"] = df[col].shift(lag)

        lagged_df = pd.DataFrame(lagged)
        lagged_df[self.target_name] = df[self.target_name]
        lagged_df.dropna(inplace=True)

        X = lagged_df.drop(columns=[self.target_name])
        y = lagged_df[self.target_name]

        # Scale before MI computation
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        mi_scores = mutual_info_regression(X_scaled, y, random_state=42)

        self.mi_scores = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
        self.ranked_features = self.mi_scores.index.tolist()
        mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
        mi_normalized = mi_series / mi_series.max()

        print(f"\nNormalized MI Scores (lag_size={lag_size})")
        print("=" * 45)
        for feature, score in mi_normalized.items():
            bar    = "█" * int(score * 20)
            print(f"{feature:25s} {score:.3f}  {bar:20s}")
        print("=" * 45)

        mi_avg = {}
        for col in other_cols:
            lag_cols = [f"{col}_lag{lag}" for lag in range(1, lag_size + 1)]
            mi_avg[col] = mi_series[lag_cols].mean()

        mi_avg_series  = pd.Series(mi_avg).sort_values(ascending=False)
        mi_normalized  = mi_avg_series / mi_avg_series.max()

        print(f"\nNormalized MI Scores — averaged across lags 1 to {lag_size}")
        print("=" * 55)
        for feature, score in mi_normalized.items():
            bar  = "█" * int(score * 20)
            print(f"{feature:25s} {score:.3f}  {bar:20s}")
        print("=" * 55)

        return self.mi_scores
    
    def get_top_k_features(self, k):
        """Return original feature names (without lag suffix) for top-k MI features."""
        if self.mi_scores is None:
            raise ValueError("Run compute_mi_scores first.")

        top_lagged = self.ranked_features[:k]
        # Strip lag suffix to get original column names
        original = list(dict.fromkeys(f.rsplit("_lag", 1)[0] for f in top_lagged))
        return original
    
filepath = '//home//fedor//Dissertation//Data//data_csv.csv'
Data = Dataset(filepath, sep=';')


# create the instance that will be used to store data and process it 
Data = Dataset(filepath, sep=';')
Data.dataset[['CPI(Energy)', 'PPI', 'UNRATE', 'M2', 'GDP', 'IP', 'GS5']]

train_size = len(Data.dataset) - 60

selector = FeatureSelector('CPI(Energy)')
result = selector.compute_mi_scores(Data.dataset[['CPI(Energy)', 'PPI', 'UNRATE', 'M2', 'GDP', 'IP', 'GS5']].iloc[:train_size])
print(result)
print(selector.get_top_k_features(100))