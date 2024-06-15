# In the Terminal, install:
# 1. Pandas: pip install pandas
# 2. Scikit: pip install -U scikit-learn
# The above initially did not work, but PyCharm suggested installing them after I tried to import them below

# Import necessary Packages
import pandas as pd
import pandas_ta as ta
import technical_analysis
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# tickers = ["NVDA", "AMD", "AVGO", "SMCI", "SOXX", "SPY"]  # Can add more tickers as needed  # TODO reactivate
tickers = ["NVDA", "AMD", "SMH", "SPY", "QQQ", "DIA", "VOO", "PYPL"]  # TODO deactivate
target_ticker = "NVDA"  # Can only take ONE stock
tickers.sort()  # For alphabetical order
stock_data_total = yf.download(tickers, start="2012-01-01", end="2024-01-01",
                               interval="1d", actions="True", prepost="True")
# prepost - for outside mkt hrs, actions - for divs/splits
# print(stock_data_total.columns.tolist())  # To check for Column Names

# needed_columns = ["Adj Close", "Volume", "Dividends", "Stock Splits"]  # TODO reactivate
needed_columns = ["Adj Close", "Volume"]  # TODO deactivate
stock_data = stock_data_total.loc[:, needed_columns]  # I have extracted the necessary data
# print(stock_data.loc[:, ("Adj Close", "NVDA")])  # This is the function to extract close px for NVDA for e.g.

# Create % Change values in DataFrame
for k in tickers:
    stock_data[("5d_pct_change", k)] = stock_data[("Adj Close", k)].pct_change(5, fill_method=None)
    # 5 days % change created
    stock_data[("5d_future_px", k)] = stock_data[("Adj Close", k)].shift(-5, axis=0)
    stock_data[("5d_future_pct_change", k)] = stock_data[("5d_future_px", k)].pct_change(5, fill_method=None)
    # 5 days future % change
# 5d_future_pct_change is the TARGET
# 5d_pct_change is one of the FEATURES

# Engineering new features (e.g. SMA200, RSI and correlation)
# Adding SMA to Stock_Data DataFrame
sma_lengths = (50, 100, 200)
for n in sma_lengths:
    for k in tickers:
        stock_data[("SMA" + str(n), k)] = stock_data.ta.sma(length=int(n), close=stock_data[("Adj Close", k)])
# Adding RSI to Stock_Data
rsi_lengths = (14, 30, 50)
for n in rsi_lengths:
    for k in tickers:
        stock_data["RSI" + str(n), k] = stock_data.ta.rsi(length=int(n), close=stock_data[("Adj Close", k)])

# Defining Target Names (Single Stock) and Extracting Target Data
target_names = [("5d_future_pct_change", target_ticker)]
target_data = stock_data[target_names]
target_data = target_data.dropna()
# print(target_data.isna().values.any())  # Checking for NaN values in df, if no NaN, returns "False"

# Defining feature names
feature_names = ["5d_pct_change"]  # Default feature name, add the others behind
for n in sma_lengths:
    feature_names.append("SMA" + str(n))  # Adding SMA to feature names
for n in rsi_lengths:
    feature_names.append("RSI" + str(n))  # Adding RSI to feature names

# Extracting Feature Data using the feature names
feature_data = stock_data[feature_names]
feature_data = feature_data.dropna()
# print(feature_data.isna().values.any())  # Checking for NaN lines, if no NaN, returns "False"

# Only keep common indices between feature and target data
common_indices = feature_data.index.intersection(target_data.index)
feature_data = feature_data.loc[common_indices]
target_data = target_data.loc[common_indices]

# Split Features/Targets into Train/Test sets
train_features, test_features, train_targets, test_targets = (
    train_test_split(feature_data, target_data, test_size=0.15, random_state=12)
)

# Use RandomForestRegressor from SciKit to create an instance of RFR
# Specify model hyperparameters
rfr = RandomForestRegressor(max_depth=3, max_features=10, random_state=12, n_estimators=200)

# RFR takes 1D arrays so, convert train_targets from DF into 1D Array  # todo Check This
train_targets = train_targets.squeeze()
# Now, train the model using the train data .fit() used
rfr.fit(train_features, train_targets)
# Forcing train_targets back into DataFrame
train_targets = train_targets.to_frame(name=target_ticker)

# Use trained model to predict outcome
train_predictions = rfr.predict(train_features)  # These are arrays
test_predictions = rfr.predict(test_features)
# Forcing predictions from Arrays into DataFrames
train_predictions = pd.DataFrame(train_predictions,
                                 columns=[target_ticker],
                                 index=train_targets.index
                                 )
test_predictions = pd.DataFrame(test_predictions,
                                columns=[target_ticker],
                                index=test_targets.index
                                )

print(rfr.score(train_features, train_targets))
print(rfr.score(test_features, test_targets))

# Testing Feature Importance
imp = rfr.feature_importances_
sorted_index = np.argsort(imp)[::-1]
imp_x_values = np.array(range(0, len(imp)))  # This is just a 1D array of 1, 2,..., len(imp), for x-axis of Bar Chart
labels = np.array(feature_data.columns)[sorted_index]
plt.figure(3)
plt.bar(imp_x_values, imp[sorted_index], tick_label=labels)
plt.subplots_adjust(bottom=0.22, top=0.955)
plt.xticks(rotation=90)

# Plotting Subplots for each stock's prediction against its actual targets
# Figure 1 is for TRAIN Predictions vs. TRAIN Targets
plt.figure(1)
plt.scatter(train_targets, train_predictions,
            label=target_ticker, s=2, alpha=0.7)
plt.legend()
plt.grid()
plt.suptitle("TRAIN Predictions vs. Targets")

# Figure 2 is for TEST Predictions vs. TEST Targets
plt.figure(2)
plt.scatter(test_targets, test_predictions,
            label=target_ticker, s=2, alpha=0.7)
plt.legend()
plt.grid()
plt.suptitle("TEST Predictions vs. Targets")

# General Plotting
plt.tight_layout()
plt.show()
# plt.clf()

# Get predictions using .predict() then plot them on a scatterplot
# .score is useful in determining the performance of the model via R2 value
# I can find the importance of features under the Random Forest Model
# In order to increase the fit of model, I can use Gradient Boosting (GB) to enhance fit
# Also, there are KNN (K-Nearest Neighbors_ and Neural Networks

# TODO Thinking of using feature importance to choose my features more carefully
# TODO Do I want to include some way to sort through hyperparameters?
# TODO Add the GBR algorithm as well
