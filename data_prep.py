import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

# Datasets
bitcoin_price_data = pd.read_csv('BTC_USD.csv')
google_trend_data = pd.read_csv('google_trend_bitcoin.csv')

# Print 
print(bitcoin_price_data.head(5))
print(google_trend_data.head(5))

# Merge 
merged = pd.concat([bitcoin_price_data,google_trend_data], axis=1)
print(last.head(5))

# to scv save  
merged.to_csv('multivariant_data.csv', index=False)
