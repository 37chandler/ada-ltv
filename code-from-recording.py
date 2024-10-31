# This code is reproduced directly from the assigned talk: 
# https://www.youtube.com/watch?v=486x8ccQThE


# Install necessary libraries
# These would be added to a requirements.txt file
# ShopifyAPI==11.0.0
# python-dotenv==0.20.0
# pandas==1.4.2
# Lifetimes==0.11.3
# pip install -r requirements.txt

import pandas as pd
import shopifydata as sd # We don't need this in ADA 

# Function to create a DataFrame from Shopify orders
def order_frame(orders):
    all_orders = []
    for order in orders:
        o = order.attributes
        record = {k: o.get(k, None) for k in ('id', 'created_at', 'subtotal_price')}
        record['customer_id'] = o['customer'].attributes['id']
        all_orders.append(record)
    df = pd.DataFrame(all_orders)
    return df

# Load orders and convert price data to float
orders = sd.get_data('Order')
df = order_frame(orders)
df.subtotal_price = df.subtotal_price.astype(float).fillna(0.0)
print(df)

# Create summary DataFrame
from lifetimes.utils import summary_data_from_transaction_data as summary

def create_summary(df):
    sf = summary(df, 'customer_id', 'created_at', observation_period_end='2022-10-31', freq='M').reset_index()
    mf = df.groupby('customer_id')[['subtotal_price']].mean()
    mf = mf.reset_index()
    mf = mf.rename(columns={'subtotal_price': 'monetary_value'})
    df = mf.merge(sf, on='customer_id')
    return df

print(create_summary(df))

# Fitting BetaGeoFitter model
from lifetimes import BetaGeoFitter

bgf = BetaGeoFitter(penalizer_coef=0.01)
bgf.fit(df['frequency'], df['recency'], df['T'])

# Fitting GammaGammaFitter model
from lifetimes import GammaGammaFitter

returning = df[df['frequency'] > 0]
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(returning['frequency'], returning['monetary_value'])

# Predicting Customer Lifetime Value (CLV)
ggf.customer_lifetime_value(
    bgf, 
    df['frequency'], 
    df['recency'], 
    df['T'], 
    df['monetary_value'], 
    time=12, 
    discount_rate=0.1, 
    freq="M"
).to_frame()

# Calculate equity and expected purchases for each customer
def equity(row):
    rest = row['rest'].values[0]
    return ggf.customer_lifetime_value(
        bgf,
        row['frequency'],
        row['recency'],
        row['T'],
        row['monetary_value'],
        time=rest,
        discount_rate=0.1,
        freq="M"
    )

def e_purchases(row, bgf):
    return bgf.conditional_expected_number_of_purchases_up_to_time(
        row['rest'],
        row['frequency'],
        row['recency'],
        row['T']
    )

# Looping through each customer and adding equity and purchases to DataFrame
df['equity'] = 0
for i, row in df.iterrows():
    df.at[i, 'equity'] = equity(row.to_frame().T).values[0]
df['clv'] = df['revenue'] + df['equity']
df['purchases'] = df.apply(lambda row: e_purchases(row, bgf) + row['frequency'] + 1, axis=1)

# Aggregating metrics by frequency
df[df.rest == 21].groupby('frequency').agg(
    clv_avg=('clv', 'mean'),
    equity_avg=('equity', 'mean'),
    revenue_avg=('revenue', 'mean'),
    purchases_avg=('purchases', 'mean')
)

# Plotting expected orders based on recency and frequency
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0., 12., 1)
plt.plot(x, orders(1, x, 12), 'r--', label='Freq 1')
plt.plot(x, orders(5, x, 12), 'b--', label='Freq 5')
plt.plot(x, orders(9, x, 12), 'y--', label='Freq 9')
plt.plot(x, orders(15, x, 12), 'g--', label='Freq 15')

plt.ylabel('Expected Orders')
plt.xlabel('Recency')
plt.legend()
plt.show()
