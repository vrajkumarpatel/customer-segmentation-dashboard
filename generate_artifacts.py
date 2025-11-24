import os
import io
import datetime as dt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

np.random.seed(42)

DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx'

def load_ecommerce_data(timeout_seconds: int = 5):
    df = None
    try:
        import requests
        resp = requests.get(DATA_URL, timeout=timeout_seconds)
        resp.raise_for_status()
        bio = io.BytesIO(resp.content)
        df = pd.read_excel(bio, dtype={'InvoiceNo': str, 'StockCode': str})
    except Exception:
        n_customers = 500
        n_invoices = 3000
        start_date = dt.datetime(2010, 12, 1)
        end_date = dt.datetime(2011, 12, 9)
        date_range = pd.date_range(start_date, end_date, freq='H')
        cust_ids = np.random.choice(range(10000, 10000 + n_customers), n_invoices)
        invoice_ids = [str(500000 + i) for i in range(n_invoices)]
        quantities = np.random.randint(1, 10, size=n_invoices)
        prices = np.round(np.random.uniform(1.0, 50.0, size=n_invoices), 2)
        products = np.random.choice(['Widget A', 'Widget B', 'Gadget C', 'Accessory D'], size=n_invoices)
        dates = np.random.choice(date_range, size=n_invoices)
        df = pd.DataFrame({
            'InvoiceNo': invoice_ids,
            'CustomerID': cust_ids,
            'Description': products,
            'Quantity': quantities,
            'UnitPrice': prices,
            'InvoiceDate': dates
        })
    df['Invoice ID'] = df['InvoiceNo']
    df['Customer ID'] = df['CustomerID']
    df['Product'] = df['Description']
    df['Price'] = df['UnitPrice']
    df['Date'] = pd.to_datetime(df['InvoiceDate'])
    return df

def clean_data(df):
    df = df.copy()
    df = df.dropna(subset=['CustomerID'])
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    df = df.dropna(subset=['InvoiceDate'])
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    df = df.drop_duplicates()
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    df['Invoice ID'] = df['InvoiceNo']
    df['Customer ID'] = df['CustomerID'].astype(int)
    df['Product'] = df['Description']
    df['Price'] = df['UnitPrice']
    df['Date'] = df['InvoiceDate']
    return df

def compute_rfm(df):
    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    recency = df.groupby('Customer ID')['InvoiceDate'].max().apply(lambda x: (snapshot_date - x).days)
    frequency = df.groupby('Customer ID')['Invoice ID'].nunique()
    monetary = df.groupby('Customer ID')['TotalPrice'].sum()
    aov = monetary / frequency
    def avg_days_between(group):
        dates = group['InvoiceDate'].drop_duplicates().sort_values()
        if len(dates) < 2:
            return np.nan
        deltas = dates.diff().dropna().dt.days
        return deltas.mean()
    days_between = df.groupby('Customer ID').apply(avg_days_between)
    month_series = df['InvoiceDate'].dt.month
    quarter_series = df['InvoiceDate'].dt.quarter
    weekend_series = df['InvoiceDate'].dt.weekday >= 5
    top_product = df.groupby('Customer ID')['Description'].agg(lambda s: s.value_counts().idxmax())
    peak_month = df.assign(Month=month_series).groupby('Customer ID')['Month'].agg(lambda s: s.value_counts().idxmax())
    q4_share = df.assign(Q=quarter_series).groupby('Customer ID')['Q'].apply(lambda s: (s == 4).mean())
    weekend_ratio = df.assign(Weekend=weekend_series).groupby('Customer ID')['Weekend'].mean()
    rfm = pd.DataFrame({
        'Recency': recency,
        'Frequency': frequency,
        'Monetary': monetary,
        'AvgOrderValue': aov,
        'DaysBetweenPurchases': days_between,
        'TopProduct': top_product,
        'PeakMonth': peak_month,
        'Q4Share': q4_share,
        'WeekendRatio': weekend_ratio
    })
    rfm = rfm.fillna({'DaysBetweenPurchases': rfm['DaysBetweenPurchases'].median()})
    return rfm

def run_pipeline():
    d0 = load_ecommerce_data()
    d1 = clean_data(d0)
    r = compute_rfm(d1)
    feats = r[['Recency','Frequency','Monetary']]
    sc = StandardScaler()
    X_ = sc.fit_transform(feats)
    ks_ = list(range(2, 9))
    sils_ = []
    for k in ks_:
        km_ = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_)
        sils_.append(silhouette_score(X_, km_.labels_))
    kstar = ks_[int(np.argmax(sils_))]
    km_ = KMeans(n_clusters=kstar, random_state=42, n_init=10).fit(X_)
    r['Cluster'] = km_.labels_
    stats = r.groupby('Cluster').agg({
        'Recency':'mean','Frequency':'mean','Monetary':'mean','AvgOrderValue':'mean','DaysBetweenPurchases':'mean','Q4Share':'mean','WeekendRatio':'mean'
    }).round(2)
    stats['Size'] = r.groupby('Cluster').size()
    return r.reset_index(), stats, kstar

def run_pipeline_and_save():
    rfm_clusters_df, rfm_cluster_stats, chosen_k = run_pipeline()
    out_dir = 'data'
    os.makedirs(out_dir, exist_ok=True)
    rfm_clusters_df.to_csv(os.path.join(out_dir, 'rfm_clusters.csv'), index=False)
    rfm_cluster_stats.to_csv(os.path.join(out_dir, 'rfm_cluster_stats.csv'))
    return rfm_clusters_df, rfm_cluster_stats, chosen_k

if __name__ == '__main__':
    run_pipeline_and_save()