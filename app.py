"""
E-Commerce Customer Segmentation Dashboard

Author: Vrajkumar Patel
Date: 2025-11-23

Description:
Interactive Streamlit dashboard performing RFM analysis and K-Means clustering
on e-commerce transactions to identify high-value, loyal, and at-risk customers.
Includes dynamic controls, interactive Plotly visuals, and downloadable insights.
"""
import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title='E-Commerce Customer Segmentation', layout='wide')

@st.cache_data(show_spinner=False)
def ensure_artifacts():
    """Ensure local CSV artifacts exist for fast startup.

    Returns
    -------
    tuple[str, str]
        Paths to clustered customers CSV and cluster stats CSV.
    """
    clusters_p = os.path.join('data', 'rfm_clusters.csv')
    stats_p = os.path.join('data', 'rfm_cluster_stats.csv')
    if not (os.path.exists(clusters_p) and os.path.exists(stats_p)):
        from generate_artifacts import run_pipeline_and_save
        run_pipeline_and_save()
    return clusters_p, stats_p

clusters_path, stats_path = ensure_artifacts()
rfm_clusters_df = pd.read_csv(clusters_path)
rfm_cluster_stats = pd.read_csv(stats_path)

st.title('E-Commerce Customer Segmentation Dashboard')
st.caption('Objective: Segment customers by purchasing behavior using RFM + K-Means. Methodology: Clean data, engineer features, cluster, visualize, and interpret for actionable insights.')
st.markdown('**Author:** Vrajkumar Patel')
st.markdown('[vrajpatel.info](https://vrajpatel.info)')

clusters = sorted(rfm_clusters_df['Cluster'].unique())
top_products = ['All'] + sorted(rfm_clusters_df['TopProduct'].dropna().unique().tolist())

with st.sidebar:
    st.header('Controls')
    st.markdown('**Author:** Vrajkumar Patel')
    st.markdown('[vrajpatel.info](https://vrajpatel.info)')
    data_source = st.selectbox('Data source', options=['Artifacts (fast)','Remote (UCI)','Synthetic'], index=0)
    k = st.slider('Number of clusters (K-Means)', min_value=2, max_value=10, value=int(rfm_clusters_df['Cluster'].nunique()))
    segment_options = ['All','High-Value','Loyal','At-Risk','Low-Value']
    segment_choice = st.selectbox('Customer segment', options=segment_options, index=0)
    st.header('Filters')
    sel_clusters = st.multiselect('Clusters', options=clusters, default=clusters)
    min_m, max_m = float(rfm_clusters_df['Monetary'].min()), float(rfm_clusters_df['Monetary'].max())
    min_r, max_r = int(rfm_clusters_df['Recency'].min()), int(rfm_clusters_df['Recency'].max())
    min_f, max_f = int(rfm_clusters_df['Frequency'].min()), int(rfm_clusters_df['Frequency'].max())
    monetary_range = st.slider('Monetary', min_value=min_m, max_value=max_m, value=(min_m, max_m))
    recency_range = st.slider('Recency', min_value=min_r, max_value=max_r, value=(min_r, max_r))
    frequency_range = st.slider('Frequency', min_value=min_f, max_value=max_f, value=(min_f, max_f))
    product_choice = st.selectbox('Top Product', options=top_products, index=0)
    id_search = st.text_input('Customer ID contains')
    date_filter_enabled = st.checkbox('Filter by date range (recent purchases)', value=False)

from generate_artifacts import load_ecommerce_data, clean_data, compute_rfm

@st.cache_data(show_spinner=True)
def cached_raw(use_synthetic: bool = False):
    """Load raw transactions from UCI or generate synthetic data.

    Parameters
    ----------
    use_synthetic : bool
        When True, generates synthetic data immediately.

    Returns
    -------
    pandas.DataFrame
        Raw transactions with columns like InvoiceNo, CustomerID, Description, Quantity, UnitPrice, InvoiceDate.
    """
    if use_synthetic:
        try:
            return load_ecommerce_data(timeout_seconds=0)
        except Exception:
            return load_ecommerce_data(timeout_seconds=0)
    return load_ecommerce_data()

@st.cache_data(show_spinner=True)
def cached_clean(df):
    """Clean raw transactions: drop invalids, compute totals, map canonical columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw transactions.

    Returns
    -------
    pandas.DataFrame
        Cleaned transactions with TotalPrice and canonical columns.
    """
    return clean_data(df)

@st.cache_data(show_spinner=True)
def cached_rfm(clean_df):
    """Compute RFM features and seasonality signals per customer.

    Parameters
    ----------
    clean_df : pandas.DataFrame
        Clean transaction records.

    Returns
    -------
    pandas.DataFrame
        Customer-level features including Recency, Frequency, Monetary, AOV,
        DaysBetweenPurchases, TopProduct, PeakMonth, Q4Share, WeekendRatio.
    """
    return compute_rfm(clean_df).reset_index()

if data_source == 'Artifacts (fast)':
    rfm_live = rfm_clusters_df.copy()
    clean_tx = None
elif data_source == 'Synthetic':
    raw_tx = cached_raw(use_synthetic=True)
    clean_tx = cached_clean(raw_tx)
    rfm_live = cached_rfm(clean_tx)
else:
    raw_tx = cached_raw(use_synthetic=False)
    clean_tx = cached_clean(raw_tx)
    if date_filter_enabled:
        dmin = clean_tx['InvoiceDate'].min().date()
        dmax = clean_tx['InvoiceDate'].max().date()
        date_range = st.date_input('Date range', [dmin, dmax])
        if isinstance(date_range, list) and len(date_range) == 2:
            start, end = date_range
        else:
            start, end = dmin, dmax
        mask = (clean_tx['InvoiceDate'].dt.date >= start) & (clean_tx['InvoiceDate'].dt.date <= end)
        clean_tx_filtered = clean_tx.loc[mask]
        rfm_live = cached_rfm(clean_tx_filtered)
    else:
        rfm_live = cached_rfm(clean_tx)
features = rfm_live[['Recency','Frequency','Monetary']]
scaler = StandardScaler()
X = scaler.fit_transform(features)
km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
rfm_live['Cluster'] = km.labels_

# Update product options from current data
top_products = ['All'] + sorted(rfm_live['TopProduct'].dropna().unique().tolist())

df = rfm_live.copy()
df = df[(df['Monetary'] >= monetary_range[0]) & (df['Monetary'] <= monetary_range[1])]
df = df[(df['Recency'] >= recency_range[0]) & (df['Recency'] <= recency_range[1])]
df = df[(df['Frequency'] >= frequency_range[0]) & (df['Frequency'] <= frequency_range[1])]
if product_choice != 'All':
    df = df[df['TopProduct'] == product_choice]
if id_search.strip():
    df = df[df['Customer ID'].astype(str).str.contains(id_search.strip())]

overall = rfm_live[['Recency','Frequency','Monetary']]
qR25, qR50, qR75 = np.quantile(overall['Recency'], [0.25, 0.50, 0.75])
qF25, qF50, qF75 = np.quantile(overall['Frequency'], [0.25, 0.50, 0.75])
qM25, qM50, qM75 = np.quantile(overall['Monetary'], [0.25, 0.50, 0.75])

stats = rfm_live.groupby('Cluster').agg({
    'Recency':'mean','Frequency':'mean','Monetary':'mean','AvgOrderValue':'mean','DaysBetweenPurchases':'mean','Q4Share':'mean','WeekendRatio':'mean','Customer ID':'count'
}).rename(columns={'Customer ID':'Size'}).reset_index()

def label_row(r, qR25=qR25, qR50=qR50, qR75=qR75, qF25=qF25, qF50=qF50, qF75=qF75, qM25=qM25, qM50=qM50, qM75=qM75):
    if (r['Monetary'] >= qM75) and (r['Frequency'] >= qF75) and (r['Recency'] <= qR25):
        return 'High-Value'
    if (r['Recency'] >= qR75) and (r['Frequency'] <= qF25):
        return 'At-Risk'
    if (r['Frequency'] >= qF75) and (qM50 <= r['Monetary'] < qM75) and (r['Recency'] <= qR50):
        return 'Loyal'
    return 'Low-Value'

stats['Segment'] = stats.apply(label_row, axis=1)
label_map = dict(zip(stats['Cluster'], stats['Segment']))
rfm_live['Segment'] = rfm_live['Cluster'].map(label_map)
df['Segment'] = df['Cluster'].map(label_map)
if segment_choice != 'All':
    df = df[df['Segment'] == segment_choice]

tab_overview, tab_rfm, tab_cluster, tab_insights, tab_reco = st.tabs(['Overview','RFM Analysis','Clustering','Insights','Recommendations'])

with tab_overview:
    col0, col1, col2, col3 = st.columns(4)
    col0.metric('Number of clusters', f"{k}")
    col1.metric('Customers', f"{len(df):,}")
    col2.metric('Total revenue', f"{df['Monetary'].sum():,.2f}")
    col3.metric('Avg frequency', f"{df['Frequency'].mean():,.2f}")
    top_cluster = stats.sort_values('Monetary', ascending=False).iloc[0]
    st.markdown(f"**Top Cluster by Revenue:** Cluster {int(top_cluster['Cluster'])} ({top_cluster['Segment']}) — Monetary {top_cluster['Monetary']:,.2f}")
    st.subheader('Cluster Statistics')
    st.dataframe(stats, use_container_width=True)
st.subheader('Average RFM per Cluster')
normalize = st.checkbox('Normalize (z-score)', value=False)
heatmap_style = st.checkbox('Show as heatmap', value=False)
cluster_order = stats.sort_values('Monetary', ascending=False)['Cluster'].tolist()
rfm_bars = stats[['Cluster','Recency','Frequency','Monetary']].melt(id_vars='Cluster', var_name='Metric', value_name='Value')
if normalize:
    rfm_bars['Value'] = rfm_bars.groupby('Metric')['Value'].transform(lambda s: (s - s.mean()) / s.std(ddof=0))
if heatmap_style:
    rfm_matrix = rfm_bars.pivot(index='Cluster', columns='Metric', values='Value').reindex(cluster_order)
    fig_bar = px.imshow(rfm_matrix, text_auto=True, aspect='auto', title='Average RFM per Cluster', color_continuous_scale='Blues', labels=dict(color=('z-score' if normalize else 'Average')))
else:
    fig_bar = px.bar(rfm_bars, x='Cluster', y='Value', color='Metric', barmode='group', height=500, title='Average RFM per Cluster', category_orders={'Cluster': cluster_order}, labels={'Value': ('z-score' if normalize else 'Average')}, color_discrete_sequence=px.colors.qualitative.Set2)
    fig_bar.update_traces(texttemplate='%{y:.2f}', textposition='outside')
fig_bar.update_layout(template='plotly_white', legend_title_text='Metric', xaxis_title='Cluster')
st.plotly_chart(fig_bar, use_container_width=True)

with tab_rfm:
    fig_hr = px.histogram(rfm_live, x='Recency', nbins=30, title='Recency Distribution')
    fig_hf = px.histogram(rfm_live, x='Frequency', nbins=30, title='Frequency Distribution')
    fig_hm = px.histogram(rfm_live, x='Monetary', nbins=30, title='Monetary Distribution')
    st.plotly_chart(fig_hr, use_container_width=True)
    st.plotly_chart(fig_hf, use_container_width=True)
    st.plotly_chart(fig_hm, use_container_width=True)
    fig_br = px.box(rfm_live, y='Recency', points=False, title='Recency Boxplot')
    fig_bf = px.box(rfm_live, y='Frequency', points=False, title='Frequency Boxplot')
    fig_bm = px.box(rfm_live, y='Monetary', points=False, title='Monetary Boxplot')
    st.plotly_chart(fig_br, use_container_width=True)
    st.plotly_chart(fig_bf, use_container_width=True)
    st.plotly_chart(fig_bm, use_container_width=True)
    corr = rfm_live[['Recency','Frequency','Monetary','AvgOrderValue','DaysBetweenPurchases','Q4Share','WeekendRatio']].corr()
    fig_corr = px.imshow(corr, text_auto=True, aspect='auto', title='Feature Correlation Heatmap', color_continuous_scale='RdBu')
    st.plotly_chart(fig_corr, use_container_width=True)

with tab_cluster:
    fig1 = px.scatter(df, x='Frequency', y='Monetary', color='Cluster', hover_data=['Customer ID','TopProduct','Segment'], height=500, title='Clusters: Frequency vs Monetary')
    st.plotly_chart(fig1, use_container_width=True)
    fig2 = px.scatter(df, x='Recency', y='Frequency', color='Cluster', hover_data=['Customer ID','TopProduct','Segment'], height=500, title='Clusters: Recency vs Frequency')
    st.plotly_chart(fig2, use_container_width=True)
    st.subheader('Monthly Sales per Cluster')
    if clean_tx is not None:
        cluster_map = rfm_live[['Customer ID','Cluster']]
        tx = clean_tx.merge(cluster_map, on='Customer ID', how='inner')
        tx['Month'] = tx['InvoiceDate'].dt.to_period('M').dt.to_timestamp()
        monthly_sales = tx.groupby(['Month','Cluster'])['TotalPrice'].sum().reset_index()
        fig3 = px.line(monthly_sales, x='Month', y='TotalPrice', color='Cluster', markers=True, title='Monthly Sales per Cluster')
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info('Using fast artifacts: switch Data source to Remote or Synthetic to view monthly sales time-series.')

with tab_insights:
    sel_cluster_for_top = st.selectbox('Select cluster for top customers', options=sorted(df['Cluster'].unique()))
    top5 = df[df['Cluster'] == sel_cluster_for_top].sort_values('Monetary', ascending=False).head(5)[['Customer ID','Monetary','Frequency','Recency','TopProduct','Segment']]
    st.dataframe(top5, use_container_width=True)
    st.subheader('Downloads')
    st.download_button('Download filtered customers CSV', df.to_csv(index=False), file_name='filtered_customers.csv')
    st.download_button('Download cluster stats CSV', stats.to_csv(index=False), file_name='cluster_stats.csv')
    at_risk = rfm_live[rfm_live['Segment'] == 'At-Risk'][['Customer ID','Cluster','Segment','Recency','Frequency','Monetary','TopProduct','PeakMonth','Q4Share','WeekendRatio']]
    st.download_button('Download re-engagement list (At-Risk)', at_risk.to_csv(index=False), file_name='at_risk_reengagement.csv')

with tab_reco:
    st.markdown('### RFM Analysis')
    st.markdown('- Recency: days since last purchase; lower is better.')
    st.markdown('- Frequency: number of purchases; higher indicates loyalty.')
    st.markdown('- Monetary: total spend; higher indicates value.')
    st.markdown('### Cluster Interpretation')
    st.markdown('- High-value clusters often have low recency, high frequency, and high monetary.')
    st.markdown('- At-risk clusters show high recency and lower frequency.')
    st.markdown('- Use TopProduct, PeakMonth, Q4Share, and WeekendRatio to tailor promotions.')
    st.markdown('- Re-engage At-Risk with timely offers and reminders.')
    st.markdown('### Recommendations')
    st.markdown('- Loyalty programs and exclusive offers for High-Value and Loyal segments.')
    st.markdown('- Win-back campaigns and limited-time discounts for At-Risk segment.')
    st.markdown('- Cross-sell essentials to Low-Value and guide onboarding.')
    st.markdown('### Conclusion')
    st.markdown('This dashboard enables data-driven segmentation and targeted actions. Adjust K, filters, and segments to refine marketing strategies and track seasonality.')
    with st.expander('About & Deployment'):
        st.markdown('- If dataset download fails, the pipeline generates synthetic data with required fields.')
        st.markdown('- All paths are relative; artifacts save under `data/`.')
        st.markdown('- To run locally: install packages and `streamlit run app.py`. Compatible with Streamlit Cloud.')