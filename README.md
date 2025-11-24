# E-Commerce Customer Segmentation Dashboard

Author: Vrajkumar Patel  
Date: 2025-11-23
Live link - https://vrajkumarpatel-customer-segmentation-dashboard-app-19pydm.streamlit.app/

## Project Overview
- Project that segments e-commerce customers using RFM metrics and K-Means clustering.  
- Objective: identify high-value, loyal, and at-risk customers, and provide actionable insights for marketing and retention.

## Dataset
- Source: UCI Online Retail (Excel). If unavailable, synthetic data is generated to ensure reproducibility.  
- Columns: `InvoiceNo`, `CustomerID`, `Description` (Product), `Quantity`, `UnitPrice`, `InvoiceDate`.  
- Cleaning: drops missing `CustomerID`, removes non-positive quantities/prices, de-duplicates, computes `TotalPrice`.

## Methodology
- Feature Engineering: RFM (Recency, Frequency, Monetary), plus `AvgOrderValue`, `TopProduct`, `PeakMonth`, `Q4Share`, `WeekendRatio`.  
- Clustering: K-Means on standardized RFM features; K selected via silhouette or dynamically in the app.  
- Visualization: Interactive charts with Plotly; tables and downloads for insights.

## Features
- Interactive Streamlit dashboard with tabs:  
  - Overview: KPIs, cluster stats, average RFM per cluster (bar/heatmap).  
  - RFM Analysis: histograms, boxplots, correlation heatmap.  
  - Clustering: scatter plots (Recency vs Frequency, Frequency vs Monetary), monthly sales per cluster.  
  - Insights: top customers per cluster, CSV downloads (filtered, cluster stats, At-Risk re-engagement).  
  - Recommendations: business interpretation, actions, and deployment notes.

## Usage Instructions
### Install Dependencies
```bash
pip install streamlit plotly scikit-learn seaborn matplotlib pandas numpy
```

### Run the Dashboard
```bash
streamlit run app.py
```
Open the URL printed in the terminal (typically `http://localhost:8501`).

### View the Notebook
- Open `Customer_Segmentation_ECommerce.ipynb` in Jupyter or VS Code and run all cells.

## Key Insights
- High-value customers tend to show low Recency (recent), high Frequency, and high Monetary values.  
- At-risk customers have high Recency and lower Frequency; they need win-back campaigns.  
- Product preferences and seasonality enable targeted offers and timing.

## Recommendations
- Loyalty programs and exclusive offers for high-value and loyal segments.  
- Re-engagement emails and limited-time discounts for at-risk segments.  
- Cross-sell and onboarding improvements for low-value segments.

## Screenshots

- Dashboard Overview  
  ![Dashboard Overview](docs/screenshots/Screenshot%202025-11-23%20232758.png)

- Clustering Scatter  
  ![Clustering Scatter](docs/screenshots/Screenshot%202025-11-23%20232838.png)



## Deployment
- All paths are relative; artifacts saved under `data/`.  
- Works on Streamlit Cloud; ensure the dependencies above are installed.  
- If dataset download fails, synthetic data ensures the app remains functional.

## Contact
- Author: Vrajkumar Patel  
- Portfolio ~ vrajpatel.info
