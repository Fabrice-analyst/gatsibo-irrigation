"""
GATSIBO SMART IRRIGATION SCHEDULER - WEEK 5
============================================
Streamlit Web Application

Author: Fabrice RUTAGARAMA
Date: November 2025

FROM JUPYTER NOTEBOOK TO REAL WEB APP!
This makes your tool accessible to farmers, extension officers, and decision makers.

To run this app:
1. Save this file as 'app.py'
2. Open terminal/cmd in the same folder
3. Run: streamlit run app.py
4. Browser will open automatically!
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Gatsibo Smart Irrigation",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #43A047;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1E88E5;
    }
    .recommendation-box {
        background-color: #e8f5e9;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #43A047;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD DATA AND MODEL
# ============================================================================

@st.cache_data
def load_data():
    """Load all data files"""
    try:
        data = pd.read_csv('gatsibo_complete_irrigation_data.csv', index_col=0, parse_dates=True)
        weekly = pd.read_csv('gatsibo_irrigation_schedule_weekly.csv', index_col=0, parse_dates=True)
        forecast = pd.read_csv('irrigation_forecast_7days.csv', parse_dates=['date'])
        return data, weekly, forecast
    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}")
        st.stop()

@st.cache_resource
def load_model():
    """Load ML model"""
    try:
        with open('irrigation_ml_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('model_features.pkl', 'rb') as f:
            features = pickle.load(f)
        return model, features
    except FileNotFoundError:
        st.warning("ML model not found. Some features will be limited.")
        return None, None

data, weekly_schedule, forecast_7day = load_data()
model, model_features = load_model()

# ============================================================================
# HEADER
# ============================================================================

st.markdown('<h1 class="main-header">Gatsibo Smart Irrigation Scheduler</h1>', unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; color: #666; margin-bottom: 2rem;'>
        AI-Powered Irrigation Recommendations for Sustainable Agriculture in Rwanda
    </div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/1/17/Flag_of_Rwanda.svg", width=100)
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose a section:",
    ["Dashboard", "7-Day Forecast", "Historical Analysis", "About Gatsibo", "About This Tool"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Quick Stats")
st.sidebar.metric("Total Days Analyzed", f"{len(data):,}")
st.sidebar.metric("Average Daily Irrigation", f"{data['Irrigation_requirement_mm'].mean():.2f} mm")
st.sidebar.metric("ML Model Accuracy", "77% (R² = 0.77)")

st.sidebar.markdown("---")
st.sidebar.markdown("""
    <div style='font-size: 0.8rem; color: #666;'>
    <b>Built by:</b> Fabrice RUTAGARAMA<br>
    <b>Institution:</b> University of Rwanda<br>
    <b>Data Period:</b> 2019-2024<br>
    <b>Location:</b> Gatsibo District, Eastern Province
    </div>
""", unsafe_allow_html=True)

# ============================================================================
# PAGE: DASHBOARD
# ============================================================================

if page == "Dashboard":
    
    st.markdown('<h2 class="sub-header">Current Week Status</h2>', unsafe_allow_html=True)
    
    latest_week = weekly_schedule.iloc[-1]
    week_date = weekly_schedule.index[-1]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Week Ending", week_date.strftime('%b %d, %Y'))
    
    with col2:
        st.metric(
            "Irrigation Needed",
            f"{latest_week['Irrigation_needed_mm']:.1f} mm",
            delta=f"{latest_week['Irrigation_needed_mm'] - weekly_schedule.iloc[-2]['Irrigation_needed_mm']:.1f} mm"
        )
    
    with col3:
        st.metric("Rainfall Received", f"{latest_week['Rainfall_week_mm']:.1f} mm")
    
    with col4:
        st.metric("Crop Water Need", f"{latest_week['ETc_week_mm']:.1f} mm")
    
    if latest_week['Irrigation_needed_mm'] < 5:
        recommendation = "MINIMAL irrigation needed"
        color = "#4CAF50"
        advice = "Recent rainfall is sufficient. Monitor crop condition."
    elif latest_week['Irrigation_needed_mm'] < 20:
        recommendation = "LIGHT irrigation recommended"
        color = "#FFC107"
        advice = "Supplement rainfall with light irrigation."
    elif latest_week['Irrigation_needed_mm'] < 40:
        recommendation = "MODERATE irrigation required"
        color = "#FF9800"
        advice = "Regular irrigation needed to maintain crop health."
    else:
        recommendation = "HEAVY irrigation required"
        color = "#F44336"
        advice = "Crop water stress likely. Irrigate immediately!"
    
    st.markdown(f"""
        <div style='background-color: {color}22; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid {color}; margin: 1rem 0;'>
            <h3 style='margin: 0; color: {color};'>{recommendation}</h3>
            <p style='margin: 0.5rem 0 0 0; font-size: 1.1rem;'>{advice}</p>
            <p style='margin: 0.5rem 0 0 0;'><b>Apply:</b> {latest_week['Irrigation_needed_mm']:.0f} mm this week (split into 2-3 applications)</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<h2 class="sub-header">Recent Water Balance (Last 12 Weeks)</h2>', unsafe_allow_html=True)
    
    recent_weeks = weekly_schedule.tail(12)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=recent_weeks.index, y=recent_weeks['ETc_week_mm'], name='Crop Water Need', marker_color='#EF5350'))
    fig.add_trace(go.Bar(x=recent_weeks.index, y=recent_weeks['Rainfall_effective_mm'], name='Effective Rainfall', marker_color='#42A5F5'))
    fig.add_trace(go.Bar(x=recent_weeks.index, y=recent_weeks['Irrigation_needed_mm'], name='Irrigation Required', marker_color='#FFA726'))
    
    fig.update_layout(
        barmode='group',
        title='Water Balance - Last 12 Weeks',
        xaxis_title='Week',
        yaxis_title='Water (mm)',
        height=400,
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<h2 class="sub-header">Daily Irrigation Trend (Last 30 Days)</h2>', unsafe_allow_html=True)
    
    recent_30 = data.tail(30)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=recent_30.index, y=recent_30['ETc_mm_day'], name='Crop Need', line=dict(color='#EF5350', width=2), fill='tozeroy', fillcolor='rgba(239, 83, 80, 0.2)'))
    fig2.add_trace(go.Scatter(x=recent_30.index, y=recent_30['Rainfall_effective_mm'], name='Rainfall', line=dict(color='#42A5F5', width=2), fill='tozeroy', fillcolor='rgba(66, 165, 245, 0.2)'))
    fig2.add_trace(go.Scatter(x=recent_30.index, y=recent_30['Irrigation_requirement_mm'], name='Irrigation', line=dict(color='#FFA726', width=3), fill='tozeroy', fillcolor='rgba(255, 167, 38, 0.3)'))
    
    fig2.update_layout(
        title='Daily Water Balance',
        xaxis_title='Date',
        yaxis_title='Water (mm/day)',
        height=400,
        hovermode='x unified'
    )
    st.plotly_chart(fig2, use_container_width=True)

# ============================================================================
# PAGE: 7-DAY FORECAST
# ============================================================================

elif page == "7-Day Forecast":
    
    st.markdown('<h2 class="sub-header">7-Day Irrigation Forecast</h2>', unsafe_allow_html=True)
    st.info("This forecast uses machine learning trained on 5 years of Gatsibo data (R² = 0.77)")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Irrigation (7 days)", f"{forecast_7day['irrigation_mm'].sum():.1f} mm")
    with col2:
        st.metric("Daily Average", f"{forecast_7day['irrigation_mm'].mean():.1f} mm")
    with col3:
        st.metric("Peak Day", f"{forecast_7day['irrigation_mm'].max():.1f} mm")
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=forecast_7day['date'],
        y=forecast_7day['irrigation_mm'],
        marker_color='#66BB6A',
        text=forecast_7day['irrigation_mm'].round(1),
        textposition='outside'
    ))
    fig.update_layout(title='Daily Irrigation Forecast', xaxis_title='Date', yaxis_title='Irrigation (mm)', height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Detailed Forecast")
    forecast_display = forecast_7day.copy()
    forecast_display['date'] = forecast_display['date'].dt.strftime('%a, %b %d')
    forecast_display['irrigation_mm'] = forecast_display['irrigation_mm'].round(2)
    forecast_display.columns = ['Date', 'Day', 'Irrigation (mm)']
    st.dataframe(forecast_display, use_container_width=True, hide_index=True)
    
    st.markdown('<h2 class="sub-header">Scenario Analysis</h2>', unsafe_allow_html=True)
    st.info("How would irrigation needs change under different weather conditions?")
    
    scenario_col1, scenario_col2, scenario_col3 = st.columns(3)
    with scenario_col1:
        st.markdown("""
            <div class='metric-card'>
                <h4>DROUGHT Scenario</h4>
                <h2 style='color: #F44336;'>1.32 mm/day</h2>
                <p>If no rainfall for a week</p>
            </div>
        """, unsafe_allow_html=True)
    with scenario_col2:
        st.markdown("""
            <div class='metric-card'>
                <h4>NORMAL Scenario</h4>
                <h2 style='color: #FFC107;'>0.67 mm/day</h2>
                <p>Typical weather conditions</p>
            </div>
        """, unsafe_allow_html=True)
    with scenario_col3:
        st.markdown("""
            <div class='metric-card'>
                <h4>WET Scenario</h4>
                <h2 style='color: #42A5F5;'>0.66 mm/day</h2>
                <p>Heavy rainfall expected</p>
            </div>
        """, unsafe_allow_html=True)

# ============================================================================
# PAGE: HISTORICAL ANALYSIS
# ============================================================================

elif page == "Historical Analysis":
    
    st.markdown('<h2 class="sub-header">Historical Data Analysis (2019-2024)</h2>', unsafe_allow_html=True)
    
    st.markdown("### Annual Water Balance")
    annual_data = data.resample('YE').agg({
        'ETc_mm_day': 'sum',
        'Rainfall_effective_mm': 'sum',
        'Irrigation_requirement_mm': 'sum'
    })
    annual_data.index = annual_data.index.year
    annual_data.columns = ['Crop Water Need (mm)', 'Effective Rainfall (mm)', 'Irrigation Needed (mm)']
    
    fig = go.Figure()
    for col in annual_data.columns:
        fig.add_trace(go.Bar(x=annual_data.index, y=annual_data[col], name=col))
    fig.update_layout(barmode='group', title='Annual Water Balance by Year', xaxis_title='Year', yaxis_title='Water (mm/year)', height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Monthly Irrigation Patterns")
    monthly_avg = data.groupby(data.index.month).agg({
        'Irrigation_requirement_mm': 'mean',
        'Rainfall_mm': 'mean',
        'ETc_mm_day': 'mean'
    })
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_avg.index = month_names
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=month_names, y=monthly_avg['Irrigation_requirement_mm'], name='Irrigation', line=dict(color='#FFA726', width=3), fill='tozeroy'))
    fig2.add_trace(go.Scatter(x=month_names, y=monthly_avg['Rainfall_mm'], name='Rainfall', line=dict(color='#42A5F5', width=2)))
    fig2.update_layout(title='Average Daily Values by Month', xaxis_title='Month', yaxis_title='Water (mm/day)', height=400)
    st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("### Key Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Avg Annual Irrigation", f"{data['Irrigation_requirement_mm'].sum() / 5:.0f} mm/year")
    with col2:
        st.metric("Rainfall Contribution", f"{(data['Rainfall_effective_mm'].sum() / data['ETc_mm_day'].sum() * 100):.0f}%")
    with col3:
        st.metric("Days Needing Irrigation", f"{(data['Irrigation_requirement_mm'] > 0).sum() / len(data) * 100:.0f}%")
    with col4:
        st.metric("Max Daily Irrigation", f"{data['Irrigation_requirement_mm'].max():.1f} mm")

# ============================================================================
# PAGE: ABOUT GATSIBO
# ============================================================================

elif page == "About Gatsibo":
    
    st.markdown('<h2 class="sub-header">About Gatsibo District</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
            ### Location
            - **Province:** Eastern Province, Rwanda
            - **Coordinates:** 1.58°S, 30.51°E
            - **Elevation:** ~1,450 meters
            - **Area:** Focus on Gabiro irrigation scheme
            
            ### Agriculture
            - **Main Crops:** Maize, rice, vegetables
            - **Irrigation Systems:** Drip, sprinkler, furrow
            - **Climate:** Highland tropical, bimodal rainfall
            - **Rainfall:** 900-1,400 mm/year
            
            ### Water Resources
            - **Rivers:** Akagera watershed
            - **Irrigation schemes:** Gabiro, Kabarore
            - **Challenges:** Seasonal water stress, drought
        """)
    with col2:
        fig = go.Figure(go.Scattermapbox(
            lat=[-1.5789], lon=[30.5089], mode='markers',
            marker=go.scattermapbox.Marker(size=20, color='red'),
            text=['Gabiro Irrigation Scheme'], hoverinfo='text'
        ))
        fig.update_layout(
            mapbox=dict(style="open-street-map", center=dict(lat=-1.65, lon=30.55), zoom=9),
            height=500, margin={"r":0,"t":0,"l":0,"b":0}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Project Impact")
    impact_col1, impact_col2, impact_col3 = st.columns(3)
    with impact_col1:
        st.info("""
            **Water Savings**
            
            For 100 hectares:
            - ~21,700 m³ saved/year
            - 20-30% reduction in water use
            - Lower pumping costs
        """)
    with impact_col2:
        st.success("""
            **Crop Benefits**
            
            - Better yields
            - Reduced water stress
            - Optimized growth
            - Climate resilience
        """)
    with impact_col3:
        st.warning("""
            **Environmental Impact**
            
            - Watershed protection
            - Sustainable farming
            - Climate adaptation
            - Resource efficiency
        """)

# ============================================================================
# PAGE: ABOUT THIS TOOL
# ============================================================================

elif page == "About This Tool":
    
    st.markdown('<h2 class="sub-header">About This Tool</h2>', unsafe_allow_html=True)
    
    st.markdown("""
        ### Project Overview
        The **Gatsibo Smart Irrigation Scheduler** is an AI-powered tool that provides data-driven 
        irrigation recommendations for farmers in Gatsibo District, Rwanda.
        
        ### Technology Stack
        - **Satellite Data:** Sentinel-2 imagery (10m resolution)
        - **Weather Data:** NASA POWER API (daily meteorological data)
        - **ET₀ Calculation:** FAO-56 Penman-Monteith equation
        - **Crop Coefficients:** NDVI-based crop water requirements
        - **Machine Learning:** Random Forest (R² = 0.77)
        - **Web Framework:** Streamlit
        
        ### Data Sources
        - **Period:** 2019-2024 (5 years)
        - **Satellite images:** 83 cloud-free scenes
        - **Weather observations:** 2,134 days
        - **Training data:** 1,940 days
        - **Testing data:** 187 days
        
        ### Methodology
        1. **Reference ET₀:** Penman-Monteith equation using weather data
        2. **Crop Coefficient (Kc):** Derived from NDVI satellite measurements
        3. **Crop ET (ETc):** ETc = ET₀ × Kc
        4. **Effective Rainfall:** 80% of total rainfall
        5. **Irrigation Need:** ETc - Effective Rainfall
        6. **ML Forecast:** Random Forest predicts 7 days ahead
        
        ### Model Performance
        - **Accuracy:** R² = 0.77 (77% variance explained)
        - **Error:** MAE = 0.54 mm/day
        - **Top Feature:** ET₀ (42.8% importance)
        - **Validation:** Last 6 months held out for testing
        
        ### About the Developer
        - **Name:** Fabrice RUTAGARAMA
        - **Institution:** University of Rwanda
        - **Program:** MSc in Agribusiness
        - **Background:** BSc in Irrigation & Drainage Engineering
        - **Skills:** Data Analytics, GIS, Python, Machine Learning
        
        ### Contact & Feedback
        This tool is continuously being improved based on user feedback. 
        If you have suggestions or would like to collaborate, please reach out!
        
        **Email:** rutagaramafabrice7@gmail.com  
        **Phone:** +250 781 587 69  
        
        ### Acknowledgments
        - **Data:** Google Earth Engine, NASA POWER
        - **Methods:** FAO Irrigation and Drainage Paper No. 56
        - **Inspiration:** Kilimo (Argentina) and global precision agriculture initiatives
        
        ### License & Usage
        This tool is developed for research and educational purposes to support 
        sustainable agriculture in Rwanda. Free to use for non-commercial applications.
        
        ---
        
        **Built with dedication for Rwanda's agricultural future**
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        Gatsibo Smart Irrigation Scheduler | Built with Streamlit | Data: 2019-2024 | 
        Accuracy: R² = 0.77 | University of Rwanda | Fabrice RUTAGARAMA
    </div>
""", unsafe_allow_html=True)