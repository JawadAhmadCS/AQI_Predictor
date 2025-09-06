import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import calendar

# Page Configuration
st.set_page_config(
    page_title="AQI Predictor - Islamabad",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    
    .aqi-excellent { border-left-color: #00e400; }
    .aqi-good { border-left-color: #ffff00; }
    .aqi-moderate { border-left-color: #ff7e00; }
    .aqi-unhealthy-sensitive { border-left-color: #ff0000; }
    .aqi-unhealthy { border-left-color: #8f3f97; }
    .aqi-hazardous { border-left-color: #7e0023; }
    
    .prediction-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        text-align: center;
    }
    
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        border-top: 1px solid #eee;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

def get_aqi_category(aqi_value):
    """Return AQI category and color based on value"""
    if aqi_value <= 50:
        return "Excellent", "#00e400", "aqi-excellent"
    elif aqi_value <= 100:
        return "Good", "#ffff00", "aqi-good"
    elif aqi_value <= 150:
        return "Moderate", "#ff7e00", "aqi-moderate"
    elif aqi_value <= 200:
        return "Unhealthy for Sensitive Groups", "#ff0000", "aqi-unhealthy-sensitive"
    elif aqi_value <= 300:
        return "Unhealthy", "#8f3f97", "aqi-unhealthy"
    else:
        return "Hazardous", "#7e0023", "aqi-hazardous"

def get_health_recommendation(aqi_value):
    """Return health recommendations based on AQI value"""
    if aqi_value <= 50:
        return "🌟 Air quality is excellent. Perfect for all outdoor activities!"
    elif aqi_value <= 100:
        return "😊 Air quality is good. Enjoy your outdoor activities!"
    elif aqi_value <= 150:
        return "⚠️ Air quality is moderate. Sensitive people should limit prolonged outdoor activities."
    elif aqi_value <= 200:
        return "🚨 Unhealthy for sensitive groups. Consider reducing outdoor activities."
    elif aqi_value <= 300:
        return "❌ Air quality is unhealthy. Avoid outdoor activities."
    else:
        return "☠️ Air quality is hazardous. Stay indoors and use air purifiers."

# Main Header
st.markdown("""
<div class="main-header">
    <h1>Air Quality Index (AQI) Prediction Dashboard</h1>
    <h3>📍 Islamabad, Pakistan</h3>
    <p>Real-time air quality monitoring and 3-day predictions</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📊 Dashboard Controls")
    
    # Data refresh button
    if st.button("🔄 Refresh Data", type="primary"):
        st.rerun()
    
    st.markdown("---")
    
    # Information section
    st.markdown("""
    ### 📖 About AQI Scale
    - **0-50**: Excellent 🌟
    - **51-100**: Good 😊
    - **101-150**: Moderate ⚠️
    - **151-200**: Unhealthy for Sensitive 🚨
    - **201-300**: Unhealthy ❌
    - **301+**: Hazardous ☠️
    """)
    
    st.markdown("---")
    st.markdown("""
    ### 📡 Data Sources
    - Weather API Integration
    - Real-time AQI Monitoring
    - Machine Learning Predictions
    """)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Load and display current weather data
    weather_file = "data/weather_init.csv"
    if os.path.exists(weather_file):
        try:
            df_weather = pd.read_csv(weather_file)
            latest = df_weather.iloc[-1]
            
            # Current AQI Status
            current_aqi = float(latest['aqius'])
            category, color, css_class = get_aqi_category(current_aqi)
            
            st.markdown("## 🌡️ Current Air Quality Status")
            
            # Create metrics display
            metric_cols = st.columns(4)
            
            with metric_cols[0]:
                st.metric(
                    label="🌫️ AQI Value",
                    value=f"{current_aqi:.0f}",
                    delta=None
                )
            
            with metric_cols[1]:
                st.metric(
                    label="🌡️ Temperature",
                    value=f"{latest['temperature']:.1f}°C",
                    delta=None
                )
            
            with metric_cols[2]:
                st.metric(
                    label="💧 Humidity",
                    value=f"{latest['humidity']:.1f}%",
                    delta=None
                )
            
            with metric_cols[3]:
                st.metric(
                    label="💨 Wind Speed",
                    value=f"{latest['wind_speed']:.1f} m/s",
                    delta=None
                )
            
            # AQI Status Card
            st.markdown(f"""
            <div class="metric-card {css_class}">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <div>
                        <h3 style="margin: 0; color: {color};">
                            <span class="status-indicator" style="background-color: {color};"></span>
                            {category}
                        </h3>
                        <p style="margin: 5px 0 0 0; font-size: 0.9em; color: #666;">
                            Last Updated: {latest['timestamp']}
                        </p>
                    </div>
                    <div style="font-size: 2.5em; font-weight: bold; color: {color};">
                        {current_aqi:.0f}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error loading weather data: {str(e)}")
    else:
        st.error("❌ Weather data file not found. Please ensure the data pipeline is running.")

with col2:
    # Health Recommendations
    if 'current_aqi' in locals():
        st.markdown("## 🏥 Health Recommendations")
        
        recommendation = get_health_recommendation(current_aqi)
        
        # Color-coded recommendation box
        _, rec_color, _ = get_aqi_category(current_aqi)
        
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {rec_color}20, {rec_color}10);
            border: 2px solid {rec_color};
            border-radius: 10px;
            padding: 1.5rem;
            margin: 1rem 0;
        ">
            <p style="margin: 0; font-weight: 500; line-height: 1.5;">
                {recommendation}
            </p>
        </div>
        """, unsafe_allow_html=True)

# AQI Trend Analysis
st.markdown("## 📈 AQI Trend Analysis")

feature_file = "data/features.csv"
if os.path.exists(feature_file):
    try:
        df_feat = pd.read_csv(feature_file)
        df_feat["timestamp"] = pd.to_datetime(df_feat["timestamp"])
        
        # Create enhanced trend chart
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('AQI Trend Over Time', 'Weather Parameters'),
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3]
        )
        
        # AQI trend line
        fig.add_trace(
            go.Scatter(
                x=df_feat["timestamp"],
                y=df_feat["aqius"],
                mode='lines+markers',
                name='AQI',
                line=dict(color='#667eea', width=3),
                marker=dict(size=6, color='#667eea'),
                hovertemplate='<b>%{y:.1f} AQI</b><br>%{x}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add AQI category background colors
        fig.add_hrect(y0=0, y1=50, fillcolor="green", opacity=0.1, row=1, col=1)
        fig.add_hrect(y0=50, y1=100, fillcolor="yellow", opacity=0.1, row=1, col=1)
        fig.add_hrect(y0=100, y1=150, fillcolor="orange", opacity=0.1, row=1, col=1)
        fig.add_hrect(y0=150, y1=200, fillcolor="red", opacity=0.1, row=1, col=1)
        fig.add_hrect(y0=200, y1=300, fillcolor="purple", opacity=0.1, row=1, col=1)
        
        # Weather parameters
        if 'temperature' in df_feat.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_feat["timestamp"],
                    y=df_feat["temperature"],
                    mode='lines',
                    name='Temperature (°C)',
                    line=dict(color='#ff6b6b', width=2),
                    yaxis='y3'
                ),
                row=2, col=1
            )
        
        if 'humidity' in df_feat.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_feat["timestamp"],
                    y=df_feat["humidity"],
                    mode='lines',
                    name='Humidity (%)',
                    line=dict(color='#4ecdc4', width=2),
                    yaxis='y4'
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            height=600,
            title_text="Comprehensive Air Quality Analysis",
            showlegend=True,
            template="plotly_white",
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="AQI", row=1, col=1)
        fig.update_yaxes(title_text="Value", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Average AQI", f"{df_feat['aqius'].mean():.1f}")
        with col2:
            st.metric("Max AQI", f"{df_feat['aqius'].max():.1f}")
        with col3:
            st.metric("Min AQI", f"{df_feat['aqius'].min():.1f}")
        with col4:
            st.metric("Data Points", len(df_feat))
            
    except Exception as e:
        st.error(f"Error loading trend data: {str(e)}")
else:
    st.warning("⚠️ Historical feature data not available yet. Please wait for data collection to begin.")

# Prediction Section
st.markdown("## 🔮 AQI Predictions (Next 3 Days)")

model_path = "models/aqi_model.pkl"
if os.path.exists(model_path) and os.path.exists(feature_file):
    try:
        # Load model and data
        model = joblib.load(model_path)
        df_feat_clean = pd.read_csv(feature_file).dropna()
        
        if len(df_feat_clean) > 0:
            last_row = df_feat_clean.iloc[-1:]
            
            # Prepare features for prediction - using exact same columns as your training
            feature_columns = ["hour", "day", "month", "weekday", "aqius", "aqius_change"]
            available_features = [col for col in feature_columns if col in last_row.columns]
            
            if len(available_features) == len(feature_columns):
                X = last_row[feature_columns].copy()
                
                predictions = []
                timestamps = []
                current_time = pd.to_datetime(last_row['timestamp'].iloc[0])
                
                # Generate predictions for 72 hours (3 days) - using your exact logic
                for i in range(72):
                    pred = model.predict(X)[0]
                    predictions.append(max(0, round(pred, 2)))  # Ensure non-negative AQI, round to 2 decimal places
                    
                    # Calculate future timestamp
                    future_time = current_time + timedelta(hours=i+1)
                    timestamps.append(future_time)
                    
                    # Update values for next hour - using your exact logic
                    current_hour = int(X["hour"].values[0])
                    current_day = int(X["day"].values[0])
                    current_month = int(X["month"].values[0])
                    current_weekday = int(X["weekday"].values[0])
                    current_aqius = float(X["aqius"].values[0])
                    
                    # Update hour
                    next_hour = (current_hour + 1) % 24
                    X.loc[X.index[0], "hour"] = next_hour
                    
                    # If 23 -> 0, then day also +1
                    if next_hour == 0:
                        # Increase day
                        days_in_month = calendar.monthrange(2025, current_month)[1]  # Assuming 2025
                        next_day = current_day + 1
                        next_weekday = (current_weekday + 1) % 7
                        
                        # If crossing last day of month
                        if next_day > days_in_month:
                            next_day = 1
                            next_month = current_month + 1
                            if next_month > 12:
                                next_month = 1
                        else:
                            next_month = current_month
                            
                        X.loc[X.index[0], "day"] = next_day
                        X.loc[X.index[0], "month"] = next_month
                        X.loc[X.index[0], "weekday"] = next_weekday
                    
                    # Update AQI values
                    X.loc[X.index[0], "aqius_change"] = pred - current_aqius
                    X.loc[X.index[0], "aqius"] = pred
                
                # Create prediction dataframe
                pred_df = pd.DataFrame({
                    "Timestamp": timestamps,
                    "Predicted AQI": predictions
                })
                
                # Display predictions in expandable section
                with st.expander("📊 View Detailed Predictions (72 hours)", expanded=False):
                    st.dataframe(
                        pred_df,
                        use_container_width=True,
                        column_config={
                            "Timestamp": st.column_config.DatetimeColumn(
                                "Date & Time",
                                format="DD/MM/YYYY HH:mm"
                            ),
                            "Predicted AQI": st.column_config.NumberColumn(
                                "AQI Value",
                                format="%.2f"
                            )
                        }
                    )
                
                # Create prediction trend chart
                fig_pred = px.line(
                    pred_df,
                    x="Timestamp",
                    y="Predicted AQI",
                    title="AQI Predictions for Next 3 Days (72 Hours)",
                    line_shape="linear",
                    markers=True
                )
                
                fig_pred.update_traces(
                    line=dict(color='#ff6b6b', width=3),
                    marker=dict(size=4, color='#ff6b6b')
                )
                
                # Add AQI category background colors to prediction chart
                fig_pred.add_hrect(y0=0, y1=50, fillcolor="green", opacity=0.1, annotation_text="Excellent")
                fig_pred.add_hrect(y0=50, y1=100, fillcolor="yellow", opacity=0.1, annotation_text="Good")
                fig_pred.add_hrect(y0=100, y1=150, fillcolor="orange", opacity=0.1, annotation_text="Moderate")
                fig_pred.add_hrect(y0=150, y1=200, fillcolor="red", opacity=0.1, annotation_text="Unhealthy for Sensitive")
                fig_pred.add_hrect(y0=200, y1=300, fillcolor="purple", opacity=0.1, annotation_text="Unhealthy")
                fig_pred.add_hrect(y0=300, y1=500, fillcolor="maroon", opacity=0.1, annotation_text="Hazardous")
                
                fig_pred.update_layout(
                    template="plotly_white",
                    height=500,
                    yaxis_title="AQI Value",
                    xaxis_title="Time",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_pred, use_container_width=True)
                
                # Combined historical and predicted data visualization
                st.markdown("### 📈 Historical vs Predicted AQI")
                
                # Get recent historical data (last 24 hours)
                recent_historical = df_feat_clean.tail(24)[['timestamp', 'aqius']].copy()
                recent_historical['timestamp'] = pd.to_datetime(recent_historical['timestamp'])
                recent_historical['type'] = 'Historical'
                recent_historical = recent_historical.rename(columns={'aqius': 'aqi'})
                
                # Prepare prediction data
                pred_display_df = pred_df.copy()
                pred_display_df['type'] = 'Predicted'
                pred_display_df = pred_display_df.rename(columns={'Predicted AQI': 'aqi'})
                
                # Combine data
                combined_df = pd.concat([recent_historical, pred_display_df], ignore_index=True)
                
                fig_combined = px.line(
                    combined_df,
                    x='Timestamp',
                    y='aqi',
                    color='type',
                    title='AQI Trend: Last 24 Hours (Historical) + Next 72 Hours (Predicted)',
                    color_discrete_map={'Historical': '#667eea', 'Predicted': '#ff6b6b'}
                )
                
                fig_combined.update_traces(line_width=3, marker_size=6)
                fig_combined.update_layout(
                    template="plotly_white",
                    height=400,
                    yaxis_title="AQI Value",
                    xaxis_title="Time",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_combined, use_container_width=True)
                
                # Prediction analysis and statistics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Next Hour AQI", f"{predictions[0]:.2f}")
                with col2:
                    st.metric("24hr Average", f"{np.mean(predictions[:24]):.2f}")
                with col3:
                    st.metric("72hr Average", f"{np.mean(predictions):.2f}")
                with col4:
                    st.metric("Max Predicted", f"{max(predictions):.2f}")
                
                # Trend analysis
                short_term_trend = predictions[5] - predictions[0]  # Next 6 hours
                long_term_trend = predictions[-1] - predictions[0]  # 72 hours
                
                if abs(short_term_trend) < 5:
                    short_trend_text = "stable"
                    short_trend_emoji = "➡️"
                elif short_term_trend > 0:
                    short_trend_text = "worsening"
                    short_trend_emoji = "📈"
                else:
                    short_trend_text = "improving"
                    short_trend_emoji = "📉"
                
                if abs(long_term_trend) < 10:
                    long_trend_text = "stable"
                    long_trend_emoji = "➡️"
                elif long_term_trend > 0:
                    long_trend_text = "worsening"
                    long_trend_emoji = "📈"
                else:
                    long_trend_text = "improving"
                    long_trend_emoji = "📉"
                
                st.markdown("### 📊 Trend Analysis")
                
                trend_col1, trend_col2 = st.columns(2)
                
                with trend_col1:
                    st.info(f"{short_trend_emoji} **Short-term (6 hours)**: Air quality is expected to be **{short_trend_text}** (Change: {short_term_trend:+.2f})")
                
                with trend_col2:
                    st.info(f"{long_trend_emoji} **Long-term (3 days)**: Overall trend is **{long_trend_text}** (Change: {long_term_trend:+.2f})")
                
            else:
                st.warning("⚠️ Insufficient feature data for predictions. Missing columns: " + 
                         str(set(feature_columns) - set(available_features)))
        else:
            st.warning("⚠️ No clean data available for predictions.")
            
    except Exception as e:
        st.error(f"❌ Error generating predictions: {str(e)}")
        st.info("💡 This might be due to missing model features or corrupted model file.")
        
        # Debug information
        with st.expander("🔍 Debug Information"):
            st.text(f"Error details: {str(e)}")
            st.text(f"Model path exists: {os.path.exists(model_path)}")
            st.text(f"Feature file exists: {os.path.exists(feature_file)}")
else:
    missing_items = []
    if not os.path.exists(model_path):
        missing_items.append("trained model")
    if not os.path.exists(feature_file):
        missing_items.append("feature data")
    
    st.warning(f"⚠️ Predictions not available. Missing: {', '.join(missing_items)}")
    
    if st.button("🔄 Check Again"):
        st.rerun()

# Footer
st.markdown("""
<div class="footer">
    <p><strong>AQI Prediction Dashboard</strong> | Powered by JawadAhmadCS</p>
    <p>📍 Islamabad, Pakistan | Real-time Air Quality Monitoring</p>
    <p><small>Data updates every hour • Predictions based on historical patterns and weather conditions</small></p>
</div>
""", unsafe_allow_html=True)

# Add system status in sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🔧 System Status")
    
    # Check file statuses
    weather_status = "✅ Online" if os.path.exists(weather_file) else "❌ Offline"
    feature_status = "✅ Online" if os.path.exists(feature_file) else "❌ Offline"
    model_status = "✅ Ready" if os.path.exists(model_path) else "❌ Not Ready"
    
    st.text(f"Weather Data: {weather_status}")
    st.text(f"Feature Data: {feature_status}")
    st.text(f"ML Model: {model_status}")
    
    st.markdown(f"**Last Update:** {datetime.now().strftime('%H:%M:%S')}")
