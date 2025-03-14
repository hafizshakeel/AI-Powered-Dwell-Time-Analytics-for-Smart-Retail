"""
Analytics dashboard components for the Streamlit application.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any


def display_heatmap_analysis(results_df: pd.DataFrame):
    """
    Display heatmap analysis of zone occupancy over time.
    
    Args:
        results_df: DataFrame containing zone occupancy data
    """
    st.markdown('<div class="header">Heatmap Analysis</div>', unsafe_allow_html=True)
    
    if results_df is None or results_df.empty:
        st.warning("No data available for heatmap analysis.")
        return
    
    # Get unique zones
    zones = sorted(results_df['zone_id'].unique())
    
    # Create time bins (every 5 seconds)
    max_time = results_df['timestamp'].max()
    time_bins = np.arange(0, max_time + 5, 5)
    time_labels = [f"{int(t//60)}:{int(t%60):02d}" for t in time_bins[:-1]]
    
    # Group data by time bins and zone
    results_df['time_bin'] = pd.cut(results_df['timestamp'], bins=time_bins, labels=time_labels, include_lowest=True)
    heatmap_data = results_df.groupby(['time_bin', 'zone_id'])['count'].mean().reset_index()
    
    # Pivot data for heatmap
    heatmap_pivot = heatmap_data.pivot(index='time_bin', columns='zone_id', values='count')
    heatmap_pivot.columns = [f"Zone {int(z)+1}" for z in heatmap_pivot.columns]
    
    # Create heatmap
    fig = px.imshow(
        heatmap_pivot.values,
        labels=dict(x="Zone", y="Time", color="People Count"),
        x=heatmap_pivot.columns,
        y=heatmap_pivot.index,
        color_continuous_scale="Viridis",
        aspect="auto"
    )
    
    fig.update_layout(
        title="Zone Occupancy Heatmap",
        xaxis_title="Zone",
        yaxis_title="Time",
        height=500,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Zone occupancy over time
    st.markdown('<div class="sub-header">Zone Occupancy Over Time</div>', unsafe_allow_html=True)
    
    # Create line chart for each zone
    fig = go.Figure()
    
    for zone in zones:
        zone_data = results_df[results_df['zone_id'] == zone]
        fig.add_trace(go.Scatter(
            x=zone_data['timestamp'],
            y=zone_data['count'],
            mode='lines',
            name=f'Zone {int(zone)+1}'
        ))
    
    fig.update_layout(
        title="Zone Occupancy Over Time",
        xaxis_title="Time (seconds)",
        yaxis_title="People Count",
        height=400,
        margin=dict(l=50, r=50, t=50, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Zone statistics
    st.markdown('<div class="sub-header">Zone Statistics</div>', unsafe_allow_html=True)
    
    # Calculate statistics for each zone
    zone_stats = []
    for zone in zones:
        zone_data = results_df[results_df['zone_id'] == zone]
        zone_stats.append({
            'Zone': f'Zone {int(zone)+1}',
            'Average Occupancy': round(zone_data['count'].mean(), 2),
            'Maximum Occupancy': zone_data['count'].max(),
            'Total Time with Occupancy (s)': len(zone_data[zone_data['count'] > 0]),
            'Percentage Time Occupied (%)': round(len(zone_data[zone_data['count'] > 0]) / len(zone_data) * 100, 2)
        })
    
    # Create statistics table
    stats_df = pd.DataFrame(zone_stats)
    st.dataframe(stats_df, use_container_width=True)


def display_alert_history():
    """Display alert history from session state."""
    st.markdown('<div class="header">Alert History</div>', unsafe_allow_html=True)
    
    if 'alert_history' not in st.session_state or not st.session_state.alert_history:
        st.warning("No alerts have been recorded.")
        return
    
    # Create DataFrame from alert history
    alert_df = pd.DataFrame(st.session_state.alert_history)
    
    # Format zone IDs
    alert_df['zone_id'] = alert_df['zone_id'].apply(lambda x: f"Zone {int(x)+1}")
    
    # Format times
    alert_df['start_time'] = alert_df['start_time'].apply(lambda x: str(timedelta(seconds=int(x))))
    alert_df['end_time'] = alert_df['end_time'].apply(lambda x: str(timedelta(seconds=int(x))))
    alert_df['duration'] = alert_df['duration'].round(1)
    
    # Rename columns
    alert_df = alert_df.rename(columns={
        'zone_id': 'Zone',
        'start_time': 'Start Time',
        'end_time': 'End Time',
        'duration': 'Duration (s)',
        'max_count': 'Max People'
    })
    
    # Display alert table
    st.dataframe(alert_df, use_container_width=True)
    
    # Alert statistics
    st.markdown('<div class="sub-header">Alert Statistics</div>', unsafe_allow_html=True)
    
    # Calculate statistics
    total_alerts = len(alert_df)
    avg_duration = alert_df['Duration (s)'].mean()
    max_duration = alert_df['Duration (s)'].max()
    avg_people = alert_df['Max People'].mean()
    max_people = alert_df['Max People'].max()
    
    # Create columns for statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Alerts", total_alerts)
    
    with col2:
        st.metric("Avg. Alert Duration", f"{avg_duration:.1f}s")
    
    with col3:
        st.metric("Max Alert Duration", f"{max_duration:.1f}s")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Avg. People in Alert", f"{avg_people:.1f}")
    
    with col2:
        st.metric("Max People in Alert", max_people)
    
    # Alerts by zone
    alerts_by_zone = alert_df['Zone'].value_counts().reset_index()
    alerts_by_zone.columns = ['Zone', 'Number of Alerts']
    
    fig = px.bar(
        alerts_by_zone,
        x='Zone',
        y='Number of Alerts',
        color='Number of Alerts',
        color_continuous_scale="Viridis"
    )
    
    fig.update_layout(
        title="Alerts by Zone",
        xaxis_title="Zone",
        yaxis_title="Number of Alerts",
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    st.plotly_chart(fig, use_container_width=True) 