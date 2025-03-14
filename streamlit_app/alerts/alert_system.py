"""
Alert system module for the Streamlit application.
"""
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any


class AlertSystem:
    """
    Alert system for zone occupancy monitoring.
    """
    def __init__(
        self,
        alert_threshold: int = 3,
        alert_duration_threshold: int = 5
    ):
        """
        Initialize the alert system.
        
        Args:
            alert_threshold: Number of people in a zone to trigger an alert
            alert_duration_threshold: Duration in seconds for an alert to be considered active
        """
        self.alert_threshold = alert_threshold
        self.alert_duration_threshold = alert_duration_threshold
        self.current_alerts = {}
        self.alert_history = []
    
    def update(self, stats: Dict[int, int], timestamp: float):
        """
        Update alert status based on current statistics.
        
        Args:
            stats: Dictionary mapping zone IDs to people counts
            timestamp: Current timestamp in seconds
        """
        # Check for alerts
        for zone_id, count in stats.items():
            if count >= self.alert_threshold:
                if zone_id not in self.current_alerts:
                    self.current_alerts[zone_id] = {
                        'start_time': timestamp,
                        'count': count
                    }
                else:
                    self.current_alerts[zone_id]['count'] = max(
                        self.current_alerts[zone_id]['count'], 
                        count
                    )
            elif zone_id in self.current_alerts:
                alert_duration = timestamp - self.current_alerts[zone_id]['start_time']
                if alert_duration >= self.alert_duration_threshold:
                    self.alert_history.append({
                        'zone_id': zone_id,
                        'start_time': self.current_alerts[zone_id]['start_time'],
                        'end_time': timestamp,
                        'duration': alert_duration,
                        'max_count': self.current_alerts[zone_id]['count']
                    })
                del self.current_alerts[zone_id]
    
    def get_current_alerts(self, timestamp: float) -> Dict[int, Dict[str, Any]]:
        """
        Get current active alerts.
        
        Args:
            timestamp: Current timestamp in seconds
            
        Returns:
            Dictionary of active alerts with duration information
        """
        active_alerts = {}
        for zone_id, alert_info in self.current_alerts.items():
            alert_duration = timestamp - alert_info['start_time']
            if alert_duration >= self.alert_duration_threshold:
                active_alerts[zone_id] = {
                    'count': alert_info['count'],
                    'duration': alert_duration,
                    'start_time': alert_info['start_time']
                }
        return active_alerts
    
    def get_alert_history(self) -> List[Dict[str, Any]]:
        """
        Get the alert history.
        
        Returns:
            List of alert events
        """
        return self.alert_history
    
    def finalize(self, timestamp: float):
        """
        Finalize alerts at the end of processing.
        
        Args:
            timestamp: Final timestamp in seconds
        """
        for zone_id, alert_info in list(self.current_alerts.items()):
            alert_duration = timestamp - alert_info['start_time']
            if alert_duration >= self.alert_duration_threshold:
                self.alert_history.append({
                    'zone_id': zone_id,
                    'start_time': alert_info['start_time'],
                    'end_time': timestamp,
                    'duration': alert_duration,
                    'max_count': alert_info['count']
                })
            del self.current_alerts[zone_id]


def display_current_alerts(current_alerts: Dict[int, Dict[str, Any]], alert_threshold: int):
    """
    Display current active alerts in the Streamlit UI.
    
    Args:
        current_alerts: Dictionary of active alerts
        alert_threshold: Threshold for triggering alerts
    """
    if not current_alerts:
        return
    
    # Create alert container with red background
    st.markdown(
        """
        <div style="background-color: rgba(255, 0, 0, 0.1); padding: 10px; border: 1px solid red; border-radius: 5px;">
            <h3 style="color: red;">⚠️ Active Alerts</h3>
            <div id="alert-content">
        """, 
        unsafe_allow_html=True
    )
    
    # Add alert text for each active alert
    for zone_id, alert_info in current_alerts.items():
        st.markdown(
            f"""
            <div style="margin-bottom: 5px;">
                <strong>Zone {int(zone_id)+1}:</strong> {alert_info['count']} people 
                (threshold: {alert_threshold}) for {alert_info['duration']:.1f} seconds
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Close the container
    st.markdown("</div></div>", unsafe_allow_html=True) 