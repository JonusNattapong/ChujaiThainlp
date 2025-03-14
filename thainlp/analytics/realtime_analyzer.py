"""
Advanced Real-time Analytics System for ThaiNLP
"""
from typing import Dict, List, Any, Optional, Union, Callable
import time
import threading
import queue
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class AnalyticsEvent:
    """Analytics event data"""
    event_type: str
    timestamp: float
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

class TimeWindow:
    """Time-based sliding window"""
    
    def __init__(self, duration: int):
        """Initialize time window
        
        Args:
            duration: Window duration in seconds
        """
        self.duration = duration
        self.events = deque()
        self._lock = threading.Lock()
        
    def add_event(self, event: AnalyticsEvent):
        """Add event to window
        
        Args:
            event: Event to add
        """
        with self._lock:
            self.events.append(event)
            self._cleanup()
            
    def get_events(self) -> List[AnalyticsEvent]:
        """Get all events in window
        
        Returns:
            List of events
        """
        with self._lock:
            self._cleanup()
            return list(self.events)
            
    def _cleanup(self):
        """Remove old events"""
        cutoff_time = time.time() - self.duration
        while self.events and self.events[0].timestamp < cutoff_time:
            self.events.popleft()

class MetricsAggregator:
    """Aggregate metrics over time windows"""
    
    def __init__(self, windows: List[int] = [60, 300, 900, 3600]):
        """Initialize aggregator
        
        Args:
            windows: List of time window durations in seconds
        """
        self.windows = {
            duration: TimeWindow(duration)
            for duration in windows
        }
        self.metrics = defaultdict(lambda: defaultdict(float))
        
    def add_metric(
        self,
        name: str,
        value: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add metric value
        
        Args:
            name: Metric name
            value: Metric value
            metadata: Additional metadata
        """
        event = AnalyticsEvent(
            event_type="metric",
            timestamp=time.time(),
            data={"name": name, "value": value},
            metadata=metadata
        )
        
        for window in self.windows.values():
            window.add_event(event)
            
    def get_stats(
        self,
        name: str,
        window_duration: int
    ) -> Dict[str, float]:
        """Get statistics for metric in time window
        
        Args:
            name: Metric name
            window_duration: Window duration in seconds
            
        Returns:
            Dictionary of statistics
        """
        if window_duration not in self.windows:
            raise ValueError(f"Invalid window duration: {window_duration}")
            
        events = [
            e for e in self.windows[window_duration].get_events()
            if e.data["name"] == name
        ]
        
        if not events:
            return {
                "count": 0,
                "sum": 0.0,
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
                "std": 0.0
            }
            
        values = [e.data["value"] for e in events]
        return {
            "count": len(values),
            "sum": sum(values),
            "mean": np.mean(values),
            "min": min(values),
            "max": max(values),
            "std": np.std(values) if len(values) > 1 else 0.0
        }

class EventProcessor:
    """Process and analyze events in real-time"""
    
    def __init__(self, window_size: int = 3600):
        """Initialize event processor
        
        Args:
            window_size: Event window size in seconds
        """
        self.window_size = window_size
        self.events = TimeWindow(window_size)
        self.event_counts = defaultdict(int)
        self.patterns = defaultdict(list)
        self._lock = threading.Lock()
        
    def add_event(
        self,
        event_type: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add event for processing
        
        Args:
            event_type: Type of event
            data: Event data
            metadata: Additional metadata
        """
        event = AnalyticsEvent(
            event_type=event_type,
            timestamp=time.time(),
            data=data,
            metadata=metadata
        )
        
        with self._lock:
            self.events.add_event(event)
            self.event_counts[event_type] += 1
            
            # Update patterns
            if event_type in self.patterns:
                for pattern in self.patterns[event_type]:
                    pattern["check_fn"](event)
                    
    def add_pattern(
        self,
        event_type: str,
        pattern_name: str,
        check_fn: Callable[[AnalyticsEvent], bool],
        callback_fn: Optional[Callable[[AnalyticsEvent], None]] = None
    ):
        """Add pattern to detect
        
        Args:
            event_type: Type of event to check
            pattern_name: Pattern name
            check_fn: Pattern detection function
            callback_fn: Callback for pattern detection
        """
        with self._lock:
            self.patterns[event_type].append({
                "name": pattern_name,
                "check_fn": check_fn,
                "callback_fn": callback_fn
            })
            
    def get_event_counts(
        self,
        start_time: Optional[float] = None
    ) -> Dict[str, int]:
        """Get event counts
        
        Args:
            start_time: Start time for counting
            
        Returns:
            Dictionary of event counts
        """
        with self._lock:
            if start_time is None:
                return dict(self.event_counts)
                
            counts = defaultdict(int)
            for event in self.events.get_events():
                if event.timestamp >= start_time:
                    counts[event.event_type] += 1
                    
            return dict(counts)
            
    def get_events_by_type(
        self,
        event_type: str,
        start_time: Optional[float] = None
    ) -> List[AnalyticsEvent]:
        """Get events of specific type
        
        Args:
            event_type: Type of events to get
            start_time: Start time for filtering
            
        Returns:
            List of events
        """
        with self._lock:
            events = self.events.get_events()
            return [
                e for e in events
                if e.event_type == event_type and
                (start_time is None or e.timestamp >= start_time)
            ]

class RealTimeAnalyzer:
    """Advanced real-time analytics system"""
    
    def __init__(
        self,
        metrics_windows: List[int] = [60, 300, 900, 3600],
        event_window: int = 3600
    ):
        """Initialize analyzer
        
        Args:
            metrics_windows: List of metrics window durations
            event_window: Event window duration
        """
        self.metrics = MetricsAggregator(metrics_windows)
        self.events = EventProcessor(event_window)
        self.alerts = []
        self.alert_callbacks = {}
        self._lock = threading.Lock()
        
        # Start monitoring thread
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self._monitor_thread.start()
        
    def track_metric(
        self,
        name: str,
        value: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Track metric value
        
        Args:
            name: Metric name
            value: Metric value
            metadata: Additional metadata
        """
        self.metrics.add_metric(name, value, metadata)
        
    def track_event(
        self,
        event_type: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Track event
        
        Args:
            event_type: Type of event
            data: Event data
            metadata: Additional metadata
        """
        self.events.add_event(event_type, data, metadata)
        
    def add_alert(
        self,
        name: str,
        condition_fn: Callable[["RealTimeAnalyzer"], bool],
        callback_fn: Optional[Callable[[], None]] = None,
        cooldown: int = 300
    ):
        """Add alert condition
        
        Args:
            name: Alert name
            condition_fn: Alert condition function
            callback_fn: Callback for alert
            cooldown: Alert cooldown period in seconds
        """
        with self._lock:
            self.alerts.append({
                "name": name,
                "condition_fn": condition_fn,
                "last_triggered": 0,
                "cooldown": cooldown
            })
            
            if callback_fn:
                self.alert_callbacks[name] = callback_fn
                
    def get_metric_stats(
        self,
        name: str,
        window: int = 300
    ) -> Dict[str, float]:
        """Get metric statistics
        
        Args:
            name: Metric name
            window: Time window in seconds
            
        Returns:
            Dictionary of statistics
        """
        return self.metrics.get_stats(name, window)
        
    def get_event_summary(
        self,
        window: int = 300
    ) -> Dict[str, Dict[str, Any]]:
        """Get event summary
        
        Args:
            window: Time window in seconds
            
        Returns:
            Dictionary of event summaries
        """
        start_time = time.time() - window
        counts = self.events.get_event_counts(start_time)
        
        summary = {}
        for event_type, count in counts.items():
            events = self.events.get_events_by_type(event_type, start_time)
            
            if events:
                latest = events[-1]
                summary[event_type] = {
                    "count": count,
                    "latest_timestamp": latest.timestamp,
                    "latest_data": latest.data
                }
                
        return summary
        
    def _check_alerts(self):
        """Check and trigger alerts"""
        current_time = time.time()
        
        with self._lock:
            for alert in self.alerts:
                if current_time - alert["last_triggered"] >= alert["cooldown"]:
                    if alert["condition_fn"](self):
                        alert["last_triggered"] = current_time
                        
                        if alert["name"] in self.alert_callbacks:
                            self.alert_callbacks[alert["name"]]()
                            
    def _monitor_loop(self):
        """Background monitoring thread"""
        while True:
            self._check_alerts()
            time.sleep(1)  # Check every second 