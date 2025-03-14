"""
Advanced Anomaly Detection System for ThaiNLP
"""
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
import time
import threading
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from ..analytics.realtime_analyzer import AnalyticsEvent, TimeWindow

@dataclass
class AnomalyScore:
    """Anomaly detection score"""
    score: float
    threshold: float
    is_anomaly: bool
    details: Dict[str, Any]
    timestamp: float

class StatisticalDetector:
    """Statistical anomaly detection"""
    
    def __init__(
        self,
        window_size: int = 3600,
        threshold: float = 3.0
    ):
        """Initialize detector
        
        Args:
            window_size: Time window size in seconds
            threshold: Number of standard deviations for anomaly
        """
        self.window = TimeWindow(window_size)
        self.threshold = threshold
        self._lock = threading.Lock()
        
    def add_point(
        self,
        value: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AnomalyScore:
        """Add data point and check for anomaly
        
        Args:
            value: Data point value
            metadata: Additional metadata
            
        Returns:
            Anomaly score
        """
        event = AnalyticsEvent(
            event_type="data_point",
            timestamp=time.time(),
            data={"value": value},
            metadata=metadata
        )
        
        with self._lock:
            self.window.add_event(event)
            return self._check_anomaly(value)
            
    def _check_anomaly(self, value: float) -> AnomalyScore:
        """Check if value is anomalous
        
        Args:
            value: Value to check
            
        Returns:
            Anomaly score
        """
        events = self.window.get_events()
        if len(events) < 2:
            return AnomalyScore(
                score=0.0,
                threshold=self.threshold,
                is_anomaly=False,
                details={},
                timestamp=time.time()
            )
            
        values = [e.data["value"] for e in events[:-1]]  # Exclude current
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            z_score = 0.0
        else:
            z_score = abs(value - mean) / std
            
        return AnomalyScore(
            score=z_score,
            threshold=self.threshold,
            is_anomaly=z_score > self.threshold,
            details={
                "mean": mean,
                "std": std,
                "z_score": z_score
            },
            timestamp=time.time()
        )

class IsolationForestDetector:
    """Isolation Forest anomaly detection"""
    
    def __init__(
        self,
        window_size: int = 3600,
        contamination: float = 0.1,
        n_estimators: int = 100,
        retrain_interval: int = 300
    ):
        """Initialize detector
        
        Args:
            window_size: Time window size in seconds
            contamination: Expected proportion of anomalies
            n_estimators: Number of trees in forest
            retrain_interval: Model retraining interval
        """
        self.window = TimeWindow(window_size)
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.retrain_interval = retrain_interval
        
        self.model = None
        self.scaler = StandardScaler()
        self.last_train_time = 0
        self._lock = threading.Lock()
        
    def add_point(
        self,
        features: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> AnomalyScore:
        """Add data point and check for anomaly
        
        Args:
            features: Feature vector
            metadata: Additional metadata
            
        Returns:
            Anomaly score
        """
        event = AnalyticsEvent(
            event_type="data_point",
            timestamp=time.time(),
            data={"features": features},
            metadata=metadata
        )
        
        with self._lock:
            self.window.add_event(event)
            
            # Retrain model if needed
            if time.time() - self.last_train_time > self.retrain_interval:
                self._train_model()
                
            return self._check_anomaly(features)
            
    def _train_model(self):
        """Train Isolation Forest model"""
        events = self.window.get_events()
        if len(events) < 10:  # Need minimum samples
            return
            
        # Prepare training data
        X = np.array([e.data["features"] for e in events])
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=42
        )
        self.model.fit(X_scaled)
        
        self.last_train_time = time.time()
        
    def _check_anomaly(self, features: List[float]) -> AnomalyScore:
        """Check if point is anomalous
        
        Args:
            features: Feature vector to check
            
        Returns:
            Anomaly score
        """
        if self.model is None:
            return AnomalyScore(
                score=0.0,
                threshold=0.0,
                is_anomaly=False,
                details={},
                timestamp=time.time()
            )
            
        # Scale features
        X = np.array(features).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # Get anomaly score
        score = -self.model.score_samples(X_scaled)[0]
        threshold = -self.model.threshold_
        
        return AnomalyScore(
            score=score,
            threshold=threshold,
            is_anomaly=score > threshold,
            details={
                "raw_score": score,
                "threshold": threshold
            },
            timestamp=time.time()
        )

class PatternDetector:
    """Pattern-based anomaly detection"""
    
    def __init__(self):
        """Initialize detector"""
        self.patterns = []
        self._lock = threading.Lock()
        
    def add_pattern(
        self,
        name: str,
        check_fn: Callable[[Dict[str, Any]], Tuple[bool, float]],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add detection pattern
        
        Args:
            name: Pattern name
            check_fn: Pattern checking function
            metadata: Pattern metadata
        """
        with self._lock:
            self.patterns.append({
                "name": name,
                "check_fn": check_fn,
                "metadata": metadata or {}
            })
            
    def check_patterns(
        self,
        data: Dict[str, Any]
    ) -> List[AnomalyScore]:
        """Check data against all patterns
        
        Args:
            data: Data to check
            
        Returns:
            List of anomaly scores
        """
        scores = []
        timestamp = time.time()
        
        with self._lock:
            for pattern in self.patterns:
                try:
                    is_anomaly, score = pattern["check_fn"](data)
                    
                    scores.append(AnomalyScore(
                        score=score,
                        threshold=0.5,  # Default threshold
                        is_anomaly=is_anomaly,
                        details={
                            "pattern_name": pattern["name"],
                            "pattern_metadata": pattern["metadata"]
                        },
                        timestamp=timestamp
                    ))
                    
                except Exception as e:
                    print(f"Error checking pattern {pattern['name']}: {e}")
                    
        return scores

class AnomalyDetector:
    """Advanced anomaly detection system"""
    
    def __init__(
        self,
        statistical_window: int = 3600,
        statistical_threshold: float = 3.0,
        isolation_forest_window: int = 3600,
        isolation_forest_contamination: float = 0.1
    ):
        """Initialize detector
        
        Args:
            statistical_window: Statistical detection window
            statistical_threshold: Statistical detection threshold
            isolation_forest_window: Isolation Forest window
            isolation_forest_contamination: Expected anomaly ratio
        """
        # Initialize detectors
        self.statistical = StatisticalDetector(
            statistical_window,
            statistical_threshold
        )
        self.isolation_forest = IsolationForestDetector(
            isolation_forest_window,
            isolation_forest_contamination
        )
        self.pattern = PatternDetector()
        
        # Detection history
        self.history = TimeWindow(max(statistical_window, isolation_forest_window))
        self._lock = threading.Lock()
        
        # Callbacks
        self.anomaly_callbacks = []
        
    def add_callback(
        self,
        callback_fn: Callable[[AnomalyScore], None]
    ):
        """Add anomaly detection callback
        
        Args:
            callback_fn: Callback function
        """
        self.anomaly_callbacks.append(callback_fn)
        
    def check_anomaly(
        self,
        value: float,
        features: List[float],
        pattern_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[AnomalyScore]:
        """Check for anomalies using all detectors
        
        Args:
            value: Single value for statistical detection
            features: Feature vector for Isolation Forest
            pattern_data: Data for pattern detection
            metadata: Additional metadata
            
        Returns:
            List of anomaly scores
        """
        scores = []
        
        # Statistical detection
        score = self.statistical.add_point(value, metadata)
        if score.is_anomaly:
            scores.append(score)
            
        # Isolation Forest detection
        score = self.isolation_forest.add_point(features, metadata)
        if score.is_anomaly:
            scores.append(score)
            
        # Pattern detection
        pattern_scores = self.pattern.check_patterns(pattern_data)
        scores.extend([s for s in pattern_scores if s.is_anomaly])
        
        # Record anomalies
        if scores:
            event = AnalyticsEvent(
                event_type="anomaly",
                timestamp=time.time(),
                data={
                    "scores": scores,
                    "value": value,
                    "features": features,
                    "pattern_data": pattern_data
                },
                metadata=metadata
            )
            
            with self._lock:
                self.history.add_event(event)
                
                # Trigger callbacks
                for callback in self.anomaly_callbacks:
                    for score in scores:
                        callback(score)
                        
        return scores
        
    def get_anomaly_history(
        self,
        start_time: Optional[float] = None
    ) -> List[AnalyticsEvent]:
        """Get anomaly detection history
        
        Args:
            start_time: Start time for filtering
            
        Returns:
            List of anomaly events
        """
        with self._lock:
            events = self.history.get_events()
            if start_time is not None:
                events = [
                    e for e in events
                    if e.timestamp >= start_time
                ]
            return events
            
    def add_pattern(
        self,
        name: str,
        check_fn: Callable[[Dict[str, Any]], Tuple[bool, float]],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add pattern for detection
        
        Args:
            name: Pattern name
            check_fn: Pattern checking function
            metadata: Pattern metadata
        """
        self.pattern.add_pattern(name, check_fn, metadata) 