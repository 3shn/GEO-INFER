"""
Performance Metrics for H3 Nested Systems.

This module provides comprehensive performance analysis and metrics
for nested geospatial systems, including computational efficiency,
memory usage, and system optimization recommendations.
"""

import logging
import uuid
import time
import psutil
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from enum import Enum
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning("NumPy not available. Some performance analysis features will be limited.")


class PerformanceMetric(Enum):
    """Types of performance metrics."""
    EXECUTION_TIME = "execution_time"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    SCALABILITY = "scalability"
    EFFICIENCY = "efficiency"
    RESOURCE_UTILIZATION = "resource_utilization"


class BenchmarkType(Enum):
    """Types of benchmarks."""
    OPERATION_SPEED = "operation_speed"
    MEMORY_EFFICIENCY = "memory_efficiency"
    SCALABILITY_TEST = "scalability_test"
    STRESS_TEST = "stress_test"
    LOAD_TEST = "load_test"
    ENDURANCE_TEST = "endurance_test"


class OptimizationTarget(Enum):
    """Optimization targets."""
    SPEED = "speed"
    MEMORY = "memory"
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    BALANCED = "balanced"


@dataclass
class PerformanceMeasurement:
    """
    Represents a single performance measurement.
    """
    
    measurement_id: str
    metric_type: PerformanceMetric
    
    # Measurement values
    value: float
    unit: str
    
    # Context
    operation_name: str
    system_context: Dict[str, Any] = field(default_factory=dict)
    
    # Timing
    timestamp: datetime = field(default_factory=datetime.now)
    duration: Optional[timedelta] = None
    
    # Metadata
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert measurement to dictionary."""
        return {
            'measurement_id': self.measurement_id,
            'metric_type': self.metric_type.value,
            'value': self.value,
            'unit': self.unit,
            'operation_name': self.operation_name,
            'system_context': self.system_context,
            'timestamp': self.timestamp.isoformat(),
            'duration': self.duration.total_seconds() if self.duration else None,
            'tags': self.tags
        }


@dataclass
class BenchmarkResult:
    """
    Result of a benchmark test.
    """
    
    benchmark_id: str
    benchmark_type: BenchmarkType
    
    # Measurements
    measurements: List[PerformanceMeasurement] = field(default_factory=list)
    
    # Summary statistics
    summary_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Performance scores
    performance_score: float = 0.0
    efficiency_score: float = 0.0
    scalability_score: float = 0.0
    
    # System information
    system_info: Dict[str, Any] = field(default_factory=dict)
    
    # Test configuration
    test_config: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    duration: timedelta = field(default_factory=lambda: timedelta(0))
    
    def add_measurement(self, measurement: PerformanceMeasurement):
        """Add a measurement to the benchmark."""
        self.measurements.append(measurement)
        self._update_summary_stats()
    
    def _update_summary_stats(self):
        """Update summary statistics."""
        if not self.measurements:
            return
        
        # Group measurements by metric type
        metric_groups = defaultdict(list)
        for measurement in self.measurements:
            metric_groups[measurement.metric_type.value].append(measurement.value)
        
        # Calculate statistics for each metric
        for metric_type, values in metric_groups.items():
            if values and NUMPY_AVAILABLE:
                self.summary_stats[metric_type] = {
                    'count': len(values),
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values))
                }
            elif values:
                self.summary_stats[metric_type] = {
                    'count': len(values),
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values)
                }


@dataclass
class PerformanceProfile:
    """
    Performance profile for a system or operation.
    """
    
    profile_id: str
    target_operation: str
    
    # Performance characteristics
    baseline_metrics: Dict[PerformanceMetric, float] = field(default_factory=dict)
    current_metrics: Dict[PerformanceMetric, float] = field(default_factory=dict)
    
    # Trends
    performance_trends: Dict[PerformanceMetric, List[float]] = field(default_factory=dict)
    
    # Thresholds
    performance_thresholds: Dict[PerformanceMetric, Dict[str, float]] = field(default_factory=dict)
    
    # Optimization recommendations
    recommendations: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def update_metrics(self, metrics: Dict[PerformanceMetric, float]):
        """Update current metrics and trends."""
        for metric, value in metrics.items():
            self.current_metrics[metric] = value
            
            if metric not in self.performance_trends:
                self.performance_trends[metric] = []
            
            self.performance_trends[metric].append(value)
            
            # Keep only recent trends (last 100 measurements)
            if len(self.performance_trends[metric]) > 100:
                self.performance_trends[metric] = self.performance_trends[metric][-100:]
        
        self.last_updated = datetime.now()


class PerformanceMonitor:
    """
    Context manager for monitoring performance.
    """
    
    def __init__(self, operation_name: str, analyzer: 'H3PerformanceAnalyzer'):
        self.operation_name = operation_name
        self.analyzer = analyzer
        self.start_time = None
        self.start_memory = None
        self.start_cpu = None
    
    def __enter__(self):
        """Start monitoring."""
        self.start_time = time.time()
        
        try:
            process = psutil.Process()
            self.start_memory = process.memory_info().rss
            self.start_cpu = process.cpu_percent()
        except:
            self.start_memory = 0
            self.start_cpu = 0
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop monitoring and record measurements."""
        end_time = time.time()
        execution_time = end_time - self.start_time
        
        # Record execution time
        self.analyzer.record_measurement(
            metric_type=PerformanceMetric.EXECUTION_TIME,
            value=execution_time,
            unit="seconds",
            operation_name=self.operation_name
        )
        
        try:
            process = psutil.Process()
            end_memory = process.memory_info().rss
            end_cpu = process.cpu_percent()
            
            # Record memory usage
            memory_delta = end_memory - self.start_memory
            self.analyzer.record_measurement(
                metric_type=PerformanceMetric.MEMORY_USAGE,
                value=memory_delta,
                unit="bytes",
                operation_name=self.operation_name
            )
            
            # Record CPU usage
            cpu_usage = (self.start_cpu + end_cpu) / 2
            self.analyzer.record_measurement(
                metric_type=PerformanceMetric.CPU_USAGE,
                value=cpu_usage,
                unit="percent",
                operation_name=self.operation_name
            )
        
        except Exception as e:
            logger.warning(f"Failed to record system metrics: {e}")


class H3PerformanceAnalyzer:
    """
    Advanced performance analyzer for H3 nested systems.
    
    Provides comprehensive performance analysis including:
    - Real-time performance monitoring
    - Benchmark testing and comparison
    - Performance profiling and optimization
    - Resource utilization analysis
    """
    
    def __init__(self, name: str = "H3PerformanceAnalyzer"):
        """
        Initialize performance analyzer.
        
        Args:
            name: Analyzer name for identification
        """
        self.name = name
        
        # Measurement storage
        self.measurements: List[PerformanceMeasurement] = []
        self.benchmark_results: Dict[str, BenchmarkResult] = {}
        self.performance_profiles: Dict[str, PerformanceProfile] = {}
        
        # Real-time monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        self.monitoring_data = deque(maxlen=1000)
        
        # Configuration
        self.config: Dict[str, Any] = {
            'measurement_retention_days': 30,
            'monitoring_interval': 1.0,  # seconds
            'benchmark_iterations': 10,
            'memory_threshold_mb': 1000,
            'cpu_threshold_percent': 80
        }
        
        # Statistics
        self.analysis_stats: Dict[str, int] = defaultdict(int)
        
        # Metadata
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    def record_measurement(self, metric_type: PerformanceMetric, value: float,
                          unit: str, operation_name: str,
                          system_context: Optional[Dict[str, Any]] = None,
                          tags: Optional[Dict[str, str]] = None) -> PerformanceMeasurement:
        """
        Record a performance measurement.
        
        Args:
            metric_type: Type of metric
            value: Measurement value
            unit: Unit of measurement
            operation_name: Name of operation being measured
            system_context: System context information
            tags: Additional tags
            
        Returns:
            Created PerformanceMeasurement instance
        """
        measurement = PerformanceMeasurement(
            measurement_id=f"measurement_{uuid.uuid4().hex[:8]}",
            metric_type=metric_type,
            value=value,
            unit=unit,
            operation_name=operation_name,
            system_context=system_context or {},
            tags=tags or {}
        )
        
        self.measurements.append(measurement)
        self.analysis_stats['measurements_recorded'] += 1
        self.updated_at = datetime.now()
        
        # Clean up old measurements
        self._cleanup_old_measurements()
        
        return measurement
    
    def monitor_operation(self, operation_name: str) -> PerformanceMonitor:
        """
        Create a performance monitor for an operation.
        
        Args:
            operation_name: Name of operation to monitor
            
        Returns:
            PerformanceMonitor context manager
        """
        return PerformanceMonitor(operation_name, self)
    
    def run_benchmark(self, benchmark_type: BenchmarkType, 
                     target_function: Callable, 
                     test_config: Optional[Dict[str, Any]] = None,
                     **kwargs) -> BenchmarkResult:
        """
        Run a benchmark test.
        
        Args:
            benchmark_type: Type of benchmark
            target_function: Function to benchmark
            test_config: Test configuration
            **kwargs: Additional parameters for the function
            
        Returns:
            BenchmarkResult instance
        """
        start_time = datetime.now()
        benchmark_id = f"benchmark_{uuid.uuid4().hex[:8]}"
        
        result = BenchmarkResult(
            benchmark_id=benchmark_id,
            benchmark_type=benchmark_type,
            test_config=test_config or {},
            system_info=self._get_system_info()
        )
        
        iterations = test_config.get('iterations', self.config['benchmark_iterations']) if test_config else self.config['benchmark_iterations']
        
        # Run benchmark iterations
        for i in range(iterations):
            try:
                with self.monitor_operation(f"benchmark_{benchmark_type.value}_iter_{i}"):
                    target_function(**kwargs)
                
                # Get the latest measurements for this iteration
                recent_measurements = [m for m in self.measurements[-10:] 
                                     if m.operation_name.startswith(f"benchmark_{benchmark_type.value}_iter_{i}")]
                
                result.measurements.extend(recent_measurements)
                
            except Exception as e:
                logger.warning(f"Benchmark iteration {i} failed: {e}")
                continue
        
        # Calculate benchmark scores
        result.performance_score = self._calculate_performance_score(result)
        result.efficiency_score = self._calculate_efficiency_score(result)
        result.scalability_score = self._calculate_scalability_score(result)
        
        result.duration = datetime.now() - start_time
        
        # Store result
        self.benchmark_results[benchmark_id] = result
        self.analysis_stats['benchmarks_run'] += 1
        self.updated_at = datetime.now()
        
        return result
    
    def create_performance_profile(self, profile_id: str, target_operation: str,
                                 baseline_metrics: Optional[Dict[PerformanceMetric, float]] = None) -> PerformanceProfile:
        """
        Create a performance profile.
        
        Args:
            profile_id: Profile identifier
            target_operation: Operation being profiled
            baseline_metrics: Baseline performance metrics
            
        Returns:
            Created PerformanceProfile instance
        """
        profile = PerformanceProfile(
            profile_id=profile_id,
            target_operation=target_operation,
            baseline_metrics=baseline_metrics or {}
        )
        
        self.performance_profiles[profile_id] = profile
        self.updated_at = datetime.now()
        
        return profile
    
    def update_performance_profile(self, profile_id: str, 
                                 metrics: Dict[PerformanceMetric, float]):
        """
        Update a performance profile with new metrics.
        
        Args:
            profile_id: Profile identifier
            metrics: New performance metrics
        """
        if profile_id not in self.performance_profiles:
            raise ValueError(f"Performance profile {profile_id} not found")
        
        profile = self.performance_profiles[profile_id]
        profile.update_metrics(metrics)
        
        # Generate recommendations
        profile.recommendations = self._generate_performance_recommendations(profile)
        
        self.updated_at = datetime.now()
    
    def start_monitoring(self):
        """Start real-time performance monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info(f"Performance monitoring started for {self.name}")
    
    def stop_monitoring(self):
        """Stop real-time performance monitoring."""
        self.monitoring_active = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        logger.info(f"Performance monitoring stopped for {self.name}")
    
    def _monitoring_loop(self):
        """Real-time monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect system metrics
                system_metrics = self._collect_system_metrics()
                
                # Store monitoring data
                self.monitoring_data.append({
                    'timestamp': datetime.now(),
                    'metrics': system_metrics
                })
                
                time.sleep(self.config['monitoring_interval'])
                
            except Exception as e:
                logger.warning(f"Monitoring loop error: {e}")
                time.sleep(self.config['monitoring_interval'])
    
    def _collect_system_metrics(self) -> Dict[str, float]:
        """Collect current system metrics."""
        metrics = {}
        
        try:
            process = psutil.Process()
            
            # Memory metrics
            memory_info = process.memory_info()
            metrics['memory_rss'] = memory_info.rss
            metrics['memory_vms'] = memory_info.vms
            
            # CPU metrics
            metrics['cpu_percent'] = process.cpu_percent()
            
            # System-wide metrics
            metrics['system_cpu_percent'] = psutil.cpu_percent()
            metrics['system_memory_percent'] = psutil.virtual_memory().percent
            
        except Exception as e:
            logger.warning(f"Failed to collect system metrics: {e}")
        
        return metrics
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        info = {}
        
        try:
            info['cpu_count'] = psutil.cpu_count()
            info['memory_total'] = psutil.virtual_memory().total
            info['python_version'] = f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}"
            
        except Exception as e:
            logger.warning(f"Failed to get system info: {e}")
        
        return info
    
    def _calculate_performance_score(self, result: BenchmarkResult) -> float:
        """Calculate overall performance score."""
        if not result.measurements:
            return 0.0
        
        # Simple scoring based on execution time
        execution_times = [
            m.value for m in result.measurements 
            if m.metric_type == PerformanceMetric.EXECUTION_TIME
        ]
        
        if not execution_times:
            return 0.5
        
        avg_time = sum(execution_times) / len(execution_times)
        
        # Score inversely related to execution time
        # Assume 1 second is baseline (score = 0.5)
        score = 1.0 / (1.0 + avg_time)
        
        return min(1.0, max(0.0, score))
    
    def _calculate_efficiency_score(self, result: BenchmarkResult) -> float:
        """Calculate efficiency score."""
        if not result.measurements:
            return 0.0
        
        # Efficiency based on memory and CPU usage
        memory_measurements = [
            m.value for m in result.measurements 
            if m.metric_type == PerformanceMetric.MEMORY_USAGE
        ]
        
        cpu_measurements = [
            m.value for m in result.measurements 
            if m.metric_type == PerformanceMetric.CPU_USAGE
        ]
        
        efficiency_scores = []
        
        if memory_measurements:
            avg_memory = sum(abs(m) for m in memory_measurements) / len(memory_measurements)
            memory_score = 1.0 / (1.0 + avg_memory / (1024 * 1024))  # Normalize by MB
            efficiency_scores.append(memory_score)
        
        if cpu_measurements:
            avg_cpu = sum(cpu_measurements) / len(cpu_measurements)
            cpu_score = 1.0 - (avg_cpu / 100.0)  # Lower CPU usage is better
            efficiency_scores.append(max(0.0, cpu_score))
        
        return sum(efficiency_scores) / len(efficiency_scores) if efficiency_scores else 0.5
    
    def _calculate_scalability_score(self, result: BenchmarkResult) -> float:
        """Calculate scalability score."""
        # This would analyze how performance scales with load
        # For now, return a placeholder
        return 0.5
    
    def _generate_performance_recommendations(self, profile: PerformanceProfile) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Check memory usage trends
        if PerformanceMetric.MEMORY_USAGE in profile.performance_trends:
            memory_trend = profile.performance_trends[PerformanceMetric.MEMORY_USAGE]
            if len(memory_trend) >= 5:
                recent_memory = memory_trend[-5:]
                if all(recent_memory[i] > recent_memory[i-1] for i in range(1, len(recent_memory))):
                    recommendations.append("Memory usage is consistently increasing - check for memory leaks")
        
        # Check execution time trends
        if PerformanceMetric.EXECUTION_TIME in profile.performance_trends:
            time_trend = profile.performance_trends[PerformanceMetric.EXECUTION_TIME]
            if len(time_trend) >= 5:
                recent_times = time_trend[-5:]
                avg_time = sum(recent_times) / len(recent_times)
                if avg_time > 1.0:  # More than 1 second
                    recommendations.append("Consider optimizing algorithm or using caching for better performance")
        
        # Check CPU usage
        if PerformanceMetric.CPU_USAGE in profile.current_metrics:
            cpu_usage = profile.current_metrics[PerformanceMetric.CPU_USAGE]
            if cpu_usage > self.config['cpu_threshold_percent']:
                recommendations.append("High CPU usage detected - consider parallel processing or optimization")
        
        return recommendations
    
    def _cleanup_old_measurements(self):
        """Clean up old measurements."""
        cutoff_date = datetime.now() - timedelta(days=self.config['measurement_retention_days'])
        
        self.measurements = [
            m for m in self.measurements 
            if m.timestamp > cutoff_date
        ]
    
    def get_performance_summary(self, operation_name: Optional[str] = None,
                              time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """
        Get performance summary.
        
        Args:
            operation_name: Filter by operation name
            time_window: Time window for analysis
            
        Returns:
            Performance summary dictionary
        """
        # Filter measurements
        filtered_measurements = self.measurements
        
        if operation_name:
            filtered_measurements = [
                m for m in filtered_measurements 
                if m.operation_name == operation_name
            ]
        
        if time_window:
            cutoff_time = datetime.now() - time_window
            filtered_measurements = [
                m for m in filtered_measurements 
                if m.timestamp > cutoff_time
            ]
        
        if not filtered_measurements:
            return {}
        
        # Calculate summary statistics
        metric_summaries = defaultdict(list)
        for measurement in filtered_measurements:
            metric_summaries[measurement.metric_type.value].append(measurement.value)
        
        summary = {}
        for metric_type, values in metric_summaries.items():
            if values and NUMPY_AVAILABLE:
                summary[metric_type] = {
                    'count': len(values),
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values))
                }
            elif values:
                summary[metric_type] = {
                    'count': len(values),
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values)
                }
        
        return {
            'summary_statistics': summary,
            'total_measurements': len(filtered_measurements),
            'time_range': {
                'start': min(m.timestamp for m in filtered_measurements).isoformat(),
                'end': max(m.timestamp for m in filtered_measurements).isoformat()
            }
        }
    
    def get_analyzer_statistics(self) -> Dict[str, Any]:
        """Get analyzer statistics."""
        return {
            'analyzer_name': self.name,
            'total_measurements': len(self.measurements),
            'total_benchmarks': len(self.benchmark_results),
            'total_profiles': len(self.performance_profiles),
            'monitoring_active': self.monitoring_active,
            'analysis_stats': dict(self.analysis_stats),
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

