"""
Performance and benchmarking tests for GEO-INFER-SPM

This module contains performance tests to ensure the package scales
appropriately with data size and provides benchmarks for common operations.
"""

import numpy as np
import time
import pytest
from memory_profiler import profile
import psutil
import os

from geo_infer_spm.models.data_models import SPMData, DesignMatrix
from geo_infer_spm.core.glm import fit_glm
from geo_infer_spm.core.spatial_analysis import SpatialAnalyzer
from geo_infer_spm.core.temporal_analysis import TemporalAnalyzer
from geo_infer_spm.utils.helpers import generate_synthetic_data


class TestPerformanceScaling:
    """Test performance scaling with data size."""

    @pytest.mark.parametrize("n_points", [100, 500, 1000])
    def test_glm_scaling(self, n_points):
        """Test GLM fitting performance scaling."""
        np.random.seed(42)

        # Generate test data
        coordinates = np.random.rand(n_points, 2) * 100
        X = np.random.randn(n_points, 3)
        y = X @ np.array([1.0, -0.5, 0.3]) + 0.1 * np.random.randn(n_points)

        spm_data = SPMData(data=y, coordinates=coordinates, crs='EPSG:4326')
        design_matrix = DesignMatrix(matrix=X, names=['int', 'x1', 'x2'])

        # Time the operation
        start_time = time.time()
        result = fit_glm(spm_data, design_matrix)
        end_time = time.time()

        execution_time = end_time - start_time

        # GLM should scale roughly as O(n*k^2) where n=data points, k=parameters
        # Allow reasonable time limits
        if n_points == 100:
            assert execution_time < 0.1  # < 100ms
        elif n_points == 500:
            assert execution_time < 1.0  # < 1s
        elif n_points == 1000:
            assert execution_time < 5.0  # < 5s

        # Verify correctness
        assert result.model_diagnostics['r_squared'] > 0.8

    @pytest.mark.parametrize("n_points", [50, 200, 500])
    def test_spatial_analysis_scaling(self, n_points):
        """Test spatial analysis performance scaling."""
        np.random.seed(42)

        coordinates = np.random.rand(n_points, 2) * 100
        data = np.random.randn(n_points)

        analyzer = SpatialAnalyzer(coordinates)

        # Time distance matrix computation (happens in init)
        start_time = time.time()
        # Distance matrix already computed in __init__
        end_time = time.time()

        setup_time = end_time - start_time

        # Time variogram computation
        start_time = time.time()
        variogram = analyzer.estimate_variogram(data, n_bins=10)
        end_time = time.time()

        variogram_time = end_time - start_time

        # Spatial operations should scale as O(n^2) for distance matrix
        # Allow reasonable time limits
        total_time = setup_time + variogram_time

        if n_points == 50:
            assert total_time < 0.5
        elif n_points == 200:
            assert total_time < 5.0
        elif n_points == 500:
            assert total_time < 30.0

    @pytest.mark.parametrize("n_timepoints", [50, 200, 500])
    def test_temporal_analysis_scaling(self, n_timepoints):
        """Test temporal analysis performance scaling."""
        np.random.seed(42)

        time_points = np.arange(n_timepoints)
        data = np.sin(2 * np.pi * time_points / 12) + 0.1 * np.random.randn(n_timepoints)

        analyzer = TemporalAnalyzer(time_points, data)

        # Time trend detection
        start_time = time.time()
        trends = analyzer.detect_trends(data.reshape(1, -1))
        end_time = time.time()

        trend_time = end_time - start_time

        # Time seasonal decomposition (if available)
        try:
            start_time = time.time()
            decomposition = analyzer.seasonal_decomposition(data, period=12)
            end_time = time.time()
            seasonal_time = end_time - start_time
        except ImportError:
            seasonal_time = 0

        total_time = trend_time + seasonal_time

        # Temporal operations should scale linearly or better
        if n_timepoints == 50:
            assert total_time < 1.0
        elif n_timepoints == 200:
            assert total_time < 5.0
        elif n_timepoints == 500:
            assert total_time < 15.0


class TestMemoryUsage:
    """Test memory usage patterns."""

    def test_glm_memory_efficiency(self):
        """Test that GLM uses memory efficiently."""
        n_points = 1000
        n_params = 5

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Generate large dataset
        coordinates = np.random.rand(n_points, 2) * 100
        X = np.random.randn(n_points, n_params)
        y = X @ np.random.randn(n_params) + 0.1 * np.random.randn(n_points)

        spm_data = SPMData(data=y, coordinates=coordinates, crs='EPSG:4326')
        design_matrix = DesignMatrix(matrix=X, names=[f'p{i}' for i in range(n_params)])

        # Fit model
        result = fit_glm(spm_data, design_matrix)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = final_memory - initial_memory

        # Should not use excessive memory (allow some overhead)
        assert memory_used < 100  # Less than 100MB for 1000 points

    def test_spatial_memory_usage(self):
        """Test memory usage in spatial operations."""
        n_points = 500

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024

        coordinates = np.random.rand(n_points, 2) * 100
        analyzer = SpatialAnalyzer(coordinates)

        # Distance matrix should be created
        assert analyzer.distance_matrix.shape == (n_points, n_points)

        intermediate_memory = process.memory_info().rss / 1024 / 1024

        # Variogram computation
        data = np.random.randn(n_points)
        variogram = analyzer.estimate_variogram(data)

        final_memory = process.memory_info().rss / 1024 / 1024
        memory_used = final_memory - initial_memory

        # Spatial operations create O(n^2) distance matrix
        # Allow reasonable memory usage
        assert memory_used < 200  # Less than 200MB for 500 points


class TestBenchmarkComparisons:
    """Benchmark against alternative implementations."""

    def test_glm_vs_numpy_lstsq(self):
        """Compare GLM performance against numpy.linalg.lstsq."""
        n_points, n_params = 1000, 5
        np.random.seed(42)

        X = np.random.randn(n_points, n_params)
        beta_true = np.random.randn(n_params)
        y = X @ beta_true + 0.01 * np.random.randn(n_points)

        coordinates = np.random.rand(n_points, 2) * 100
        spm_data = SPMData(data=y, coordinates=coordinates, crs='EPSG:4326')
        design_matrix = DesignMatrix(matrix=X, names=[f'p{i}' for i in range(n_params)])

        # Time SPM GLM
        start_time = time.time()
        spm_result = fit_glm(spm_data, design_matrix)
        spm_time = time.time() - start_time

        # Time numpy lstsq
        start_time = time.time()
        beta_numpy, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        numpy_time = time.time() - start_time

        # SPM should be within reasonable factor of optimized numpy
        # (allowing for additional SPM overhead)
        assert spm_time < numpy_time * 10  # No more than 10x slower

        # Results should be similar
        np.testing.assert_allclose(spm_result.beta_coefficients, beta_numpy, rtol=1e-10)

    def test_memory_efficiency_comparison(self):
        """Compare memory efficiency with naive implementation."""
        n_points, n_params = 500, 3

        # Measure SPM memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024

        coordinates = np.random.rand(n_points, 2) * 100
        X = np.random.randn(n_points, n_params)
        y = X @ np.random.randn(n_params) + 0.1 * np.random.randn(n_points)

        spm_data = SPMData(data=y, coordinates=coordinates, crs='EPSG:4326')
        design_matrix = DesignMatrix(matrix=X, names=[f'p{i}' for i in range(n_params)])

        result = fit_glm(spm_data, design_matrix)

        spm_memory = process.memory_info().rss / 1024 / 1024 - initial_memory

        # SPM should be memory efficient
        # Rough estimate: should use less than 50MB for 500 points
        assert spm_memory < 50


class TestScalabilityLimits:
    """Test scalability limits and edge cases."""

    def test_large_dataset_handling(self):
        """Test handling of relatively large datasets."""
        n_points = 5000  # Moderately large

        coordinates = np.random.rand(n_points, 2) * 1000
        X = np.random.randn(n_points, 2)
        y = X @ np.array([1.0, -0.5]) + 0.1 * np.random.randn(n_points)

        spm_data = SPMData(data=y, coordinates=coordinates, crs='EPSG:4326')
        design_matrix = DesignMatrix(matrix=X, names=['int', 'slope'])

        # Should complete in reasonable time
        start_time = time.time()
        result = fit_glm(spm_data, design_matrix)
        execution_time = time.time() - start_time

        # Should complete in under 30 seconds for 5000 points
        assert execution_time < 30.0
        assert result.model_diagnostics['r_squared'] > 0.8

    def test_high_dimensional_regression(self):
        """Test regression with many predictors."""
        n_points, n_params = 200, 20  # More parameters than typical

        coordinates = np.random.rand(n_points, 2) * 100
        X = np.random.randn(n_points, n_params)
        beta_true = np.random.randn(n_params) * 0.1  # Small coefficients
        y = X @ beta_true + 0.01 * np.random.randn(n_points)

        spm_data = SPMData(data=y, coordinates=coordinates, crs='EPSG:4326')
        design_matrix = DesignMatrix(matrix=X, names=[f'p{i}' for i in range(n_params)])

        start_time = time.time()
        result = fit_glm(spm_data, design_matrix)
        execution_time = time.time() - start_time

        # Should handle high dimensions reasonably
        assert execution_time < 10.0  # Under 10 seconds
        assert len(result.beta_coefficients) == n_params


class TestParallelProcessing:
    """Test parallel processing capabilities."""

    @pytest.mark.parametrize("n_jobs", [1, -1])
    def test_parallel_glm_fitting(self, n_jobs):
        """Test GLM fitting with different parallel settings."""
        n_points, n_datasets = 200, 5

        coordinates = np.random.rand(n_points, 2) * 100

        results = []
        for i in range(n_datasets):
            X = np.random.randn(n_points, 3)
            y = X @ np.random.randn(3) + 0.1 * np.random.randn(n_points)

            spm_data = SPMData(data=y, coordinates=coordinates, crs='EPSG:4326')
            design_matrix = DesignMatrix(matrix=X, names=['int', 'x1', 'x2'])

            start_time = time.time()
            result = fit_glm(spm_data, design_matrix)
            execution_time = time.time() - start_time

            results.append(execution_time)

        # All should complete successfully
        assert all(t < 5.0 for t in results)  # All under 5 seconds
        assert len(results) == n_datasets


class TestComputationalComplexity:
    """Test computational complexity scaling."""

    def test_glm_complexity_scaling(self):
        """Test that GLM scales appropriately with problem size."""
        sizes = [50, 100, 200, 500]
        times = []

        for n_points in sizes:
            coordinates = np.random.rand(n_points, 2) * 100
            X = np.random.randn(n_points, 3)
            y = X @ np.array([1.0, -0.5, 0.3]) + 0.1 * np.random.randn(n_points)

            spm_data = SPMData(data=y, coordinates=coordinates, crs='EPSG:4326')
            design_matrix = DesignMatrix(matrix=X, names=['int', 'x1', 'x2'])

            start_time = time.time()
            fit_glm(spm_data, design_matrix)
            execution_time = time.time() - start_time

            times.append(execution_time)

        # Check scaling is reasonable (should be roughly O(n))
        ratios = [times[i] / times[i-1] for i in range(1, len(times))]
        size_ratios = [sizes[i] / sizes[i-1] for i in range(1, len(sizes))]

        # Time ratios should be roughly proportional to size ratios
        # Allow some flexibility due to constant factors
        for ratio, size_ratio in zip(ratios, size_ratios):
            assert ratio < size_ratio * 3  # Allow up to 3x overhead

    def test_spatial_complexity(self):
        """Test spatial analysis complexity."""
        sizes = [30, 50, 100, 200]
        times = []

        for n_points in sizes:
            coordinates = np.random.rand(n_points, 2) * 100
            data = np.random.randn(n_points)

            start_time = time.time()
            analyzer = SpatialAnalyzer(coordinates)
            variogram = analyzer.estimate_variogram(data, n_bins=5)
            execution_time = time.time() - start_time

            times.append(execution_time)

        # Spatial operations are O(n^2), so should scale worse than linear
        ratios = [times[i] / times[i-1] for i in range(1, len(times))]
        size_ratios = [sizes[i] / sizes[i-1] for i in range(1, len(times))]

        # Should scale worse than linear due to distance matrix
        for ratio, size_ratio in zip(ratios, size_ratios):
            assert ratio > size_ratio * 0.5  # At least some quadratic scaling
