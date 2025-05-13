"""
Base class for Bayesian models in the GEO-INFER-BAYES framework.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
from abc import ABC, abstractmethod


class BayesianModel(ABC):
    """
    Abstract base class for all Bayesian models.
    
    This class defines the interface that all Bayesian models
    must implement to be compatible with the inference framework.
    
    Parameters
    ----------
    name : str
        Name of the model
    **kwargs : dict
        Additional model-specific parameters
    """
    
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.parameters = {}
        self.priors = {}
        self._setup_model(**kwargs)
    
    @abstractmethod
    def _setup_model(self, **kwargs) -> None:
        """
        Set up the model structure and parameters.
        
        This method should be implemented by subclasses to define
        the specific model structure, parameters, and priors.
        
        Parameters
        ----------
        **kwargs : dict
            Model-specific parameters
        """
        pass
    
    @abstractmethod
    def log_likelihood(self, theta: Dict[str, Any], data: Any) -> float:
        """
        Compute the log-likelihood for the model.
        
        Parameters
        ----------
        theta : dict
            Dictionary of parameter values
        data : any
            Data to compute the likelihood for
            
        Returns
        -------
        float
            Log-likelihood value
        """
        pass
    
    @abstractmethod
    def log_prior(self, theta: Dict[str, Any]) -> float:
        """
        Compute the log-prior for the model parameters.
        
        Parameters
        ----------
        theta : dict
            Dictionary of parameter values
            
        Returns
        -------
        float
            Log-prior value
        """
        pass
    
    def log_posterior(self, theta: Dict[str, Any], data: Any) -> float:
        """
        Compute the log-posterior for the model.
        
        Parameters
        ----------
        theta : dict
            Dictionary of parameter values
        data : any
            Data to compute the posterior for
            
        Returns
        -------
        float
            Log-posterior value
        """
        return self.log_likelihood(theta, data) + self.log_prior(theta)
    
    def prepare_data(self, data: Union[np.ndarray, xr.Dataset, Dict[str, Any]]) -> Any:
        """
        Prepare data for inference.
        
        Parameters
        ----------
        data : array-like, Dataset, or dict
            Raw data to prepare
            
        Returns
        -------
        any
            Prepared data in the format expected by the model
        """
        # Default implementation - override for specific data preprocessing
        return data
    
    @abstractmethod
    def predict(
        self, 
        X_new: np.ndarray, 
        posterior: Any = None, 
        samples: int = 100,
        return_std: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions at new locations.
        
        Parameters
        ----------
        X_new : array-like
            New locations to predict at
        posterior : PosteriorAnalysis, optional
            Posterior analysis object. If None, use prior predictive.
        samples : int, default=100
            Number of posterior samples to use
        return_std : bool, default=False
            Whether to return standard deviations
            
        Returns
        -------
        y_pred : ndarray
            Predictions
        y_std : ndarray, optional
            Standard deviations of predictions
        """
        pass
    
    @abstractmethod
    def posterior_predictive(
        self, 
        posterior: Any,
        X: Optional[np.ndarray] = None,
        samples: int = 100
    ) -> np.ndarray:
        """
        Generate posterior predictive samples.
        
        Parameters
        ----------
        posterior : PosteriorAnalysis
            Posterior analysis object
        X : array-like, optional
            Locations to generate predictions for. If None, use observed locations.
        samples : int, default=100
            Number of posterior samples to use
            
        Returns
        -------
        ndarray
            Posterior predictive samples
        """
        pass
    
    def plot_prediction(
        self, 
        posterior: Any,
        grid: Optional[np.ndarray] = None,
        uncertainty: bool = True,
        **kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot model predictions from the posterior.
        
        Parameters
        ----------
        posterior : PosteriorAnalysis
            Posterior analysis object
        grid : array-like, optional
            Grid points to predict on. If None, use a default grid
        uncertainty : bool, default=True
            Whether to plot uncertainty bounds
        **kwargs : dict
            Additional plotting options
            
        Returns
        -------
        fig, ax : matplotlib Figure and Axes
            Plot objects
        """
        # Default implementation for 2D spatial data
        # Subclasses should override for specific plotting needs
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create default grid if none provided
        if grid is None:
            x_range = np.linspace(0, 1, 50)
            y_range = np.linspace(0, 1, 50)
            grid_x, grid_y = np.meshgrid(x_range, y_range)
            grid = np.column_stack((grid_x.flatten(), grid_y.flatten()))
        
        # Make predictions
        if uncertainty:
            mean, std = self.predict(grid, posterior, return_std=True)
            lower = mean - 1.96 * std
            upper = mean + 1.96 * std
        else:
            mean = self.predict(grid, posterior)
        
        # Reshape for 2D plotting if needed
        if grid.shape[1] == 2:
            n = int(np.sqrt(len(grid)))
            if n*n == len(grid):  # Perfect square
                grid_x = grid[:, 0].reshape(n, n)
                grid_y = grid[:, 1].reshape(n, n)
                mean_grid = mean.reshape(n, n)
                if uncertainty:
                    lower_grid = lower.reshape(n, n)
                    upper_grid = upper.reshape(n, n)
                    
                # Plot the mean prediction as a contour plot
                contour = ax.contourf(grid_x, grid_y, mean_grid, cmap='viridis')
                fig.colorbar(contour, ax=ax, label='Mean prediction')
                
                # Plot uncertainty contours if requested
                if uncertainty:
                    ax.contour(grid_x, grid_y, lower_grid, colors='white', alpha=0.5, 
                              linestyles='dashed', levels=5)
                    ax.contour(grid_x, grid_y, upper_grid, colors='white', alpha=0.5,
                              linestyles='dashed', levels=5)
            else:
                # Non-square grid, use scatter plot
                scatter = ax.scatter(grid[:, 0], grid[:, 1], c=mean, cmap='viridis')
                fig.colorbar(scatter, ax=ax, label='Mean prediction')
        else:
            # 1D data or higher dimensions - just do a scatter plot
            scatter = ax.scatter(range(len(mean)), mean, c=mean, cmap='viridis')
            if uncertainty:
                ax.fill_between(range(len(mean)), lower, upper, alpha=0.3)
            fig.colorbar(scatter, ax=ax, label='Prediction')
        
        ax.set_title(f"{self.name} Spatial Prediction")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        
        return fig, ax 