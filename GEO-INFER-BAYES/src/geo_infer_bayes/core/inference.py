"""
Main inference engine for Bayesian analysis of geospatial data.
"""

import numpy as np
import xarray as xr
from typing import Dict, Any, Optional, Union, List, Tuple

from ..models.base import BayesianModel
from .posterior import PosteriorAnalysis


class BayesianInference:
    """
    Main class for performing Bayesian inference on geospatial data.
    
    This class serves as the primary interface for running different
    Bayesian inference methods on geospatial models.
    
    Parameters
    ----------
    model : BayesianModel
        The Bayesian model to use for inference
    method : str
        Inference method to use: 'mcmc', 'hmc', 'vi', 'smc', or 'abc'
    sampler_config : dict, optional
        Configuration parameters for the sampler
    """
    
    def __init__(
        self, 
        model: 'BayesianModel', 
        method: str = 'mcmc',
        sampler_config: Optional[Dict[str, Any]] = None
    ):
        self.model = model
        self.method = method.lower()
        self.sampler_config = sampler_config or {}
        self.backend = None
        self._initialize_backend()
        
    def _initialize_backend(self) -> None:
        """Initialize the computational backend based on the method."""
        from .mcmc import MCMC
        from .hmc import HMC
        from .variational import VariationalInference
        from .smc import SequentialMonteCarlo
        from .abc import ApproximateBayesianComputation
        
        backends = {
            'mcmc': MCMC,
            'hmc': HMC,
            'vi': VariationalInference,
            'smc': SequentialMonteCarlo,
            'abc': ApproximateBayesianComputation,
        }
        
        if self.method not in backends:
            raise ValueError(
                f"Unsupported inference method: {self.method}. "
                f"Choose from: {', '.join(backends.keys())}"
            )
            
        self.backend = backends[self.method](self.model, **self.sampler_config)
    
    def run(
        self, 
        data: Union[np.ndarray, xr.Dataset, Dict[str, Any]],
        **kwargs
    ) -> PosteriorAnalysis:
        """
        Run the inference algorithm on the provided data.
        
        Parameters
        ----------
        data : array-like or Dataset
            The geospatial data to use for inference
        **kwargs : dict
            Additional arguments to pass to the inference method
            
        Returns
        -------
        PosteriorAnalysis
            Object containing posterior samples and analysis tools
        """
        # Prepare data for the model
        prepared_data = self.model.prepare_data(data)
        
        # Run inference with the selected backend
        samples = self.backend.run(prepared_data, **kwargs)
        
        # Create and return a PosteriorAnalysis object
        return PosteriorAnalysis(
            model=self.model,
            samples=samples,
            data=prepared_data,
            method=self.method
        )
    
    def update(
        self, 
        new_data: Union[np.ndarray, xr.Dataset, Dict[str, Any]],
        previous_posterior: PosteriorAnalysis,
        **kwargs
    ) -> PosteriorAnalysis:
        """
        Update a previous posterior with new data (sequential inference).
        
        Parameters
        ----------
        new_data : array-like or Dataset
            New data to update the posterior
        previous_posterior : PosteriorAnalysis
            Previous posterior analysis result
        **kwargs : dict
            Additional arguments for the update process
            
        Returns
        -------
        PosteriorAnalysis
            Updated posterior analysis
        """
        if not hasattr(self.backend, 'update'):
            raise NotImplementedError(
                f"The {self.method} backend does not support sequential updates"
            )
        
        prepared_data = self.model.prepare_data(new_data)
        updated_samples = self.backend.update(
            prepared_data, 
            previous_posterior.samples,
            **kwargs
        )
        
        return PosteriorAnalysis(
            model=self.model,
            samples=updated_samples,
            data=prepared_data,
            method=self.method
        ) 