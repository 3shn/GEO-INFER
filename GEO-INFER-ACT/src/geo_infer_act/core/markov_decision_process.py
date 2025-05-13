"""
Markov Decision Process modeling for Active Inference.
"""
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np

from geo_infer_act.utils.math import softmax


class MarkovDecisionProcess:
    """
    Markov Decision Process implementation for active inference.
    
    This class implements the state transition and observation dynamics
    for discrete state spaces within the active inference framework.
    """
    
    def __init__(self, 
                n_states: int, 
                n_observations: int, 
                n_actions: int,
                transition_prob: Optional[np.ndarray] = None,
                observation_prob: Optional[np.ndarray] = None):
        """
        Initialize the Markov Decision Process.
        
        Args:
            n_states: Number of states
            n_observations: Number of observations
            n_actions: Number of actions
            transition_prob: Transition probability tensor (n_states × n_states × n_actions)
            observation_prob: Observation probability matrix (n_observations × n_states)
        """
        self.n_states = n_states
        self.n_observations = n_observations
        self.n_actions = n_actions
        
        # Initialize transition probabilities: P(s'|s,a)
        if transition_prob is None:
            # Default: uniform transitions
            self.transition_prob = np.ones((n_states, n_states, n_actions)) / n_states
        else:
            self._validate_transition_prob(transition_prob)
            self.transition_prob = transition_prob
            
        # Initialize observation probabilities: P(o|s)
        if observation_prob is None:
            # Default: uniform observations
            self.observation_prob = np.ones((n_observations, n_states)) / n_observations
        else:
            self._validate_observation_prob(observation_prob)
            self.observation_prob = observation_prob
        
        # Initialize policy space
        self.policies = self._initialize_policies()
        
    def _validate_transition_prob(self, transition_prob: np.ndarray) -> None:
        """
        Validate transition probability tensor.
        
        Args:
            transition_prob: Transition probability tensor
        """
        expected_shape = (self.n_states, self.n_states, self.n_actions)
        if transition_prob.shape != expected_shape:
            raise ValueError(
                f"Transition probability tensor shape should be {expected_shape}, "
                f"got {transition_prob.shape}"
            )
        
        # Check that probabilities sum to 1 for each state-action pair
        for s in range(self.n_states):
            for a in range(self.n_actions):
                prob_sum = np.sum(transition_prob[s, :, a])
                if not np.isclose(prob_sum, 1.0, rtol=1e-5):
                    raise ValueError(
                        f"Transition probabilities for state {s}, action {a} "
                        f"sum to {prob_sum}, expected 1.0"
                    )
    
    def _validate_observation_prob(self, observation_prob: np.ndarray) -> None:
        """
        Validate observation probability matrix.
        
        Args:
            observation_prob: Observation probability matrix
        """
        expected_shape = (self.n_observations, self.n_states)
        if observation_prob.shape != expected_shape:
            raise ValueError(
                f"Observation probability matrix shape should be {expected_shape}, "
                f"got {observation_prob.shape}"
            )
        
        # Check that probabilities sum to 1 for each state
        for s in range(self.n_states):
            prob_sum = np.sum(observation_prob[:, s])
            if not np.isclose(prob_sum, 1.0, rtol=1e-5):
                raise ValueError(
                    f"Observation probabilities for state {s} "
                    f"sum to {prob_sum}, expected 1.0"
                )
    
    def _initialize_policies(self, horizon: int = 2) -> List[np.ndarray]:
        """
        Initialize the set of possible policies.
        
        A policy is a sequence of actions over a time horizon.
        
        Args:
            horizon: Time horizon for policies
            
        Returns:
            List of policy arrays
        """
        # For a horizon of T and n_actions, we have n_actions^T possible policies
        # This approach might not scale well for large horizons or action spaces
        # In practice, we would use a more efficient way to represent and explore policies
        
        # For simple cases, enumerate all policies
        if horizon <= 3 and self.n_actions <= 5:
            policies = []
            
            # Generate all combinations of actions over the horizon
            for i in range(self.n_actions ** horizon):
                policy = []
                temp = i
                for _ in range(horizon):
                    policy.append(temp % self.n_actions)
                    temp //= self.n_actions
                policies.append(np.array(policy))
                
            return policies
        
        # For larger spaces, sample a subset of policies
        else:
            n_policies = min(100, self.n_actions ** horizon)
            policies = []
            
            for _ in range(n_policies):
                policy = np.random.randint(0, self.n_actions, size=horizon)
                policies.append(policy)
                
            return policies
    
    def get_transition_prob(self, state: int, action: int) -> np.ndarray:
        """
        Get transition probabilities for a given state and action.
        
        Args:
            state: Current state index
            action: Action index
            
        Returns:
            Distribution over next states
        """
        return self.transition_prob[state, :, action]
    
    def get_observation_prob(self, state: int) -> np.ndarray:
        """
        Get observation probabilities for a given state.
        
        Args:
            state: State index
            
        Returns:
            Distribution over observations
        """
        return self.observation_prob[:, state]
    
    def transition(self, state: int, action: int) -> int:
        """
        Sample next state given current state and action.
        
        Args:
            state: Current state index
            action: Action index
            
        Returns:
            Next state index
        """
        # Get transition distribution for the current state and action
        transition_dist = self.transition_prob[state, :, action]
        
        # Sample next state
        next_state = np.random.choice(self.n_states, p=transition_dist)
        
        return next_state
    
    def observe(self, state: int) -> int:
        """
        Sample observation given current state.
        
        Args:
            state: Current state index
            
        Returns:
            Observation index
        """
        # Get observation distribution for the current state
        observation_dist = self.observation_prob[:, state]
        
        # Sample observation
        observation = np.random.choice(self.n_observations, p=observation_dist)
        
        return observation
    
    def simulate(self, 
                initial_state: int, 
                policy: Union[List[int], np.ndarray],
                stochastic: bool = True) -> Tuple[List[int], List[int]]:
        """
        Simulate a trajectory through the MDP following a policy.
        
        Args:
            initial_state: Initial state index
            policy: Sequence of actions to take
            stochastic: Whether to sample stochastically or take most likely outcome
            
        Returns:
            Tuple of (state_trajectory, observation_trajectory)
        """
        state_trajectory = [initial_state]
        observation_trajectory = [self.observe(initial_state)]
        
        current_state = initial_state
        
        for action in policy:
            if stochastic:
                # Stochastic transition
                next_state = self.transition(current_state, action)
            else:
                # Deterministic transition (most likely)
                transition_dist = self.transition_prob[current_state, :, action]
                next_state = np.argmax(transition_dist)
            
            state_trajectory.append(next_state)
            
            if stochastic:
                # Stochastic observation
                observation = self.observe(next_state)
            else:
                # Deterministic observation (most likely)
                observation_dist = self.observation_prob[:, next_state]
                observation = np.argmax(observation_dist)
                
            observation_trajectory.append(observation)
            current_state = next_state
            
        return state_trajectory, observation_trajectory
    
    def get_predictive_state(self, 
                            belief: np.ndarray, 
                            action: int) -> np.ndarray:
        """
        Get predictive state distribution after an action.
        
        Args:
            belief: Current belief distribution over states
            action: Action index
            
        Returns:
            Predicted belief distribution
        """
        # For each possible current state, compute distribution over next states
        # and weight by current belief
        predictive_state = np.zeros(self.n_states)
        
        for s in range(self.n_states):
            state_prob = belief[s]
            transition_dist = self.transition_prob[s, :, action]
            predictive_state += state_prob * transition_dist
            
        return predictive_state
    
    def get_predictive_observation(self, 
                                 state_dist: np.ndarray) -> np.ndarray:
        """
        Get predictive observation distribution given a state distribution.
        
        Args:
            state_dist: Distribution over states
            
        Returns:
            Predicted observation distribution
        """
        # For each possible state, compute distribution over observations
        # and weight by state probability
        predictive_obs = np.zeros(self.n_observations)
        
        for s in range(self.n_states):
            state_prob = state_dist[s]
            observation_dist = self.observation_prob[:, s]
            predictive_obs += state_prob * observation_dist
            
        return predictive_obs
    
    def update_belief(self, 
                     prior_belief: np.ndarray, 
                     observation: int) -> np.ndarray:
        """
        Update belief distribution using Bayes' rule.
        
        Args:
            prior_belief: Prior belief distribution over states
            observation: Observed outcome
            
        Returns:
            Posterior belief distribution
        """
        # Likelihood: P(o|s)
        likelihood = self.observation_prob[observation, :]
        
        # Posterior (unnormalized): P(s|o) ∝ P(o|s) * P(s)
        posterior = likelihood * prior_belief
        
        # Normalize
        posterior = posterior / np.sum(posterior)
        
        return posterior
    
    def set_transition_matrix(self, 
                             state: int, 
                             action: int, 
                             distribution: np.ndarray) -> None:
        """
        Set transition distribution for a specific state-action pair.
        
        Args:
            state: State index
            action: Action index
            distribution: Distribution over next states (must sum to 1)
        """
        if distribution.shape != (self.n_states,):
            raise ValueError(
                f"Distribution shape should be ({self.n_states},), got {distribution.shape}"
            )
            
        if not np.isclose(np.sum(distribution), 1.0, rtol=1e-5):
            raise ValueError(
                f"Distribution should sum to 1, got {np.sum(distribution)}"
            )
            
        self.transition_prob[state, :, action] = distribution
        
    def set_observation_matrix(self, 
                              state: int, 
                              distribution: np.ndarray) -> None:
        """
        Set observation distribution for a specific state.
        
        Args:
            state: State index
            distribution: Distribution over observations (must sum to 1)
        """
        if distribution.shape != (self.n_observations,):
            raise ValueError(
                f"Distribution shape should be ({self.n_observations},), got {distribution.shape}"
            )
            
        if not np.isclose(np.sum(distribution), 1.0, rtol=1e-5):
            raise ValueError(
                f"Distribution should sum to 1, got {np.sum(distribution)}"
            )
            
        self.observation_prob[:, state] = distribution 