from typing import List, Optional, Dict
from datetime import datetime, timedelta

from geo_infer_health.models import Location, EnvironmentalData
from geo_infer_health.utils.geospatial_utils import haversine_distance

class EnvironmentalHealthAnalyzer:
    """Analyzes environmental data in relation to health."""

    def __init__(self, environmental_readings: List[EnvironmentalData]):
        self.readings = sorted(environmental_readings, key=lambda r: r.timestamp)
        # Potential pre-processing: spatial/temporal indexing for readings

    def get_environmental_readings_near_location(
        self, 
        center_loc: Location, 
        radius_km: float, 
        parameter_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[EnvironmentalData]:
        """Retrieves environmental readings near a location within a given time window and for a specific parameter."""
        nearby_readings = []
        for reading in self.readings:
            if haversine_distance(reading.location, center_loc) <= radius_km:
                if parameter_name and reading.parameter_name.lower() != parameter_name.lower():
                    continue
                if start_time and reading.timestamp < start_time:
                    continue
                if end_time and reading.timestamp > end_time:
                    continue
                nearby_readings.append(reading)
        return nearby_readings

    def calculate_average_exposure(
        self, 
        target_locations: List[Location], 
        radius_km: float, 
        parameter_name: str,
        time_window_days: int
    ) -> Dict[str, Optional[float]]: # Returns dict mapping location str to avg value
        """Calculates the average exposure to an environmental parameter for a list of locations.
        
        Args:
            target_locations: A list of Location objects.
            radius_km: Radius to search for environmental data around each target location.
            parameter_name: The specific environmental parameter to analyze (e.g., 'PM2.5').
            time_window_days: How many days back from the most recent reading to consider.

        Returns:
            A dictionary where keys are string representations of target locations 
            and values are the average exposure, or None if no data.
        """
        if not self.readings:
            return {str(loc): None for loc in target_locations}
        
        latest_reading_time = self.readings[-1].timestamp
        start_time = latest_reading_time - timedelta(days=time_window_days)

        avg_exposure_results = {}
        for loc in target_locations:
            relevant_readings = self.get_environmental_readings_near_location(
                center_loc=loc,
                radius_km=radius_km,
                parameter_name=parameter_name,
                start_time=start_time,
                end_time=latest_reading_time
            )
            if not relevant_readings:
                avg_exposure_results[f"{loc.latitude},{loc.longitude}"] = None
            else:
                total_value = sum(r.value for r in relevant_readings)
                avg_value = total_value / len(relevant_readings)
                avg_exposure_results[f"{loc.latitude},{loc.longitude}"] = avg_value
        return avg_exposure_results

    # Placeholder for more complex exposure modeling
    # def estimate_cumulative_exposure(self, person_trajectory: List[Tuple[Location, datetime]], parameter: str):
    #     pass

    # Placeholder for linking environmental data to health outcomes
    # def correlate_env_health(self, disease_reports: List[DiseaseReport], env_parameter: str, lag_time_days: int):
    #     pass 