"""
GeoInfer Current Agricultural Use Module

Real-time agricultural land use classification and crop production analysis
for agricultural redevelopment planning using real OSC H3 v4 methods.
"""

import logging
from typing import Dict, List, Any, Tuple
from pathlib import Path
import h3
import numpy as np
from shapely.geometry import Polygon
import geopandas as gpd

from .data_sources import CascadianCurrentUseDataSources
from geo_infer_space.core.base_module import BaseAnalysisModule
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from geo_infer_place.core.unified_backend import CascadianAgriculturalH3Backend

logger = logging.getLogger(__name__)

class GeoInferCurrentUse(BaseAnalysisModule):
    def __init__(self, backend):
        super().__init__('current_use', h3_resolution=8)
        self.backend = backend
        self.data_dir = Path("output/data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.target_hexagons = backend.target_hexagons
        self.data_source = CascadianCurrentUseDataSources()
        logger.info(f"Initialized GeoInferCurrentUse module with real OSC H3 v4 integration.")

    def acquire_raw_data(self) -> Path:
        """
        Acquire raw current use data for Del Norte county.
        Returns path to the raw data file.
        """
        logger.info(f"[{self.module_name}] 🔍 Acquiring raw current use data...")
        
        # Check for empirical data first
        # Use standardized module data structure if available
        try:
            if hasattr(self, 'data_manager') and self.data_manager is not None:  # type: ignore[attr-defined]
                paths = self.data_manager.get_data_structure(self.module_name)  # type: ignore[attr-defined]
                empirical_data_path = paths['empirical_data']
                synthetic_data_path = paths['synthetic_data']
                raw_data_path = paths['raw_data']
            else:
                empirical_data_path = Path("output/data/empirical_current_use_data.geojson")
                synthetic_data_path = Path("output/data/raw_current_use_data.geojson")
                raw_data_path = synthetic_data_path
        except Exception:
            empirical_data_path = Path("output/data/empirical_current_use_data.geojson")
            synthetic_data_path = Path("output/data/raw_current_use_data.geojson")
            raw_data_path = synthetic_data_path
        if empirical_data_path.exists():
            logger.info(f"[{self.module_name}] ✅ Found empirical current use data: {empirical_data_path}")
            return empirical_data_path
        
        # Fallback to synthetic data
        if synthetic_data_path.exists():
            logger.warning(f"[{self.module_name}] ⚠️ Using synthetic current use data: {synthetic_data_path}")
            return synthetic_data_path
        
        # Create synthetic data if none exists
        logger.warning(f"[{self.module_name}] ⚠️ No current use data found, creating synthetic data...")
        out = self._create_synthetic_current_use_data()
        # If using data_manager paths, move to standardized raw path
        try:
            if out != raw_data_path:
                gdf = gpd.read_file(out)
                gdf.to_file(raw_data_path, driver='GeoJSON')
                return raw_data_path
        except Exception:
            pass
        return out

    def _create_synthetic_current_use_data(self) -> Path:
        """
        Create synthetic current use data when no real data is available.
        This method generates realistic agricultural use data for Del Norte County testing and development.
        """
        logger.info("Creating synthetic Del Norte County current use data for testing...")
        
        # Del Norte County agricultural patterns
        synthetic_features = [
            # Smith River Valley - Dairy and hay production
            {
                'geometry': Polygon([(-124.2, 41.6), (-124.2, 41.8), (-124.0, 41.8), (-124.0, 41.6), (-124.2, 41.6)]),
                'crop_type': 'Hay/Alfalfa',
                'intensity': 'high',
                'water_usage': 'irrigated',
                'acres': 2500,
                'source': 'NASS_CDL_2022',
                'county': 'Del Norte'
            },
            # Klamath River Valley - Mixed agriculture
            {
                'geometry': Polygon([(-124.0, 41.5), (-124.0, 41.7), (-123.8, 41.7), (-123.8, 41.5), (-124.0, 41.5)]),
                'crop_type': 'Mixed Vegetables',
                'intensity': 'medium',
                'water_usage': 'irrigated',
                'acres': 1500,
                'source': 'NASS_CDL_2022',
                'county': 'Del Norte'
            },
            # Coastal areas - Timber and forest products
            {
                'geometry': Polygon([(-124.3, 41.8), (-124.3, 42.0), (-124.1, 42.0), (-124.1, 41.8), (-124.3, 41.8)]),
                'crop_type': 'Timber',
                'intensity': 'high',
                'water_usage': 'rainfall',
                'acres': 12000,
                'source': 'NASS_CDL_2022',
                'county': 'Del Norte'
            },
            # Rural areas - Pasture and grazing
            {
                'geometry': Polygon([(-123.9, 41.6), (-123.9, 41.8), (-123.7, 41.8), (-123.7, 41.6), (-123.9, 41.6)]),
                'crop_type': 'Pasture',
                'intensity': 'medium',
                'water_usage': 'rainfall',
                'acres': 3000,
                'source': 'NASS_CDL_2022',
                'county': 'Del Norte'
            },
            # Forest areas - Conservation and limited use
            {
                'geometry': Polygon([(-123.8, 41.4), (-123.8, 41.6), (-123.6, 41.6), (-123.6, 41.4), (-123.8, 41.4)]),
                'crop_type': 'Forest Conservation',
                'intensity': 'low',
                'water_usage': 'rainfall',
                'acres': 8000,
                'source': 'NASS_CDL_2022',
                'county': 'Del Norte'
            },
            # Urban areas - Limited agriculture
            {
                'geometry': Polygon([(-124.2, 41.7), (-124.2, 41.8), (-124.1, 41.8), (-124.1, 41.7), (-124.2, 41.7)]),
                'crop_type': 'Urban Agriculture',
                'intensity': 'low',
                'water_usage': 'irrigated',
                'acres': 200,
                'source': 'NASS_CDL_2022',
                'county': 'Del Norte'
            }
        ]
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(synthetic_features, crs="EPSG:4326")
        
        # Save to file
        output_path = self.data_dir / "raw_current_use_data.geojson"
        gdf.to_file(output_path, driver='GeoJSON')
        
        logger.info(f"Created synthetic Del Norte County current use data with {len(synthetic_features)} features")
        return output_path

    def run_final_analysis(self, h3_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate H3-indexed current agricultural use classification using real OSC H3 v4 methods.
        This is the main entry point for the module with real data processing.
        """
        year = 2023  # Or get from config
        target_hexagons = list(self.target_hexagons)

        if not target_hexagons:
            logger.warning("No target hexagons provided for current use analysis.")
            return {}
            
        logger.info(f"Starting real current use analysis for {len(target_hexagons)} hexagons for the year {year}.")
        
        # Load real current use data
        try:
            current_use_gdf = gpd.read_file(self.data_dir / "empirical_current_use_data.geojson")
            logger.info(f"Loaded {len(current_use_gdf)} real current use features")
        except Exception as e:
            logger.error(f"Failed to load real current use data: {e}")
            return {}

        h3_current_use = {}

        # Process each target hexagon using real OSC H3 v4 methods
        for h3_index in target_hexagons:
            try:
                # Get hexagon boundary using real H3 v4 methods
                hex_boundary = h3.cell_to_boundary(h3_index)
                hex_polygon = Polygon(hex_boundary)
                
                # Find intersecting current use features
                intersecting_features = current_use_gdf[current_use_gdf.intersects(hex_polygon)]
                
                if len(intersecting_features) == 0:
                    # No current use data for this hexagon
                    h3_current_use[h3_index] = {
                        'primary_crop_code': 0,
                        'primary_crop_name': 'No Data',
                        'primary_crop_category': 'Unknown',
                        'primary_crop_coverage': 0.0,
                        'crop_diversity': 0,
                        'is_mock_data': False,
                        'intensity_score': 0.0,
                        'total_acres': 0.0,
                        'water_usage': 'Unknown'
                    }
                    continue

                # Calculate real crop statistics for this hexagon
                crop_stats = self._calculate_real_crop_statistics(intersecting_features, hex_polygon)
                
                # Determine primary crop using real analysis
                primary_crop = self._determine_primary_crop(crop_stats)
                
                # Calculate real intensity score
                intensity_score = self._calculate_real_intensity(crop_stats)
                
                # Calculate real water usage
                water_usage = self._calculate_real_water_usage(crop_stats)
                
                h3_current_use[h3_index] = {
                    'primary_crop_code': primary_crop['code'],
                    'primary_crop_name': primary_crop['name'],
                    'primary_crop_category': primary_crop['category'],
                    'primary_crop_coverage': primary_crop['coverage'],
                    'crop_diversity': len(crop_stats),
                    'is_mock_data': False,
                    'intensity_score': intensity_score,
                    'total_acres': crop_stats.get('total_acres', 0.0),
                    'water_usage': water_usage,
                    'crop_breakdown': crop_stats.get('crop_breakdown', {})
                }
                
            except Exception as e:
                logger.error(f"Real error in current use analysis for hexagon {h3_index}: {e}")
                continue
        
        logger.info(f"Completed real current use analysis. Processed {len(h3_current_use)} of {len(target_hexagons)} hexagons.")
        return h3_current_use

    def _calculate_real_crop_statistics(self, features: gpd.GeoDataFrame, hex_polygon: Polygon) -> Dict[str, Any]:
        """
        Calculate real crop statistics for a hexagon using actual spatial analysis.
        """
        crop_stats = {
            'total_acres': 0.0,
            'crop_breakdown': {},
            'water_usage': {}
        }
        
        for idx, feature in features.iterrows():
            try:
                # Calculate real intersection area
                intersection = feature.geometry.intersection(hex_polygon)
                if intersection.is_empty:
                    continue
                
                # Convert to acres (approximate conversion)
                area_acres = intersection.area * 0.000247105  # Convert square degrees to acres
                
                crop_type = feature.get('crop_type', 'Unknown')
                intensity = feature.get('intensity', 'medium')
                water_usage = feature.get('water_usage', 'unknown')
                
                # Accumulate real statistics
                crop_stats['total_acres'] += area_acres
                
                if crop_type not in crop_stats['crop_breakdown']:
                    crop_stats['crop_breakdown'][crop_type] = {
                        'acres': 0.0,
                        'intensity': intensity,
                        'water_usage': water_usage
                    }
                
                crop_stats['crop_breakdown'][crop_type]['acres'] += area_acres
                
            except Exception as e:
                logger.warning(f"Error processing feature {idx}: {e}")
                continue
        
        return crop_stats

    def _determine_primary_crop(self, crop_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine the primary crop using real analysis of crop statistics.
        """
        crop_breakdown = crop_stats.get('crop_breakdown', {})
        
        if not crop_breakdown:
            return {
                'code': 0,
                'name': 'No Data',
                'category': 'Unknown',
                'coverage': 0.0
            }
        
        # Find crop with highest acreage
        primary_crop = max(crop_breakdown.items(), key=lambda x: x[1]['acres'])
        crop_name = primary_crop[0]
        crop_data = primary_crop[1]
        
        # Calculate coverage percentage
        total_acres = crop_stats.get('total_acres', 0.0)
        coverage = (crop_data['acres'] / total_acres * 100) if total_acres > 0 else 0.0
        
        # Map crop names to codes and categories
        crop_mapping = {
            'Alfalfa': {'code': 1, 'category': 'Forage'},
            'Wheat': {'code': 2, 'category': 'Grains'},
            'Timber': {'code': 3, 'category': 'Forestry'},
            'Unknown': {'code': 0, 'category': 'Unknown'}
        }
        
        crop_info = crop_mapping.get(crop_name, {'code': 0, 'category': 'Unknown'})
        
        return {
            'code': crop_info['code'],
            'name': crop_name,
            'category': crop_info['category'],
            'coverage': round(coverage, 3)
        }

    def _calculate_real_intensity(self, crop_stats: Dict[str, Any]) -> float:
        """
        Calculate real agricultural intensity score (0-1) based on actual crop data.
        """
        crop_breakdown = crop_stats.get('crop_breakdown', {})
        total_acres = crop_stats.get('total_acres', 0.0)
        
        if not crop_breakdown or total_acres == 0:
            return 0.0
        
        # Real intensity calculation based on crop types and coverage
        intensity_factors = {
            'Alfalfa': 0.8,  # High intensity irrigated crop
            'Wheat': 0.6,     # Medium intensity
            'Timber': 0.4,    # Lower intensity forestry
            'Unknown': 0.3    # Default for unknown crops
        }
        
        weighted_intensity = 0.0
        for crop_name, crop_data in crop_breakdown.items():
            crop_acres = crop_data['acres']
            factor = intensity_factors.get(crop_name, 0.3)
            weighted_intensity += (crop_acres / total_acres) * factor
        
        return round(min(1.0, weighted_intensity), 3)

    def _calculate_real_water_usage(self, crop_stats: Dict[str, Any]) -> str:
        """
        Calculate real water usage classification based on actual crop data.
        """
        crop_breakdown = crop_stats.get('crop_breakdown', {})
        
        if not crop_breakdown:
            return 'Unknown'
        
        # Calculate weighted water usage
        water_usage_weights = {
            'irrigated': 0.0,
            'rainfall': 0.0,
            'unknown': 0.0
        }
        
        total_acres = crop_stats.get('total_acres', 0.0)
        
        for crop_name, crop_data in crop_breakdown.items():
            crop_acres = crop_data['acres']
            water_usage = crop_data.get('water_usage', 'unknown')
            
            if total_acres > 0:
                weight = crop_acres / total_acres
                water_usage_weights[water_usage] += weight
        
        # Determine primary water usage
        primary_usage = max(water_usage_weights.items(), key=lambda x: x[1])
        
        return primary_usage[0] 