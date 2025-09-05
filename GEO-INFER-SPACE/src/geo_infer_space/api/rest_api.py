"""
FastAPI REST API for GEO-INFER-SPACE spatial services.

This module provides HTTP endpoints for accessing spatial analysis capabilities
with automatic documentation, validation, and error handling.
"""

import logging
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import geopandas as gpd
import json
from shapely.geometry import shape

from .schemas import (
    SpatialAnalysisRequest,
    SpatialAnalysisResponse,
    BufferAnalysisRequest,
    ProximityAnalysisRequest,
    InterpolationRequest,
    ClusteringRequest,
    HotspotRequest,
    NetworkAnalysisRequest,
    TerrainAnalysisRequest,
    H3AnalysisRequest,
    ErrorResponse
)

from ..analytics import (
    buffer_and_intersect,
    overlay_analysis,
    proximity_analysis,
    spatial_interpolation,
    clustering_analysis,
    hotspot_detection,
    shortest_path,
    service_area,
    network_connectivity,
    terrain_analysis
)

from ..utils.h3_utils import (
    polygon_to_cells,
    cell_to_latlng_boundary,
    latlng_to_cell,
    grid_disk,
    compact_cells
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="GEO-INFER-SPACE API",
    description="Advanced spatial analysis and processing services",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create API router
from fastapi import APIRouter
router = APIRouter(prefix="/api/v1", tags=["spatial"])


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="InternalServerError",
            message="An internal server error occurred",
            details={"exception": str(exc)}
        ).dict()
    )


def geojson_to_gdf(geojson_data: Dict[str, Any], crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """Convert GeoJSON data to GeoDataFrame."""
    try:
        if geojson_data.get("type") == "FeatureCollection":
            return gpd.GeoDataFrame.from_features(geojson_data["features"], crs=crs)
        elif geojson_data.get("type") == "Feature":
            return gpd.GeoDataFrame.from_features([geojson_data], crs=crs)
        else:
            # Assume it's a geometry
            geom = shape(geojson_data)
            return gpd.GeoDataFrame([{"geometry": geom}], crs=crs)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid GeoJSON data: {e}")


def gdf_to_geojson(gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
    """Convert GeoDataFrame to GeoJSON."""
    try:
        return json.loads(gdf.to_json())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to convert to GeoJSON: {e}")


@router.post("/buffer", response_model=SpatialAnalysisResponse)
async def buffer_analysis_endpoint(request: BufferAnalysisRequest):
    """
    Perform buffer analysis on geometries.
    
    Creates buffers around input geometries with specified distance.
    Optionally dissolves overlapping buffers into single features.
    """
    try:
        # Convert GeoJSON to GeoDataFrame
        gdf = geojson_to_gdf(request.data.dict(), request.crs)
        
        # Create buffers
        buffered_gdf = gdf.copy()
        buffered_gdf['geometry'] = gdf.geometry.buffer(request.buffer_distance)
        
        # Dissolve if requested
        if request.dissolve:
            from shapely.ops import unary_union
            dissolved_geom = unary_union(buffered_gdf.geometry.tolist())
            buffered_gdf = gpd.GeoDataFrame([{"geometry": dissolved_geom}], crs=gdf.crs)
        
        # Convert back to GeoJSON
        result_geojson = gdf_to_geojson(buffered_gdf)
        
        return SpatialAnalysisResponse(
            success=True,
            result=result_geojson,
            message=f"Buffer analysis completed with distance {request.buffer_distance}",
            metadata={
                "buffer_distance": request.buffer_distance,
                "dissolved": request.dissolve,
                "num_features": len(buffered_gdf)
            }
        )
        
    except Exception as e:
        logger.error(f"Buffer analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/proximity", response_model=SpatialAnalysisResponse)
async def proximity_analysis_endpoint(request: ProximityAnalysisRequest):
    """
    Perform proximity analysis between two sets of geometries.
    
    Calculates distance metrics and identifies nearest features
    between source and target geometries.
    """
    try:
        # Convert GeoJSON to GeoDataFrames
        source_gdf = geojson_to_gdf(request.source_data.dict(), request.crs)
        target_gdf = geojson_to_gdf(request.target_data.dict(), request.crs)
        
        # Perform proximity analysis
        result_gdf = proximity_analysis(source_gdf, target_gdf, request.max_distance)
        
        # Convert to GeoJSON
        result_geojson = gdf_to_geojson(result_gdf)
        
        return SpatialAnalysisResponse(
            success=True,
            result=result_geojson,
            message="Proximity analysis completed",
            metadata={
                "num_source_features": len(source_gdf),
                "num_target_features": len(target_gdf),
                "num_results": len(result_gdf),
                "max_distance": request.max_distance
            }
        )
        
    except Exception as e:
        logger.error(f"Proximity analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/interpolation", response_model=SpatialAnalysisResponse)
async def interpolation_endpoint(request: InterpolationRequest):
    """
    Perform spatial interpolation on point data.
    
    Creates a continuous surface from discrete point observations
    using various interpolation methods (IDW, Kriging, RBF, etc.).
    """
    try:
        # Convert points to GeoDataFrame
        points_gdf = geojson_to_gdf(request.points.dict(), request.crs)
        
        # Validate value column exists
        if request.value_column not in points_gdf.columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Column '{request.value_column}' not found in point data"
            )
        
        # Perform interpolation
        result_gdf = spatial_interpolation(
            points_gdf=points_gdf,
            value_column=request.value_column,
            grid_bounds=tuple(request.bounds),
            grid_resolution=request.resolution,
            method=request.method,
            **request.parameters
        )
        
        # Convert to GeoJSON
        result_geojson = gdf_to_geojson(result_gdf)
        
        return SpatialAnalysisResponse(
            success=True,
            result=result_geojson,
            message=f"Spatial interpolation completed using {request.method}",
            metadata={
                "method": request.method,
                "resolution": request.resolution,
                "num_input_points": len(points_gdf),
                "num_grid_points": len(result_gdf),
                "bounds": request.bounds
            }
        )
        
    except Exception as e:
        logger.error(f"Interpolation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clustering", response_model=SpatialAnalysisResponse)
async def clustering_endpoint(request: ClusteringRequest):
    """
    Perform spatial clustering analysis on point data.
    
    Groups points into clusters based on spatial proximity
    using various clustering algorithms (DBSCAN, K-means, etc.).
    """
    try:
        # Convert points to GeoDataFrame
        points_gdf = geojson_to_gdf(request.points.dict(), request.crs)
        
        # Perform clustering
        result_gdf = clustering_analysis(
            points_gdf=points_gdf,
            method=request.method,
            **request.parameters
        )
        
        # Convert to GeoJSON
        result_geojson = gdf_to_geojson(result_gdf)
        
        # Calculate cluster statistics
        cluster_stats = {}
        if 'cluster' in result_gdf.columns:
            cluster_counts = result_gdf['cluster'].value_counts()
            cluster_stats = {
                "num_clusters": len(cluster_counts),
                "largest_cluster": int(cluster_counts.max()),
                "smallest_cluster": int(cluster_counts.min()),
                "noise_points": int(cluster_counts.get(-1, 0))
            }
        
        return SpatialAnalysisResponse(
            success=True,
            result=result_geojson,
            message=f"Clustering analysis completed using {request.method}",
            metadata={
                "method": request.method,
                "num_points": len(points_gdf),
                "cluster_statistics": cluster_stats
            }
        )
        
    except Exception as e:
        logger.error(f"Clustering analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/hotspots", response_model=SpatialAnalysisResponse)
async def hotspot_detection_endpoint(request: HotspotRequest):
    """
    Detect spatial hotspots and coldspots in point data.
    
    Identifies statistically significant clusters of high or low values
    using spatial statistics methods (Getis-Ord Gi*, Local Moran's I, etc.).
    """
    try:
        # Convert points to GeoDataFrame
        points_gdf = geojson_to_gdf(request.points.dict(), request.crs)
        
        # Validate value column if provided
        if request.value_column and request.value_column not in points_gdf.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Column '{request.value_column}' not found in point data"
            )
        
        # Perform hotspot detection
        result_gdf = hotspot_detection(
            points_gdf=points_gdf,
            value_column=request.value_column,
            method=request.method,
            **request.parameters
        )
        
        # Convert to GeoJSON
        result_geojson = gdf_to_geojson(result_gdf)
        
        # Calculate hotspot statistics
        hotspot_stats = {}
        if 'hotspot_type' in result_gdf.columns:
            hotspot_counts = result_gdf['hotspot_type'].value_counts()
            hotspot_stats = {
                "hot_spots": int(hotspot_counts.get('Hot Spot', 0)),
                "cold_spots": int(hotspot_counts.get('Cold Spot', 0)),
                "not_significant": int(hotspot_counts.get('Not Significant', 0))
            }
        
        return SpatialAnalysisResponse(
            success=True,
            result=result_geojson,
            message=f"Hotspot detection completed using {request.method}",
            metadata={
                "method": request.method,
                "value_column": request.value_column,
                "num_points": len(points_gdf),
                "hotspot_statistics": hotspot_stats
            }
        )
        
    except Exception as e:
        logger.error(f"Hotspot detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/network", response_model=SpatialAnalysisResponse)
async def network_analysis_endpoint(request: NetworkAnalysisRequest):
    """
    Perform network analysis operations.
    
    Analyzes transportation networks for routing, accessibility,
    and connectivity using graph-based algorithms.
    """
    try:
        # Convert network to GeoDataFrame
        network_gdf = geojson_to_gdf(request.network.dict(), request.crs)
        
        if request.analysis_type == "connectivity":
            # Network connectivity analysis
            result = network_connectivity(
                network_gdf=network_gdf,
                **request.parameters
            )
            
            return SpatialAnalysisResponse(
                success=True,
                result=result,
                message="Network connectivity analysis completed",
                metadata={
                    "analysis_type": request.analysis_type,
                    "num_edges": len(network_gdf)
                }
            )
            
        elif request.analysis_type == "service_area":
            # Service area analysis requires center point
            if not request.origins or len(request.origins.features) == 0:
                raise HTTPException(
                    status_code=400,
                    detail="Service area analysis requires origin points"
                )
            
            origins_gdf = geojson_to_gdf(request.origins.dict(), request.crs)
            center_point = origins_gdf.geometry.iloc[0]
            
            max_distance = request.parameters.get('max_distance', 1000)
            
            result_gdf = service_area(
                network_gdf=network_gdf,
                center_point=center_point,
                max_distance=max_distance
            )
            
            result_geojson = gdf_to_geojson(result_gdf)
            
            return SpatialAnalysisResponse(
                success=True,
                result=result_geojson,
                message="Service area analysis completed",
                metadata={
                    "analysis_type": request.analysis_type,
                    "max_distance": max_distance,
                    "num_areas": len(result_gdf)
                }
            )
            
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Analysis type '{request.analysis_type}' not yet implemented"
            )
        
    except Exception as e:
        logger.error(f"Network analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/h3", response_model=SpatialAnalysisResponse)
async def h3_analysis_endpoint(request: H3AnalysisRequest):
    """
    Perform H3 hexagonal grid operations.
    
    Converts geometries to H3 cells, performs grid operations,
    and provides H3-based spatial indexing capabilities.
    """
    try:
        if request.operation == "polygon_to_cells":
            # Convert geometry to H3 cells
            geom_dict = request.geometry.dict()
            
            h3_cells = polygon_to_cells(geom_dict, request.resolution)
            
            # Create GeoJSON features for each cell
            from ..utils.h3_utils import cell_to_latlng_boundary
            
            features = []
            for cell in h3_cells:
                boundary = cell_to_latlng_boundary(cell)
                # Convert to proper coordinate format for GeoJSON
                coords = [[list(coord) for coord in boundary]]
                
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": coords
                    },
                    "properties": {
                        "h3_index": cell,
                        "resolution": request.resolution
                    }
                }
                features.append(feature)
            
            result = {
                "type": "FeatureCollection",
                "features": features
            }
            
            return SpatialAnalysisResponse(
                success=True,
                result=result,
                message=f"H3 polygon conversion completed",
                metadata={
                    "operation": request.operation,
                    "resolution": request.resolution,
                    "num_cells": len(h3_cells)
                }
            )
            
        elif request.operation == "grid_disk":
            # Grid disk operation requires center cell
            center_cell = request.parameters.get('center_cell')
            k = request.parameters.get('k', 1)
            
            if not center_cell:
                raise HTTPException(
                    status_code=400,
                    detail="Grid disk operation requires 'center_cell' parameter"
                )
            
            disk_cells = grid_disk(center_cell, k)
            
            return SpatialAnalysisResponse(
                success=True,
                result={"cells": disk_cells},
                message=f"H3 grid disk completed",
                metadata={
                    "operation": request.operation,
                    "center_cell": center_cell,
                    "k": k,
                    "num_cells": len(disk_cells)
                }
            )
            
        else:
            raise HTTPException(
                status_code=400,
                detail=f"H3 operation '{request.operation}' not yet implemented"
            )
        
    except Exception as e:
        logger.error(f"H3 analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "GEO-INFER-SPACE"}


@router.get("/capabilities")
async def get_capabilities():
    """Get available analysis capabilities."""
    return {
        "vector_operations": [
            "buffer", "overlay", "proximity", "spatial_join", "geometric_calculations"
        ],
        "raster_operations": [
            "terrain_analysis", "map_algebra", "focal_statistics", "zonal_statistics"
        ],
        "network_analysis": [
            "shortest_path", "service_area", "connectivity", "routing", "accessibility"
        ],
        "geostatistics": [
            "interpolation", "clustering", "hotspot_detection", "autocorrelation"
        ],
        "h3_operations": [
            "polygon_to_cells", "grid_disk", "compact_cells", "cell_boundaries"
        ],
        "point_cloud": [
            "filtering", "feature_extraction", "classification", "surface_generation"
        ]
    }


# Include router in app
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
