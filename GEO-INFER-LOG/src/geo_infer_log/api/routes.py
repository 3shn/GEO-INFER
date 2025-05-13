"""
API endpoints for route optimization in GEO-INFER-LOG.

This module provides FastAPI endpoints for route optimization functionality.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel, Field

from geo_infer_log.models.schemas import (
    Vehicle, Location, Route, RoutingParameters
)
from geo_infer_log.core.routing import RouteOptimizer, FleetManager, VehicleRouter


router = APIRouter(
    prefix="/routes",
    tags=["routes"],
    responses={404: {"description": "Not found"}},
)


class RouteRequest(BaseModel):
    """Request model for route optimization."""
    origin: Tuple[float, float] = Field(..., description="(lon, lat) of origin")
    destination: Tuple[float, float] = Field(..., description="(lon, lat) of destination")
    waypoints: Optional[List[Tuple[float, float]]] = Field(
        default=None, 
        description="List of (lon, lat) waypoints"
    )
    parameters: Optional[RoutingParameters] = None
    vehicle_id: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "origin": (13.404954, 52.520008),  # Berlin
                "destination": (11.576124, 48.137154),  # Munich
                "waypoints": [
                    (9.993682, 53.551086),  # Hamburg
                    (8.682127, 50.110924)   # Frankfurt
                ],
                "parameters": {
                    "weight_factor": "time",
                    "avoid_highways": False,
                    "avoid_tolls": True
                },
                "vehicle_id": "truck-001"
            }
        }


class VehicleRegistration(BaseModel):
    """Request model for vehicle registration."""
    vehicle: Vehicle

    class Config:
        schema_extra = {
            "example": {
                "vehicle": {
                    "id": "truck-001",
                    "type": "truck",
                    "capacity": 1000,
                    "max_range": 500,
                    "speed": 80,
                    "cost_per_km": 1.2,
                    "emissions_per_km": 0.8,
                    "location": (13.404954, 52.520008),
                    "fuel_type": "diesel",
                    "fuel_capacity": 200,
                    "fuel_level": 150
                }
            }
        }


class VRPRequest(BaseModel):
    """Request model for vehicle routing problem."""
    depot: Location
    deliveries: List[Location]
    vehicles: List[Vehicle]
    constraints: Dict = Field(default_factory=dict)

    class Config:
        schema_extra = {
            "example": {
                "depot": {
                    "name": "Berlin Warehouse",
                    "coordinates": (13.404954, 52.520008),
                    "type": "depot"
                },
                "deliveries": [
                    {
                        "name": "Customer A",
                        "coordinates": (13.5, 52.5),
                        "type": "customer",
                        "service_time": 15
                    },
                    {
                        "name": "Customer B",
                        "coordinates": (13.4, 52.4),
                        "type": "customer",
                        "service_time": 10
                    }
                ],
                "vehicles": [
                    {
                        "id": "truck-001",
                        "type": "truck",
                        "capacity": 1000,
                        "max_range": 500,
                        "speed": 80,
                        "cost_per_km": 1.2,
                        "emissions_per_km": 0.8,
                        "location": (13.404954, 52.520008)
                    }
                ],
                "constraints": {
                    "max_route_duration": 480,
                    "max_stops_per_route": 20
                }
            }
        }


# Get a route optimizer instance
def get_route_optimizer():
    """Dependency for route optimizer."""
    return RouteOptimizer()


# Get a fleet manager instance
def get_fleet_manager():
    """Dependency for fleet manager."""
    return FleetManager()


# Get a vehicle router instance
def get_vehicle_router():
    """Dependency for vehicle router."""
    fleet_manager = get_fleet_manager()
    return VehicleRouter(fleet_manager)


@router.post("/optimize", response_model=Dict)
async def optimize_route(
    request: RouteRequest,
    optimizer: RouteOptimizer = Depends(get_route_optimizer)
):
    """Optimize a route between origin and destination."""
    try:
        # Apply parameters if provided
        if request.parameters:
            optimizer.parameters = request.parameters
            
        # Optimize route
        route = optimizer.optimize_route(
            origin=request.origin,
            destination=request.destination,
            waypoints=request.waypoints
        )
        
        return route
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/vehicles", response_model=Dict)
async def register_vehicle(
    registration: VehicleRegistration,
    fleet_manager: FleetManager = Depends(get_fleet_manager)
):
    """Register a vehicle with the fleet manager."""
    try:
        fleet_manager.add_vehicle(registration.vehicle)
        return {"status": "success", "message": f"Vehicle {registration.vehicle.id} registered"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/vrp", response_model=Dict)
async def solve_vrp(
    request: VRPRequest,
    router: VehicleRouter = Depends(get_vehicle_router)
):
    """Solve a vehicle routing problem."""
    try:
        # Register vehicles with fleet manager
        for vehicle in request.vehicles:
            router.fleet_manager.add_vehicle(vehicle)
            
        # Solve VRP
        result = router.solve_vrp(
            deliveries=[delivery.dict() for delivery in request.deliveries],
            depots=[request.depot.coordinates],
            constraints=request.constraints
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/vehicles", response_model=List[Vehicle])
async def get_vehicles(
    fleet_manager: FleetManager = Depends(get_fleet_manager)
):
    """Get all registered vehicles."""
    try:
        return list(fleet_manager.vehicles.values())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) 