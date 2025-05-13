"""
REST API for geospatial security services.

This module provides API endpoints for security-related operations
on geospatial data.
"""

from typing import Dict, List, Optional, Any, Union
from flask import Flask, request, jsonify, Blueprint, g
import jwt
import json
import datetime
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon, MultiPolygon, shape
import functools

from geo_infer_sec.core.anonymization import GeospatialAnonymizer
from geo_infer_sec.core.access_control import GeospatialAccessManager, Role, SpatialPermission
from geo_infer_sec.core.compliance import ComplianceFramework, ComplianceRegime


# Create a Blueprint for security API endpoints
security_api = Blueprint('security_api', __name__)

# Global objects (should be initialized with proper config in a real application)
access_manager = None
anonymizer = None
compliance_framework = None


def init_security_api(
    app: Flask, 
    secret_key: str,
    enable_anonymization: bool = True,
    enable_compliance: bool = True
) -> None:
    """
    Initialize the security API with necessary components.
    
    Args:
        app: Flask application
        secret_key: Secret key for JWT token generation/validation
        enable_anonymization: Whether to enable anonymization features
        enable_compliance: Whether to enable compliance checking
    """
    global access_manager, anonymizer, compliance_framework
    
    # Initialize components
    access_manager = GeospatialAccessManager(secret_key)
    
    if enable_anonymization:
        anonymizer = GeospatialAnonymizer()
        
    if enable_compliance:
        compliance_framework = ComplianceFramework()
    
    # Register blueprint with the app
    app.register_blueprint(security_api, url_prefix='/api/security')
    
    # Add error handlers
    @app.errorhandler(401)
    def unauthorized(error):
        return jsonify({"error": "Unauthorized access"}), 401
    
    @app.errorhandler(403)
    def forbidden(error):
        return jsonify({"error": "Access forbidden"}), 403


def token_required(f):
    """Decorator to require a valid JWT token for API access."""
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        # Extract token from Authorization header
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
            
        if not token:
            return jsonify({"error": "Token is missing"}), 401
            
        try:
            # Validate token
            payload = access_manager.validate_token(token)
            if not payload:
                return jsonify({"error": "Invalid or expired token"}), 401
                
            # Store user info in g for use in the route function
            g.user_id = payload['user_id']
            g.roles = payload['roles']
            
        except Exception as e:
            return jsonify({"error": f"Token validation error: {str(e)}"}), 401
            
        return f(*args, **kwargs)
    
    return decorated


@security_api.route('/token', methods=['POST'])
def get_token():
    """Generate a JWT token for a user."""
    data = request.get_json()
    
    if not data or 'user_id' not in data:
        return jsonify({"error": "Missing user_id in request"}), 400
        
    user_id = data['user_id']
    expiration_hours = data.get('expiration_hours', 24)
    
    # Check if user exists and has roles
    user_roles = access_manager.user_roles.get(user_id, [])
    if not user_roles:
        return jsonify({"error": "User has no assigned roles"}), 403
        
    # Generate token
    token = access_manager.generate_token(user_id, expiration_hours)
    
    return jsonify({
        "token": token,
        "expires_in": expiration_hours * 3600,  # seconds
        "roles": user_roles
    })


@security_api.route('/roles', methods=['GET'])
@token_required
def get_roles():
    """Get all roles assigned to the authenticated user."""
    user_id = g.user_id
    
    # Get user roles
    roles = access_manager.get_user_roles(user_id)
    
    # Convert to serializable format
    role_data = [{
        "name": role.name,
        "permission_count": len(role.permissions),
        "permissions": [p.name for p in role.permissions]
    } for role in roles]
    
    return jsonify({"roles": role_data})


@security_api.route('/anonymize', methods=['POST'])
@token_required
def anonymize_data():
    """Anonymize geospatial data."""
    if anonymizer is None:
        return jsonify({"error": "Anonymization service not enabled"}), 501
        
    data = request.get_json()
    
    if not data or 'features' not in data:
        return jsonify({"error": "Invalid or missing GeoJSON data"}), 400
        
    try:
        # Convert GeoJSON to GeoDataFrame
        gdf = gpd.GeoDataFrame.from_features(data['features'])
        
        # Get anonymization parameters
        method = data.get('method', 'location_perturbation')
        params = data.get('parameters', {})
        
        # Apply the specified anonymization method
        if method == 'location_perturbation':
            epsilon = params.get('epsilon', 100.0)
            result = anonymizer.location_perturbation(gdf, epsilon=epsilon)
        elif method == 'spatial_k_anonymity':
            k = params.get('k', 5)
            h3_resolution = params.get('h3_resolution', 9)
            result = anonymizer.spatial_k_anonymity(
                gdf, k=k, h3_resolution=h3_resolution
            )
        elif method == 'geographic_masking':
            # This method requires admin boundaries, which would need to be provided
            # or loaded from a database in a real application
            return jsonify({"error": "Geographic masking requires admin boundaries"}), 400
        else:
            return jsonify({"error": f"Unknown anonymization method: {method}"}), 400
            
        # Convert result back to GeoJSON
        result_geojson = json.loads(result.to_json())
        
        return jsonify({
            "anonymized_data": result_geojson,
            "method": method,
            "parameters": params
        })
        
    except Exception as e:
        return jsonify({"error": f"Anonymization error: {str(e)}"}), 500


@security_api.route('/check-access', methods=['POST'])
@token_required
def check_location_access():
    """Check if the user has access to a specific location."""
    user_id = g.user_id
    data = request.get_json()
    
    if not data or 'latitude' not in data or 'longitude' not in data:
        return jsonify({"error": "Missing location coordinates"}), 400
        
    lat = data['latitude']
    lon = data['longitude']
    
    # Check access
    has_access = access_manager.can_access_location(user_id, lat, lon)
    
    return jsonify({
        "has_access": has_access,
        "location": {
            "latitude": lat,
            "longitude": lon
        }
    })


@security_api.route('/check-compliance', methods=['POST'])
@token_required
def check_compliance():
    """Check data compliance with regulations."""
    if compliance_framework is None:
        return jsonify({"error": "Compliance service not enabled"}), 501
        
    data = request.get_json()
    
    if not data or 'features' not in data:
        return jsonify({"error": "Invalid or missing GeoJSON data"}), 400
        
    try:
        # Convert GeoJSON to GeoDataFrame
        gdf = gpd.GeoDataFrame.from_features(data['features'])
        
        # Get regimes to check against
        regime_names = data.get('regimes', ['gdpr'])
        regimes = [ComplianceRegime(name) for name in regime_names if hasattr(ComplianceRegime, name.upper())]
        
        # Check compliance
        data_reference = data.get('data_reference', 'api_submission')
        violations = compliance_framework.check_geodataframe_compliance(
            gdf, data_reference, regimes
        )
        
        # Convert violations to serializable format
        violation_data = [v.to_dict() for v in violations]
        
        return jsonify({
            "compliant": len(violations) == 0,
            "violations": violation_data,
            "violation_count": len(violations),
            "regimes_checked": [r.value for r in regimes]
        })
        
    except Exception as e:
        return jsonify({"error": f"Compliance check error: {str(e)}"}), 500


@security_api.route('/filter-data', methods=['POST'])
@token_required
def filter_data():
    """Filter geospatial data based on user permissions."""
    user_id = g.user_id
    data = request.get_json()
    
    if not data or 'features' not in data:
        return jsonify({"error": "Invalid or missing GeoJSON data"}), 400
        
    try:
        # Convert GeoJSON to GeoDataFrame
        gdf = gpd.GeoDataFrame.from_features(data['features'])
        
        # Filter data based on user permissions
        filtered = access_manager.filter_geodataframe(user_id, gdf)
        
        # Convert filtered data back to GeoJSON
        filtered_geojson = json.loads(filtered.to_json())
        
        return jsonify({
            "filtered_data": filtered_geojson,
            "original_count": len(gdf),
            "filtered_count": len(filtered)
        })
        
    except Exception as e:
        return jsonify({"error": f"Data filtering error: {str(e)}"}), 500 