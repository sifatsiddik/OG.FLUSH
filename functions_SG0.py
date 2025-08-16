# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 20:24:09 2025

@author: Sifat
"""

from qgis.core import QgsSettings, QgsGeometry, QgsWkbTypes

settings = QgsSettings()
settings.setValue("Processing/OutputFormats/GeoJSON/enabled", True)

def calculate_outlet_distance(outlet_layer, study_area_layer):
    """
    Calculates the maximum distance from outlets to the study area boundary.
    
    Parameters:
        outlet_layer (QgsVectorLayer): Point layer containing outlets.
        study_area_layer (QgsVectorLayer): Polygon layer of the study area.
    
    Returns:
        float: Maximum distance in meters (0.0 if all outlets are within boundary).
    """
    # Merge study area polygons into one
    area_geoms = [feature.geometry() for feature in study_area_layer.getFeatures()]
    merged_area_geom = QgsGeometry.unaryUnion(area_geoms)
    
    # Get boundary using the original working method
    boundary_geom = merged_area_geom.convertToType(QgsWkbTypes.LineGeometry, True)
    
    # all_within_boundary = True
    max_distance = 0.0
    
    for outlet_feature in outlet_layer.getFeatures():
        outlet_geom = outlet_feature.geometry()
        
        if merged_area_geom.contains(outlet_geom):
            continue  # Outlet is within boundary
        else:
            # all_within_boundary = False
            distance = outlet_geom.distance(boundary_geom)
            if distance > max_distance:
                max_distance = distance
    
    # Return 0.0 if all outlets within boundary, otherwise return max distance
    return max_distance