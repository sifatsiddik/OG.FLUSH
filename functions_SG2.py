# -*- coding: utf-8 -*-
"""
Created on Wed Feb 05 21:31:34 2025

@author: Sifat
"""

from qgis.core import QgsVectorLayer, QgsSettings, QgsField, QgsFeature, QgsProperty
from qgis import processing
from PyQt5.QtCore import QVariant
import os
import pandas as pd
import numpy as np

settings = QgsSettings()
settings.setValue("Processing/OutputFormats/GeoJSON/enabled", True)
#%%

# Set up field data types for selected fields for consistent conversion between dataframe and shapefile
field_type_mapping = {
    'NODE_ID': {'pandas': 'string', 'qgis': QVariant.String},
    'LINK_ID': {'pandas': 'string', 'qgis': QVariant.String},
    'FROM_NODE': {'pandas': 'string', 'qgis': QVariant.String},
    'TO_NODE': {'pandas': 'string', 'qgis': QVariant.String},
    'sub_id': {'pandas': 'string', 'qgis': QVariant.String},
    'Cover_Elev': {'pandas': 'float64', 'qgis': QVariant.Double},
    'INVERT_ELV': {'pandas': 'float64', 'qgis': QVariant.Double},
    'area': {'pandas': 'float64', 'qgis': QVariant.Double},
    'length': {'pandas': 'float64', 'qgis': QVariant.Double},
    'slope_perc': {'pandas': 'float64', 'qgis': QVariant.Double},
    'slope': {'pandas': 'float64', 'qgis': QVariant.Double},
    'Diameter': {'pandas': 'float64', 'qgis': QVariant.Double},
}

def shapefile_to_dataframe(shapefile):
    if not shapefile.isValid():
        print("Layer failed to load!")
        return None
    
    # List to hold the feature attributes
    data = []
    
    # Iterate through the features in the layer
    for feature in shapefile.getFeatures():
        # Create a dictionary for the feature's attributes
        feature_dict = feature.attributes()
        # Append the dictionary to the data list
        data.append(feature_dict)
    
    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(data)
    
    # Set the column names
    df.columns = shapefile.fields().names()
    
    # Apply data type conversions based on lookup
    for field_name, type_info in field_type_mapping.items():
        if field_name in df.columns:
            target_type = type_info['pandas']
            try:
                df[field_name] = df[field_name].astype(target_type)
            except (ValueError, TypeError) as e:
                print(f"Warning: Could not convert field '{field_name}' to {target_type}. Error: {e}")
                print(f"Keeping original data type for field '{field_name}'")
    
    return df

def shapefile_to_array(shapefile):
    # Convert to numpy array
    df_array = shapefile_to_dataframe(shapefile).to_numpy()
    
    return df_array


# 3. Determination of contributing areas
#---------------------------------------------------------------------------------------------------------------

def remove_duplicate_features(shapefile, overlap_threshold=0.8):
    """
    Removes duplicate features based on geometry overlap, keeping the larger feature when duplicates are found.
    
    Parameters:
        shapefile (QgsVectorLayer): Input multipolygon layer
        overlap_threshold (float): Threshold for considering features as duplicates (0.0 to 1.0)
    
    Returns:
        QgsVectorLayer: New layer with duplicates removed
    """
    if not shapefile.isValid():
        print("Invalid input layer")
        return None
    
    # Create output layer with the same structure as input
    # output_layer = QgsVectorLayer("Polygon?crs=" + shapefile.crs().authid(), 
    #                              shapefile.name(), "memory")
    output_layer = QgsVectorLayer(f"Polygon?crs={shapefile.crs().authid()}", shapefile.name(), "memory")
    output_provider = output_layer.dataProvider()
    
    # Copy fields from input layer to output layer
    output_provider.addAttributes(shapefile.fields())
    output_layer.updateFields()
    
    # Identify and remove overlapping features
    features = list(shapefile.getFeatures())
    features_to_keep = []
    
    for i in range(len(features)):
        feature = features[i]
        geom = feature.geometry()
        
        if not geom or geom.isEmpty():
            continue
        
        # Check if this feature overlaps with any already-kept feature
        is_duplicate = False
        for kept_feature in features_to_keep:
            kept_geom = kept_feature.geometry()
            
            if geom.intersects(kept_geom):
                intersection = geom.intersection(kept_geom)
                if not intersection.isEmpty():
                    intersection_area = intersection.area()
                    smaller_area = min(geom.area(), kept_geom.area())
                    
                    if smaller_area > 0:
                        overlap_ratio = intersection_area / smaller_area
                        if overlap_ratio >= overlap_threshold:
                            is_duplicate = True
                            # Keep the larger one
                            if geom.area() > kept_geom.area():
                                features_to_keep.remove(kept_feature)
                                features_to_keep.append(QgsFeature(feature))
                            break
        
        if not is_duplicate:
            features_to_keep.append(QgsFeature(feature))
    
    # Add all features to output layer
    output_provider.addFeatures(features_to_keep)
    
    return output_layer

def reorganise_fields(input_layer, desired_field_order, output_layer_name, prefix, old_id_field='full_id', new_id_field='sub_id'):
    """
    Creates a new vector layer with fields in the specified order and renames the ID field to a sequential prefix_number format. Also handles data type conversions and sets default values for missing fields.
    
    Parameters:
        input_layer (QgsVectorLayer): The input layer with source data
        desired_field_order (list of tuples): List of (field_name, field_type, type_name, length, precision) tuples
        output_layer_name (str): Name for the output layer
        prefix (str): Prefix to use for new IDs
        old_id_field (str): Original ID field name to replace
        new_id_field (str): New ID field name in desired_field_order
    
    Returns:
        QgsVectorLayer: New layer with ordered fields and sequential IDs
    """
    # Create a new memory layer with the correct field order
    # ordered_layer = QgsVectorLayer("Polygon?crs=" + input_layer.crs().authid(), 
    #                          output_layer_name, "memory")
    ordered_layer = QgsVectorLayer(f"Polygon?crs={input_layer.crs().authid()}", output_layer_name, "memory")
    provider = ordered_layer.dataProvider()
    
    # Add fields with explicit types and precision
    fields = []
    for field_info in desired_field_order:
        field_name, field_type, type_name, length, precision = field_info
        fields.append(QgsField(field_name, field_type, type_name, length, precision))
    
    provider.addAttributes(fields)
    ordered_layer.updateFields()
    
    # Get the index of the new ID field in the output layer
    new_id_idx = -1
    for i, field_info in enumerate(desired_field_order):
        if field_info[0] == new_id_field:
            new_id_idx = i
            break
    
    if new_id_idx == -1:
        print(f"Error: {new_id_field} not found in desired_field_order")
        return None
    
    # Copy features with ordered fields and assign sequential IDs
    features = []
    counter = 1
    
    for feat in input_layer.getFeatures():
        new_feat = QgsFeature(ordered_layer.fields())
        new_feat.setGeometry(feat.geometry())
        
        # Create the new ID with sequential numbering
        new_id_value = f"{prefix}_{counter}"
        counter += 1
        
        # Set attributes in the correct order with explicit type handling
        for i, field_info in enumerate(desired_field_order):
            field_name, field_type, _, _, _ = field_info
            
            # Special handling for the new ID field
            if field_name == new_id_field:
                new_feat[i] = new_id_value
                continue
                
            # For all other fields, copy from original if they exist
            if field_name != old_id_field and feat.fieldNameIndex(field_name) != -1:
                value = feat[field_name]
                
                # Handle numeric types explicitly to maintain precision
                if field_type == QVariant.Double and value is not None:
                    value = float(value)  # Ensure it's a float with full precision
                elif field_type == QVariant.Int and value is not None:
                    value = int(value)
                    
                new_feat[i] = value
            else:
                # Set default values for fields that don't exist in the input
                if field_type == QVariant.String:
                    new_feat[i] = ""
                elif field_type == QVariant.Int:
                    new_feat[i] = 0
                elif field_type == QVariant.Double:
                    new_feat[i] = 0.0
                else:
                    new_feat[i] = None
        
        features.append(new_feat)
    
    provider.addFeatures(features)
    return ordered_layer

# 3.1 Processing buildings
#---------------------------------------------------------------------------------------------------------------

def clip_and_filter_buildings(buildings, study_area, base_sewer, house_connection):
    """Clips buildings to study area and filters those within house connection range."""
    clipped = processing.run("native:extractbylocation", {
        'INPUT': buildings,
        'PREDICATE': [0, 4, 5],
        'INTERSECT': study_area,
        'OUTPUT': 'memory:clipped_buildings'
    })['OUTPUT']
    
    nearest = processing.run("native:joinbynearest", {
        'INPUT': clipped,
        'INPUT_2': base_sewer,
        'FIELDS_TO_COPY': [],
        'DISCARD_NONMATCHING': False,
        'PREFIX': '',
        'NEIGHBORS': 1,
        'MAX_DISTANCE': None,
        'OUTPUT': 'memory:nearest_buildings'
    })['OUTPUT']
    
    filtered = processing.run("native:extractbyexpression", {
        'INPUT': nearest,
        'EXPRESSION': f'"distance" < {house_connection}',
        'OUTPUT': 'memory:filtered_buildings'
    })['OUTPUT']
    
    return filtered

def compute_building_attributes(buildings):
    """Computes area, perimeter, and assumes imperviousness of 100%."""
    geom = processing.run("qgis:exportaddgeometrycolumns", {
        'INPUT': buildings,
        'CALC_METHOD': 1,
        'OUTPUT': 'memory:buildings_with_geom'
    })['OUTPUT']
    
    impervious = processing.run("native:fieldcalculator", {
        'INPUT': geom,
        'FIELD_NAME': 'imperv',
        'FIELD_TYPE': 0,
        'FIELD_LENGTH': 5,
        'FIELD_PRECISION': 5,
        'FORMULA': '100',
        'OUTPUT': 'memory:impervious_buildings'
    })['OUTPUT']
    
    return impervious

def connect_buildings_to_sewer(buildings, base_sewer_nodes, threshold):
    """Finds nearest sewer node and selects buildings within threshold distance."""
    connected = processing.run("native:joinbynearest", {
        'INPUT': buildings,
        'INPUT_2': base_sewer_nodes,
        'FIELDS_TO_COPY': ['NODE_ID'],
        'DISCARD_NONMATCHING': False,
        'PREFIX': '',
        'NEIGHBORS': 1,
        'MAX_DISTANCE': None,
        'OUTPUT': 'memory:connected_buildings'
    })['OUTPUT']
    
    processing.run("qgis:selectbyexpression", {
        'INPUT': connected,
        'EXPRESSION': f'"distance" < {threshold}',
        'METHOD': 0
    })
    
    selected = processing.run("native:saveselectedfeatures", {
        'INPUT': connected,
        'OUTPUT': 'memory:selected_buildings'
    })['OUTPUT']
    
    return selected

def process_building_slopes(buildings, dem, slope_limits):
    """Computes slope percentages for buildings using DEM and applies slope limits."""
    slopes = processing.run("native:slope", {
        'INPUT': dem,
        'Z_FACTOR': 1,
        'OUTPUT': 'TEMPORARY_OUTPUT'
    })['OUTPUT']
    
    zonal_stats = processing.run("native:zonalstatisticsfb", {
        'INPUT': buildings,
        'INPUT_RASTER': slopes,
        'RASTER_BAND': 1,
        'COLUMN_PREFIX': '_',
        'STATISTICS': [3],
        'OUTPUT': 'memory:buildings_with_slope'
    })['OUTPUT']
    
    slope_perc = processing.run("native:fieldcalculator", {
        'INPUT': zonal_stats,
        'FIELD_NAME': 'slope_perc',
        'FIELD_TYPE': 0,
        'FIELD_LENGTH': 0,
        'FIELD_PRECISION': 5,
        'FORMULA': 'tan("_median")',
        'OUTPUT': 'memory:buildings_with_slope_perc'
    })['OUTPUT']
    
    final_slope = processing.run("native:fieldcalculator", {
        'INPUT': slope_perc, 
        'FIELD_NAME': 'slope_perc', 
        'FIELD_TYPE': 0,  # Float
        'FIELD_LENGTH': 0, 
        'FIELD_PRECISION': 5, 
        'FORMULA': f'CASE WHEN "slope_perc" < 0 THEN {slope_limits["min_slope_buildings"]} WHEN "slope_perc" > {slope_limits["max_slope_buildings"]} THEN {slope_limits["max_slope_buildings"]} ELSE "slope_perc" END', 
        'OUTPUT': 'memory:final_slope'
    })['OUTPUT']
    
    return final_slope

def compute_building_widths(buildings):
    """Estimates two possible widths assuming rectangular building shapes and ensures correct field order."""
    # Use a higher precision in the field calculator
    width_1 = processing.run("native:fieldcalculator", {
        'INPUT': buildings,
        'FIELD_NAME': 'width_1',
        'FIELD_TYPE': 0,  # 0 is Float
        'FIELD_LENGTH': 20,
        'FIELD_PRECISION': 10,
        'FORMULA': '("perimeter" + sqrt("perimeter"^2 - 8 * "area")) / 4',
        'OUTPUT': 'memory:buildings_with_width_1'
    })['OUTPUT']
    
    final_buildings = processing.run("native:fieldcalculator", {
        'INPUT': width_1,
        'FIELD_NAME': 'width_2',
        'FIELD_TYPE': 0,
        'FIELD_LENGTH': 20,
        'FIELD_PRECISION': 10,
        'FORMULA': '("perimeter" - sqrt("perimeter"^2 - 8 * "area")) / 4',
        'OUTPUT': 'memory:final_buildings'
    })['OUTPUT']
    
    return final_buildings

def process_buildings(buildings, study_area, base_sewer, base_sewer_nodes, dem, slope_limits, threshold, prefix):
    """Main function to process buildings and determine sewer connections.
    
    Parameters:
        buildings (QgsVectorLayer): Input buildings layer
        study_area (QgsVectorLayer): Area to clip buildings to
        base_sewer (QgsVectorLayer): Base sewer network for distance filtering
        base_sewer_nodes (QgsVectorLayer): Sewer nodes for connection analysis
        dem (QgsRasterLayer): Digital elevation model for slope calculations
        slope_limits (dictionary): Min/max slope values
        threshold (float): Distance threshold for sewer connections
        prefix (str): Prefix for sequential ID generation
    
    Returns:
        QgsVectorLayer: New buildings layer connected to nodes
    """
    house_connection = 100
    overlap_threshold = 0.8
    
    filtered_buildings = clip_and_filter_buildings(buildings, study_area, base_sewer, house_connection)
    attributed_buildings = compute_building_attributes(filtered_buildings)
    connected_buildings = connect_buildings_to_sewer(attributed_buildings, base_sewer_nodes, threshold)
    buildings_with_slopes = process_building_slopes(connected_buildings, dem, slope_limits)
    final_buildings = compute_building_widths(buildings_with_slopes)
    
    # First, remove duplicates based purely on geometry
    no_duplicates = remove_duplicate_features(final_buildings, overlap_threshold)
    
    # Define field order for the final output, with sub_id instead of full_id
    field_order = [
        ('sub_id', QVariant.String, 'string', 20, 0),  # Changed from full_id to sub_id
        ('area', QVariant.Double, 'double', 20, 5),
        ('imperv', QVariant.Double, 'double', 20, 5),
        ('NODE_ID', QVariant.String, 'string', 20, 0),
        ('feature_x', QVariant.Double, 'double', 20, 5),
        ('feature_y', QVariant.Double, 'double', 20, 5),
        ('slope_perc', QVariant.Double, 'double', 20, 5),
        ('width_1', QVariant.Double, 'double', 20, 5),
        ('width_2', QVariant.Double, 'double', 20, 5)
    ]
    
    # Then reorganize fields and create sequential IDs
    clean_buildings = reorganise_fields(
        no_duplicates, 
        field_order, 
        'connected_buildings',
        prefix
    )
    
    clean_buildings = processing.run("native:renametablefield", 
                                     {'INPUT':clean_buildings, 
                                      'FIELD':'NODE_ID', 
                                      'NEW_NAME':'Out_NODE', 
                                      'OUTPUT':'memory:'
    })['OUTPUT']
    
    return clean_buildings


# 3.3 Processing streets
#---------------------------------------------------------------------------------------------------------------

def process_street_data(streets_data):
    """Process street data and calculate width based on highway type."""
    # Use the shapefile_to_array function to directly convert to numpy array
    np_links = shapefile_to_array(streets_data)
    
    # Extract column names to find indexes
    df_columns = shapefile_to_dataframe(streets_data).columns
    link_id_index = df_columns.get_loc('LINK_ID')
    highway_type_index = df_columns.get_loc('highway')
    
    # Add width column
    links_width = np.c_[np_links, np.ones(len(np_links))]
    
    # Process width based on highway type
    for i in range(len(np_links)):
        highway_type = str(links_width[i][highway_type_index])
        if highway_type in ['primary', 'secondary']:
            links_width[i][-1] = 14.0
        elif highway_type == 'tertiary':
            links_width[i][-1] = 13.0
        elif highway_type == 'motorway':
            links_width[i][-1] = 21.0
        else:
            links_width[i][-1] = 2.0
    
    # Create DataFrame with just the columns we need
    width_df = pd.DataFrame({
        'field_2': [row[link_id_index] for row in links_width],
        'field_3': links_width[:, -1]
    })
    
    return width_df

def create_street_buffers(streets_data, base_sewer_nodes, dem, slope_limits):
    """Creates buffered street areas based on width, computes geometric attributes, slopes and connects to nodes."""
    links_width_df = process_street_data(streets_data)
    
    slopes = processing.run("native:slope", {
        'INPUT': dem,
        'Z_FACTOR': 1,
        'OUTPUT': 'TEMPORARY_OUTPUT'
    })['OUTPUT']
    
    temp_street_area = processing.run("native:fieldcalculator", {
        'INPUT': streets_data,
        'FIELD_NAME': 'new_id',
        'FIELD_TYPE': 2,  # String type
        'FIELD_LENGTH': 254,  # Increased length
        'FIELD_PRECISION': 0,
        'FORMULA': '"LINK_ID"',
        'OUTPUT': 'memory:temp_street_area'
    })['OUTPUT']
    
    # Create a temporary layer from the DataFrame with explicit field definitions
    # temp_width_layer = QgsVectorLayer("None", "width_layer", "memory")
    temp_width_layer = QgsVectorLayer(f"None?crs={streets_data.crs().authid()}", "width_layer", "memory")
    temp_provider = temp_width_layer.dataProvider()
    
    # Add fields to the temporary layer with explicit types and precision
    temp_provider.addAttributes([
        QgsField("field_2", QVariant.String, "string", 254, 0),
        QgsField("field_3", QVariant.Double, "double", 20, 10)  # High precision
    ])
    temp_width_layer.updateFields()
    
    # Add features to the temporary layer with explicit type handling
    features = []
    for _, row in links_width_df.iterrows():
        feat = QgsFeature()
        feat.setAttributes([
            str(row['field_2']), 
            float(row['field_3'])  # Explicit float conversion
        ])
        features.append(feat)
    
    temp_provider.addFeatures(features)
    temp_width_layer.updateExtents()
    
    # Join attributes with explicit settings
    temp2_street_area = processing.run("native:joinattributestable", {
        'INPUT': temp_street_area,
        'FIELD': 'new_id',
        'INPUT_2': temp_width_layer,
        'FIELD_2': 'field_2',
        'FIELDS_TO_COPY': [],
        'METHOD': 1,
        'DISCARD_NONMATCHING': False,
        'PREFIX': '',
        'OUTPUT': 'memory:temp2_street_area'
    })['OUTPUT']
    
    # Buffer with explicit QgsProperty expression
    temp3_street_area = processing.run("native:buffer", {
        'INPUT': temp2_street_area,
        'DISTANCE': QgsProperty.fromExpression('"field_3" / 2'),
        'SEGMENTS': 5,
        'END_CAP_STYLE': 0,
        'JOIN_STYLE': 0,
        'MITER_LIMIT': 2,
        'DISSOLVE': False,
        'OUTPUT': 'memory:temp3_street_area'
    })['OUTPUT']
    
    # Add geometry columns
    temp4_street_area = processing.run("qgis:exportaddgeometrycolumns", {
        'INPUT': temp3_street_area,
        'CALC_METHOD': 1,
        'OUTPUT': 'memory:temp4_street_area'
    })['OUTPUT']
    
    # Calculate imperviousness with explicit high precision
    temp5_street_area = processing.run("native:fieldcalculator", {
        'INPUT': temp4_street_area,
        'FIELD_NAME': 'imperv',
        'FIELD_TYPE': 0,  # Float
        'FIELD_LENGTH': 20,  # Increased length
        'FIELD_PRECISION': 10,  # Increased precision
        'FORMULA': '100',
        'OUTPUT': 'memory:temp5_street_area'
    })['OUTPUT']
    
    # Join by nearest
    temp6_street_area = processing.run("native:joinbynearest", {
        'INPUT': temp5_street_area,
        'INPUT_2': base_sewer_nodes,
        'FIELDS_TO_COPY': ['NODE_ID'],  # Explicitly copy NODE_ID
        'DISCARD_NONMATCHING': False,
        'PREFIX': '',
        'NEIGHBORS': 1,
        'MAX_DISTANCE': None,
        'OUTPUT': 'memory:temp6_street_area'
    })['OUTPUT']
    
    # Remove duplicates
    temp7_street_area = processing.run("native:removeduplicatesbyattribute", {
        'INPUT': temp6_street_area,
        'FIELDS': ['LINK_ID'],
        'OUTPUT': 'memory:temp7_street_area'
    })['OUTPUT']

    # Zonal statistics
    temp8_street_area = processing.run("native:zonalstatisticsfb", {
        'INPUT': temp7_street_area,
        'INPUT_RASTER': slopes,
        'RASTER_BAND': 1,
        'COLUMN_PREFIX': '_',
        'STATISTICS': [3],
        'OUTPUT': 'memory:temp8_street_area'
    })['OUTPUT']
    
    # Calculate slope with explicit high precision
    temp9_street_area = processing.run("native:fieldcalculator", {
        'INPUT': temp8_street_area,
        'FIELD_NAME': 'slope_perc',
        'FIELD_TYPE': 0,  # Float
        'FIELD_LENGTH': 0,  # Increased length
        'FIELD_PRECISION': 5,  # Increased precision
        'FORMULA': 'tan("_median")',
        'OUTPUT': 'memory:temp9_street_area'
    })['OUTPUT']
    
    # Calculate width_1 with explicit high precision
    temp10_street_area = processing.run("native:fieldcalculator", {
        'INPUT': temp9_street_area,
        'FIELD_NAME': 'width_1',
        'FIELD_TYPE': 0,  # Float
        'FIELD_LENGTH': 20,  # Increased length
        'FIELD_PRECISION': 10,  # Increased precision
        'FORMULA': ' "area" / "field_3" ',
        'OUTPUT': 'memory:temp10_street_area'
    })['OUTPUT']
    
    # Calculate width_2 with explicit high precision
    temp11_street_area = processing.run("native:fieldcalculator", {
        'INPUT': temp10_street_area,
        'FIELD_NAME': 'width_2',
        'FIELD_TYPE': 0,  # Float
        'FIELD_LENGTH': 20,  # Increased length
        'FIELD_PRECISION': 10,  # Increased precision
        'FORMULA': '( "perimeter" /2)- "field_3" ',
        'OUTPUT': 'memory:temp11_street_area'
    })['OUTPUT']
    
    slope_corrected_street_area = processing.run("native:fieldcalculator", {
        'INPUT': temp11_street_area, 
        'FIELD_NAME': 'slope_perc', 
        'FIELD_TYPE': 0,  # Float
        'FIELD_LENGTH': 0, 
        'FIELD_PRECISION': 5, 
        'FORMULA': f'CASE WHEN "slope_perc" < 0 THEN {slope_limits["min_slope_buildings"]} WHEN "slope_perc" > {slope_limits["max_slope_buildings"]} THEN {slope_limits["max_slope_buildings"]} ELSE "slope_perc" END', 
        'OUTPUT': 'memory:final_slope'
    })['OUTPUT']
    
    # Rename field
    renamed_street_areas = processing.run("native:renametablefield", {
        'INPUT': slope_corrected_street_area,
        'FIELD': 'field_3',
        'NEW_NAME': 'osm_width',
        'OUTPUT': 'memory:renamed_street_areas'
    })['OUTPUT']
    
    return renamed_street_areas

def process_streets(connected_streets, base_sewer_nodes, dem, slope_limits, prefix):
    """
    Main function to process streets and determine sewer connections.
    
    Parameters:
        connected_streets (QgsVectorLayer): Input streets layer
        base_sewer_nodes (QgsVectorLayer): Sewer nodes for connection analysis
        dem (QgsRasterLayer): Digital elevation model for slope calculations
        slope_limits (dictionary): Min/max slope values
        prefix (str): Prefix for sequential ID generation
    
    Returns:
        QgsVectorLayer: New streets layer connected to nodes
    
    """
    overlap_threshold=0.8
    
    street_buffers = create_street_buffers(connected_streets, base_sewer_nodes, dem, slope_limits)
    
    # First, remove duplicates based purely on geometry
    no_duplicates = remove_duplicate_features(street_buffers, overlap_threshold)
    
    # Define field order for the final output, with sub_id instead of full_id
    field_order = [
        ('sub_id', QVariant.String, 'string', 20, 0),  # Changed from full_id to sub_id
        ('osm_id', QVariant.String, 'string', 20, 0),
        ('highway', QVariant.String, 'string', 20, 0),
        ('length', QVariant.Double, 'double', 20, 5),
        ('LINK_ID', QVariant.String, 'string', 20, 0),
        ('osm_width', QVariant.Double, 'double', 20, 5),
        ('area', QVariant.Double, 'double', 20, 5),
        ('imperv', QVariant.Double, 'double', 20, 5),
        ('NODE_ID', QVariant.String, 'string', 20, 0),
        ('feature_x', QVariant.Double, 'double', 20, 5),
        ('feature_y', QVariant.Double, 'double', 20, 5),
        ('slope_perc', QVariant.Double, 'double', 20, 5),
        ('width_1', QVariant.Double, 'double', 20, 5),
        ('width_2', QVariant.Double, 'double', 20, 5)
    ]
    
    # Then reorganise fields and create sequential IDs
    clean_street_areas = reorganise_fields(
        no_duplicates, 
        field_order, 
        'connected_street_areas',
        prefix
    )
    
    clean_street_areas = processing.run("native:renametablefield", 
                                     {'INPUT':clean_street_areas, 
                                      'FIELD':'NODE_ID', 
                                      'NEW_NAME':'Out_NODE', 
                                      'OUTPUT':'memory:'
    })['OUTPUT']
    
    return clean_street_areas


# 3.4 Determine dry weather flow
#---------------------------------------------------------------------------------------------------------------

def process_population(pop_density, connected_buildings, annual_growth, years):
    """
    Process population density data if the system is combined.
    
    Parameters:
        pop_density (QgsVectorLayer): Layer containing population density information
        connected_buildings (QgsVectorLayer): Layer with connected buildings
        annual_growth (float): Annual population growth rate as percentage
        years (int): Number of years for population projection
    
    Returns:
        QgsVectorLayer: Processed population density layer if combined system
    """
    if not pop_density or not pop_density.isValid():
        print("Population density layer is not valid or not provided")
        return None
    
    # Update population with linear growth assumption
    updated_pop = processing.run("native:fieldcalculator", {
        'INPUT': pop_density,
        'FIELD_NAME': 'update_pop',
        'FIELD_TYPE': 1,
        'FIELD_LENGTH': 0,
        'FIELD_PRECISION': 0,
        'FORMULA': f'"Einwohner" * {1 + ((annual_growth/100) * years)}',
        'OUTPUT': 'memory:updated_population'
    })['OUTPUT']
    
    # Join population data with connected buildings using nearest neighbor
    pop_with_nodes = processing.run("native:joinbynearest", {
        'INPUT': updated_pop,
        'INPUT_2': connected_buildings,
        'FIELDS_TO_COPY': ['Out_NODE'],
        'DISCARD_NONMATCHING': False,
        'PREFIX': '',
        'NEIGHBORS': 1,
        'MAX_DISTANCE': None,
        'OUTPUT': 'memory:population_with_nodes'
    })['OUTPUT']
    
    # Keep only necessary fields
    final_pop_density = processing.run("native:retainfields", {
        'INPUT': pop_with_nodes,
        'FIELDS': ['Gitter_ID_', 'update_pop', 'Out_NODE', 'distance'],
        'OUTPUT': 'memory:processed_population'
    })['OUTPUT']
    
    return final_pop_density


# 4. Input Files for Design
#---------------------------------------------------------------------------------------------------------------

# 4.2 Pipes
#---------------------------------------------------------------------------------------------------------------

def process_pipe_data(np_links, np_nodes, output_csv):
    """
    Process pipe data to include length and slope calculations.
    
    Parameters:
        np_links (numpy.ndarray): Array containing pipe information
        np_nodes (numpy.ndarray): Array containing node information
        output_csv (str): Path to save the processed pipe data
    
    Returns:
        numpy.ndarray: Processed pipe data with columns [LINK_ID, FROM_NODE, TO_NODE, length, slope]
    """
    # Initialize array with additional columns for length and slope
    pipes4design = np.c_[np_links, np.ones(len(np_links)), np.ones(len(np_links))]
    
    # Calculate length and slope for each pipe
    for i in range(len(pipes4design)):
        pipes4design[i, 8] = round(pipes4design[i, 4], 3)
        
        start_node = np.where(np_nodes[:, 0] == pipes4design[i, 6])[0][0]
        end_node = np.where(np_nodes[:, 0] == pipes4design[i, 7])[0][0]
        
        # Calculate slope using single indices
        elevation_diff = np_nodes[start_node, 4] - np_nodes[end_node, 4]
        slope = elevation_diff / pipes4design[i, 8]
        pipes4design[i, 9] = round(100 * slope, 3)
    
    # Remove unnecessary columns and keep only [ID, FROM_NODE, TO_NODE, length, slope]
    pipes4design = np.delete(pipes4design, np.s_[0:5], 1)
    
    # Save to CSV
    pd.DataFrame(pipes4design).to_csv(output_csv, index=False)
    
    return pipes4design


# 4.3 Areas
#---------------------------------------------------------------------------------------------------------------

def process_subcatchments(np_buildings, np_streets, slope_limits, combined=False, np_pop=None):
    """
    Process subcatchment areas, calculating total areas, imperviousness and slopes.

    Parameters:
        np_buildings (numpy.ndarray): Columns including [area, slope, outlet ID, ...].
        np_streets (numpy.ndarray): Columns including [area, slope, outlet ID, ...].
        slope_limits (dictionary): Min/max slope limits
        combined (bool): Whether to include population density in calculations
        np_pop (numpy.ndarray): Optional, population data, required if combined True
    
    Returns:
        numpy.ndarray: Data for subcatchment design
    """
    # Create initial vector of outlets
    out_building = np_buildings[:,3]
    out_street = np_streets[:,8]
    temp = np.concatenate([out_building, out_street])
    outlets = np.unique(temp)
    
    # Create an array to identify and store the ID of outlet, which buildings and streets discharge into
    subcatch = np.c_[outlets, np.zeros(len(outlets)) , np.zeros(len(outlets)), np.zeros(len(outlets)), np.zeros(len(outlets)) ]  
    
    for i in range(0,len(outlets)):
        # Find which and how many buildings and streets are connected to a specific outlet
        p_bu = np.where(np_buildings[:,3]==subcatch[i,0])
        p_str = np.where(np_streets[:,8]==subcatch[i,0])
    
        area_buildings = 0
        slope_buildings = 0
        #imperv_build = 0
        area_street = 0
        slope_street = 0
        #imperv_street = 0
        
        # Calculate total areas from buildings and streets and mean slopes and imperviousness
        if p_bu[0].size > 0:
            for j in range(0, len(p_bu[0])):
                area_buildings = area_buildings + np_buildings[p_bu[0][j],1]
                if np_buildings[p_bu[0][j],7] <= 0:
                    temp_slope = 0.01
                    slope_buildings = slope_buildings + temp_slope
                else:
                    slope_buildings = slope_buildings + np_buildings[p_bu[0][j],6]
        
        if p_str[0].size > 0:
            for j in range(0, len(p_str[0])):
                area_street = area_street + np_streets[p_str[0][j],6]
                if np_streets[p_str[0][j],11] <= 0:
                    temp_slope = 0.01
                    slope_street = slope_street + temp_slope
                else:
                    slope_street = slope_street + np_streets[p_str[0][j],11]
        
        # Calculate final values
        total_area = (area_buildings + area_street) / 10000  # To hectares
        total_slope = slope_buildings + slope_street
        total_count = p_bu[0].size + p_str[0].size
        subcatch[i, 1] = round(total_area, 3)
        subcatch[i, 2] = 100
        if total_count > 0:
            subcatch[i, 3] = round(total_slope / total_count, 3)
        else:
            subcatch[i, 3] = 0
        
        if subcatch[i,3] > slope_limits['max_slope_buildings']:
             subcatch[i,3]= (slope_limits['max_slope_buildings'] +
                                     slope_limits['max_slope_streets']) / 2
    
    # Process population density if combined system
    if combined and np_pop is not None:
        for i in range(len(outlets)):
            p_pop = np.where(np_pop[:, 2] == subcatch[i, 0])[0]
            if len(p_pop) > 0:
                subcatch[i, 4] = np.sum(np_pop[p_pop, 1])
         
    return subcatch


# 4.4 Save designed pipes and buildings
#---------------------------------------------------------------------------------------------------------------

def design_data(node_data, pipe_data, subcatchment_data, output_folder, save_csv=False):
    """
    Processes all design data into appropriate formates, optionally saves them as CSV files.
    
    Parameters:
        node_data (numpy.ndarray): Existing node data
        pipe_data (numpy.ndarray): Processed pipe data
        subcatchment_data (numpy.ndarray): Processed subcatchment data
        output_folder (str): Folder path to save the CSV files
        save_csv (bool): Optional, whether to save as CSV files
    
    Returns:
        dataframes: The final design data for nodes, pipes (links) and subcatchments
    """
    node_csv = os.path.join(output_folder, 'node_data_design.csv')
    pipe_csv = os.path.join(output_folder, 'pipe_data_design.csv')
    subcatch_csv = os.path.join(output_folder, 'subcatchment_data_design.csv')
    
    # Convert column names to strings explicitly
    node_columns = ['NODE_ID','x_coord','y_coord','Cover_Elevation','Inv_Elevation','Depth']
    pipe_columns = ['LINK_ID','FROM_NODE','TO_NODE','length','slope']
    subcatch_columns = ['NODE_OUTLET','AREA','%imp','slope','population']
    
    # Create DataFrames with string column names
    node_df = pd.DataFrame(node_data, columns=node_columns)
    pipe_df = pd.DataFrame(pipe_data, columns=pipe_columns)
    subcatch_df = pd.DataFrame(subcatchment_data, columns=subcatch_columns)
    
    if save_csv == True:
        node_df.to_csv(node_csv)
        pipe_df.to_csv(pipe_csv)
        subcatch_df.to_csv(subcatch_csv)
    
    return node_df, pipe_df, subcatch_df
