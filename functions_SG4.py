# -*- coding: utf-8 -*-
"""
Created on Fri May 23 19:46:36 2025

@author: Sifat
"""

from qgis.core import QgsVectorLayer, QgsField, QgsFields, QgsFeature, QgsFeatureRequest, QgsGeometry
from qgis import processing
from PyQt5.QtCore import QVariant
import numpy as np
import pandas as pd

def extract_skipped_streets(all_streets, MST_streets):
    """
    Extract the skipped streets from all streets, using the slected streets from minimum spanning tree or MST.
    
    Parameters:
        all_streets (QgsVectorLayer): The shapefile with all streets within the study area
        MST_streets (QgsVectorLayer): The shapefile with the selected streets after MST
    
    Returns:
        QgsVectorLayer: Layer with all the skipped streets
    """
    desired_fields = ['full_id', 'osm_id', 'osm_type', 'highway', 'length', 'LINK_ID']
    
    difference = processing.run("native:difference", {
        'INPUT': all_streets,
        'OVERLAY': MST_streets,
        'OUTPUT': 'memory:'
    })['OUTPUT']
    
    difference_with_fields = processing.run("native:retainfields", {
        'INPUT': difference,
        'FIELDS': desired_fields,
        'OUTPUT': 'memory:'
    })['OUTPUT']
    
    return difference_with_fields
       
def correct_node_elevations(streets_layer, nodes_layer):
    """
    Corrects street node ordering based on elevation and calculates elevation difference.
    Ensures FROM_NODE has higher elevation than TO_NODE by swapping nodes when necessary.
    
    Parameters:
        streets_layer (QgsVectorLayer): Streets layer with FROM_node and TO_node
        nodes_layer (QgsVectorLayer): Nodes layer containing NODE_ID and INVERT_ELV data
        
    Returns:
        QgsVectorLayer: New memory layer with corrected node ordering and elevation difference
    """
    # Step 1: Create a dictionary mapping NODE_ID to INVERT_ELV
    node_elevations = {}
    for feature in nodes_layer.getFeatures():
        node_id = feature['NODE_ID']
        elev = feature['INVERT_ELV']
        node_elevations[node_id] = elev
    
    # Create a new memory layer to store results
    # Get the CRS and fields from the original streets layer
    crs = streets_layer.crs()
    fields = streets_layer.fields()
    
    # Create new fields list without FROM_node and TO_node (which will be renamed)
    new_fields = QgsFields()
    for field in fields:
        if field.name() == 'FROM_node':
            new_fields.append(QgsField('FROM_NODE', field.type(), field.typeName(), field.length(), field.precision()))
        elif field.name() == 'TO_node':
            new_fields.append(QgsField('TO_NODE', field.type(), field.typeName(), field.length(), field.precision()))
        else:
            new_fields.append(field)
    
    # Add new fields for elevations and difference
    new_fields.append(QgsField('FROM_INVERT_ELV', QVariant.Double, 'double', 10, 3))
    new_fields.append(QgsField('TO_INVERT_ELV', QVariant.Double, 'double', 10, 3))
    new_fields.append(QgsField('Elev_Diff', QVariant.Double, 'double', 10, 3))
    
    # Use new_fields instead of original fields for renaming
    fields = new_fields
    new_layer = QgsVectorLayer(f'LineString?crs={crs.authid()}', 'Corrected_Streets', 'memory')
    new_provider = new_layer.dataProvider()
    new_provider.addAttributes(fields)
    new_layer.updateFields()
    
    # Process each street feature
    features_to_add = []
    for street in streets_layer.getFeatures():
        new_feature = QgsFeature(fields)
        # Copy the original geometry and attributes
        new_feature.setGeometry(street.geometry())
        for i, field in enumerate(street.fields()):
            if i < len(street.attributes()):
                if field.name() == 'FROM_node':
                    new_feature['FROM_NODE'] = street.attributes()[i]
                elif field.name() == 'TO_node':
                    new_feature['TO_NODE'] = street.attributes()[i]
                elif i < len(fields):
                    new_feature.setAttribute(i, street.attributes()[i])
        
        # Get original FROM and TO nodes
        from_node = street['FROM_node']
        to_node = street['TO_node']
        
        # Get elevations from our dictionary (with safety checks)
        from_elev = node_elevations.get(from_node)
        to_elev = node_elevations.get(to_node)
        
        # Add the elevation values to the feature
        new_feature['from_INVERT_ELV'] = from_elev
        new_feature['to_INVERT_ELV'] = to_elev
        
        # If both elevations are available, ensure FROM node is higher than TO node
        if from_elev is not None and to_elev is not None:
            # Calculate elevation difference
            elev_diff = abs(from_elev - to_elev)
            new_feature['Elev_Diff'] = elev_diff
            
            # Swap nodes if needed (if FROM elevation is lower than TO elevation)
            if from_elev < to_elev:
                new_feature['FROM_NODE'] = to_node
                new_feature['TO_NODE'] = from_node
                # Update the elevation fields to match the new nodes
                new_feature['FROM_INVERT_ELV'] = to_elev
                new_feature['TO_INVERT_ELV'] = from_elev
        else:
            # Handle missing elevation data
            new_feature['Elev_Diff'] = None
        
        features_to_add.append(new_feature)
    
    # Add all features to the new layer
    new_provider.addFeatures(features_to_add)
    
    return new_layer

def add_further_data_to_links(difference, pipes_df, head_df):
    """
    Adds aggregated slope, Q (total flow), diameter and ishead flag from two dataframes to a shapefile.
    Creates node-based aggregation for Q (sum), diameters (minimum value) and slope Elev_Diff/length.
    Marks head nodes based on FROM_NODE entries in head_df.
    
    Parameters:
        difference (QgsVectorLayer): The shapefile with FROM_NODE, Elev_Diff, and length fields
        pipes_df (pandas.DataFrame): Dataframe with FROM_NODE, TO_NODE, Qtot, and Diameter columns
        head_df (pandas.DataFrame): Dataframe with FROM_NODE, TO_NODE, Qtot, and Diameter columns
    
    Returns:
        QgsVectorLayer: A copy of difference layer with added slope, Q, Diameter, and isHead fields
    """
    # Step 1: Get all unique node IDs from both dataframes
    all_nodes = set()
    
    # Add FROM_NODE and TO_NODE nodes from pipes_df
    if 'FROM_NODE' in pipes_df.columns and 'TO_NODE' in pipes_df.columns:
        all_nodes.update(pipes_df['FROM_NODE'].astype(str).tolist())
        all_nodes.update(pipes_df['TO_NODE'].astype(str).tolist())
    
    # Add FROM_NODE and TO_NODE nodes from head_df
    if 'FROM_NODE' in head_df.columns and 'TO_NODE' in head_df.columns:
        all_nodes.update(head_df['FROM_NODE'].astype(str).tolist())
        all_nodes.update(head_df['TO_NODE'].astype(str).tolist())
    
    # Step 2: Create aggregated lookup dictionary for each unique node
    node_data = {node: {'Qtot': [], 'Diameter': [], 'ishead': 0} for node in all_nodes}
    
    # Process pipes_df
    for _, row in pipes_df.iterrows():
        # Collect D values for both FROM_NODE and TO_NODE nodes
        for node_type in ['FROM_NODE', 'TO_NODE']:
            if node_type in row:
                node = str(row[node_type])
                
                if node in node_data:
                    # Add D values if they exist
                    if 'Diameter' in row and row['Diameter'] is not None and pd.notna(row['Diameter']):
                        node_data[node]['Diameter'].append(row['Diameter'])
        
        # Add Qtot, associated with both FROM_NODE and TO_NODE nodes
        if 'Qtot' in row and row['Qtot'] is not None and pd.notna(row['Qtot']):
            qtot_value = row['Qtot']
            
            if 'FROM_NODE' in row and str(row['FROM_NODE']) in node_data:
                node_data[str(row['FROM_NODE'])]['Qtot'].append(qtot_value)
            
            if 'TO_NODE' in row and str(row['TO_NODE']) in node_data:
                node_data[str(row['TO_NODE'])]['Qtot'].append(qtot_value)
    
    # Process head_df and mark head nodes
    head_from_nodes = set()
    if 'FROM_NODE' in head_df.columns:
        head_from_nodes = set(head_df['FROM_NODE'].astype(str).tolist())
    
    for _, row in head_df.iterrows():
        # Collect D values for both FROM_NODE and TO_NODE nodes
        for node_type in ['FROM_NODE', 'TO_NODE']:
            if node_type in row:
                node = str(row[node_type])
                
                if node in node_data:
                    # Add D values if they exist
                    if 'Diameter' in row and row['Diameter'] is not None and pd.notna(row['Diameter']):
                        node_data[node]['Diameter'].append(row['Diameter'])
                    
                    # Mark as head if it's in the FROM_NODE list of head_df
                    if node in head_from_nodes:
                        node_data[node]['ishead'] = 1
        
        # Add Qtot - this should be associated with both FROM and TO nodes
        if 'Qtot' in row and row['Qtot'] is not None and pd.notna(row['Qtot']):
            qtot_value = row['Qtot']
            
            if 'FROM_NODE' in row and str(row['FROM_NODE']) in node_data:
                node_data[str(row['FROM_NODE'])]['Qtot'].append(qtot_value)
            
            if 'TO_NODE' in row and str(row['TO_NODE']) in node_data:
                node_data[str(row['TO_NODE'])]['Qtot'].append(qtot_value)
    
    # Step 3: Apply aggregation rules for each node
    for node, data in node_data.items():
        # For D: use minimum value or 0.1
        data['Diameter'] = min(data['Diameter']) if data['Diameter'] else 0.1
        
        # For Qtot: sum all values or 0
        data['Qtot'] = sum(data['Qtot']) if data['Qtot'] else 0
        
    # Step 4: Create a layer with same geometry type and CRS
    result_layer = QgsVectorLayer(f"LineString?crs={difference.crs().authid()}", "upstream_data_result", "memory")
    
    # Copy fields from difference layer
    provider = result_layer.dataProvider()
    provider.addAttributes(difference.fields())
    
    # Add new fields
    provider.addAttributes([
        QgsField("slope", QVariant.Double),
        QgsField("Q", QVariant.Double),
        QgsField("Diameter", QVariant.Double),
        QgsField("isHead", QVariant.Int)
    ])
    
    result_layer.updateFields()
    
    # Process each feature in the difference layer
    features = []
    for feature in difference.getFeatures():
        new_feature = QgsFeature(result_layer.fields())
        
        # Copy attributes and geometry
        new_feature.setAttributes(feature.attributes())
        new_feature.setGeometry(QgsGeometry(feature.geometry()))
        
        # Get FROM node ID and convert to string for consistent comparison
        from_node = str(feature["FROM_NODE"])
        
        # Calculate slope using Elev_Diff / length and convert to percentage
        length = feature["length"] if feature.fieldNameIndex("length") != -1 else 0
        elev_diff = feature["Elev_Diff"] if feature.fieldNameIndex("Elev_Diff") != -1 else 0
        
        # Calculates slope in percentage with default being 0.3% if length is zero or Elev_Diff is missing
        slope = (elev_diff / length) * 100 if length > 0 and elev_diff is not None else 0.3
        
        # Get upstream data for the FROM node
        upstream_data = node_data.get(from_node, {'Qtot': 0, 'Diameter': 0.1, 'ishead': 0})
        qtot = upstream_data['Qtot']
        diameter = upstream_data['Diameter']
        ishead = upstream_data['ishead']
        
        # Add upstream data to the feature
        attributes = new_feature.attributes()
        attributes.append(slope)
        attributes.append(qtot)
        attributes.append(diameter)
        attributes.append(ishead)
        new_feature.setAttributes(attributes)
        
        features.append(new_feature)
    
    # Add features to the result layer
    provider.addFeatures(features)
    result_layer.updateExtents()
    
    return result_layer

def sort_features(layer):
    """
    Sort features in a shapefile based on weighted scores of normalised length and Q values.
    Features with isHead=0 are ranked before isHead=1, maintaining score order within each group.
    Adds weighted_score and final_rank fields.
    
    Parameters:
        layer (QgsVectorLayer): The input line layer with length, isHead and Q fields
    
    Returns:
        QgsVectorLayer: The same layer with added final_rank field and removed intermediate fields
    """
    # Check if the required fields exist
    field_names = [field.name() for field in layer.fields()]
    required_fields = ['length', 'isHead', 'Q']
    for field in required_fields:
        if field not in field_names:
            raise ValueError(f"Required field '{field}' not found in the layer.")
    
    # Add new fields if they don't exist already
    if 'weighted_score' not in field_names:
        layer.dataProvider().addAttributes([QgsField('weighted_score', QVariant.Double)])
    if 'final_rank' not in field_names:
        layer.dataProvider().addAttributes([QgsField('final_rank', QVariant.Int)])
    
    layer.updateFields()
    
    # Get all features
    features = list(layer.getFeatures())
    
    # Find maximum values for normalization
    max_length = max(feature['length'] for feature in features)
    max_q = max(feature['Q'] for feature in features)
    
    # Calculate normalized values and scores for each feature
    feature_data = []
    for feat in features:
        # Normalize values (higher is better)
        norm_length = 1 - (feat['length'] / max_length) if max_length > 0 else 0
        norm_q = feat['Q'] / max_q if max_q > 0 else 0
        
        # Calculate weighted score
        weighted_score = 0.5 * norm_length + 0.5 * norm_q
        
        # Store feature and its calculated values
        feature_data.append({
            'feature': feat,
            'weighted_score': weighted_score,
            'isHead': feat['isHead'],
            'fid': feat.id()
        })
    
    # Sort features by isHead (0 first, then 1) and within each group by weighted score (descending)
    feature_data.sort(key=lambda x: (x['isHead'], -x['weighted_score']))
    
    # Assign final ranks
    for rank, data in enumerate(feature_data, 1):
        data['final_rank'] = rank
    
    # Update the layer with new values
    layer.startEditing()
    
    for data in feature_data:
        feat = data['feature']
        fid = data['fid']
        
        # Get field indices
        weighted_score_idx = layer.fields().indexOf('weighted_score')
        final_rank_idx = layer.fields().indexOf('final_rank')
        
        # Update attribute values
        layer.changeAttributeValue(fid, weighted_score_idx, data['weighted_score'])
        layer.changeAttributeValue(fid, final_rank_idx, data['final_rank'])
    
    layer.commitChanges()
    
    # Remove the specified fields
    fields_to_drop = ['FROM_INVERT_ELV', 'TO_INVERT_ELV', 'Elev_Diff', 'Q', 'isHead', 'weighted_score']
    field_indices = [layer.fields().indexOf(field) for field in fields_to_drop if field in field_names]
    
    if field_indices:
        layer.dataProvider().deleteAttributes(field_indices)
        layer.updateFields()
    
    return layer

def add_skipped_links_to_designed(designed_layer, skipped_layer, meshness_perc):
    """
    Merge top-ranked features from skipped shapefile into designed shapefile based on meshness percentage.
    Creates a final layer with only essential fields in specified order.
    
    Parameters:
        designed_layer (QgsVectorLayer): The designed vector layer
        skipped_layer (QgsVectorLayer): The skipped vector layer with final_rank field
        meshness_perc (int): Percentage of skipped features to add (1-100)
    
    Returns:
        QgsVectorLayer: New vector layer with merged features
    """
    
    # Common setup: Create a memory layer with the same fields and geometry type as the designed layer
    result_layer = QgsVectorLayer(f"LineString?crs={designed_layer.crs().authid()}", "merged_layer", "memory")
    result_provider = result_layer.dataProvider()
        
    # Add fields from designed layer
    result_provider.addAttributes(designed_layer.fields())
    result_layer.updateFields()
    
    # Copy all features from designed layer
    designed_features = list(designed_layer.getFeatures())
    result_provider.addFeatures(designed_features)
    
    # Conditional: Add skipped features only if skipped_layer is available as an argument
    if skipped_layer is not None:
        # Verify that final_rank exists in skipped
        skipped_fields = [field.name() for field in skipped_layer.fields()]
        if 'final_rank' not in skipped_fields:
            raise ValueError("The 'final_rank' field is missing from the skipped layer")
        
        # Get fields from both layers and find common fields
        designed_fields = [field.name() for field in designed_layer.fields()]
        common_fields = [field for field in skipped_fields if field in designed_fields]
        
        # Calculate number of features to add based on meshness_perc
        total_skipped_features = skipped_layer.featureCount()
        features_to_add_count = int(np.floor(total_skipped_features * meshness_perc / 100))
        
        # Get features from skipped layer, ordered by final_rank
        request = QgsFeatureRequest()
        request.addOrderBy('final_rank')  # Order by final_rank ascending
        skipped_features = list(skipped_layer.getFeatures(request))
        
        # Take only the features we need based on the calculated count
        features_to_add = skipped_features[:features_to_add_count]
        
        # Add selected features to the result layer
        for feature in features_to_add:
            new_feature = QgsFeature(result_layer.fields())
            
            # Copy geometry
            new_feature.setGeometry(feature.geometry())
            
            # Set attributes for common fields
            for field_name in common_fields:
                result_field_index = result_layer.fields().indexOf(field_name)
                skipped_field_index = skipped_layer.fields().indexOf(field_name)
                
                if result_field_index >= 0 and skipped_field_index >= 0:
                    new_feature.setAttribute(result_field_index, feature[skipped_field_index])
            
            result_provider.addFeature(new_feature)
    
    # Define ordered fields to keep
    ordered_fields_to_keep = ['LINK_ID', 'length', 'FROM_NODE', 'TO_NODE', 'slope', 'Diameter']
    
    # Create new layer with ordered fields
    final_layer = QgsVectorLayer(f"LineString?crs={designed_layer.crs().authid()}", "final_layer", "memory")
    final_provider = final_layer.dataProvider()
    
    # Add fields in the specified order
    fields_to_add = []
    for field_name in ordered_fields_to_keep:
        original_field = result_layer.fields().field(field_name)
        if original_field:
            fields_to_add.append(original_field)
    
    final_provider.addAttributes(fields_to_add)
    final_layer.updateFields()
    
    # Copy features with only the ordered fields
    for feature in result_layer.getFeatures():
        new_feature = QgsFeature(final_layer.fields())
        new_feature.setGeometry(feature.geometry())
        
        for i, field_name in enumerate(ordered_fields_to_keep):
            old_index = result_layer.fields().indexOf(field_name)
            if old_index >= 0:
                new_feature.setAttribute(i, feature[old_index])
        
        final_provider.addFeature(new_feature)
    
    final_layer.updateExtents()
    
    return final_layer