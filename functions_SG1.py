# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 14:46:43 2024

@author: Diego
"""

from qgis.core import QgsCoordinateReferenceSystem, QgsVectorLayer, QgsField, QgsFeature, QgsSettings, QgsFields 
from qgis import processing
from PyQt5.QtCore import QVariant
import pandas as pd
import networkx as nx
import numpy as np

settings = QgsSettings()
settings.setValue("Processing/OutputFormats/GeoJSON/enabled", True)

'''Base functions'''
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

# Function to get the attribute table from a shapefile and turn it into a pandas df
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

    return df

# Function to save any shapefile into disk
def save_to_disk(vector_layer, output_path):
    processing.run("native:savefeatures", {
        'INPUT': vector_layer,
        'OUTPUT': output_path,
        'LAYER_NAME': '',
        'DATASOURCE_OPTIONS': '',
        'LAYER_OPTIONS': ''
    })
    
# Function to load a pandas df into QGIS as a table, without adding it to the map
def turn_dataframe_to_qgis(df, layer_name="Pandas Layer"):
    """ Adds a Pandas DataFrame as a QGIS layer, treating it like a CSV. """
    # Ensure column names are strings
    df.columns = df.columns.astype(str)  

    # Determine field types
    fields = QgsFields()
    for col in df.columns:
        dtype = df[col].dtype
        if dtype == 'int64':
            field_type = QVariant.Int
        elif dtype == 'float64':
            field_type = QVariant.Double
        else:
            field_type = QVariant.String  # Default to string for other types
        fields.append(QgsField(col, field_type))

    # Create a memory layer (like loading a CSV into QGIS)
    layer = QgsVectorLayer("None", f'{layer_name}', "memory")
    provider = layer.dataProvider()
    provider.addAttributes(fields)
    layer.updateFields()

    # Add features (rows)
    features = []
    for _, row in df.iterrows():
        feature = QgsFeature()
        feature.setAttributes(list(row))
        features.append(feature)

    provider.addFeatures(features)
    layer.updateExtents()
    
    return layer

def clip_streets(streets,study_area,out):
    '''Function to clip inputs to study area and move the outlet to the 
    intersection of the study area and and closest street'''
    filtered_streets = processing.run("native:extractbyexpression", 
        {'INPUT': streets,
         'EXPRESSION': ('"highway" IN (\'trunk\', \'primary\', \'secondary\', \'tertiary\', '
                '\'unclassified\', \'residential\', \'service\', \'track\', \'cycleway\')'),
         'OUTPUT': 'memory:filtered_streets'  # Unique in-memory layer name
        }
    )['OUTPUT']
    
    # Clip streets to the study area
    clipped_streets = processing.run(
        "native:clip", 
        {
            'INPUT': filtered_streets,
            'OVERLAY': study_area,
            'OUTPUT': 'memory:clipped_streets'  # Unique in-memory layer name
        }
    )['OUTPUT']
    
    # Split clipped streets at intersections
    split_streets = processing.run(
        "native:splitwithlines", 
        {
            'INPUT': clipped_streets,
            'LINES': clipped_streets,
            'OUTPUT': 'memory:split_streets'  # Unique in-memory layer name
        }
    )['OUTPUT']
    
    clip_2 = processing.run("native:retainfields", {
            'INPUT': split_streets,
            'FIELDS': ['full_id', 'osm_id', 'osm_type', 'highway'],
            'OUTPUT':  'memory:clip_2'})['OUTPUT']
    
    TEMP_snap = processing.run("native:snapgeometries", 
                       {'INPUT':out,
                        'REFERENCE_LAYER':clip_2,
                        'TOLERANCE':25,'BEHAVIOR':0,
                        'OUTPUT':'memory:TEMP_snap'})['OUTPUT']
        
    #add XY coordinates to snap outlet
    snap = processing.run("qgis:exportaddgeometrycolumns", 
                   {'INPUT': TEMP_snap,
                    'CALC_METHOD':0,
                    'OUTPUT': 'memory:snap_outlet'})['OUTPUT']


    #remove unconnected streets to outlet
    #as a function of travel cost. This is calculated as a function of the number of features in the clip_street
    travel_cost= clip_2.featureCount()*100
    
    connected_area = processing.run("native:serviceareafromlayer", 
                                    {'INPUT':clip_2,
                    'STRATEGY':0,'DIRECTION_FIELD':'','VALUE_FORWARD':'',
                    'VALUE_BACKWARD':'','VALUE_BOTH':'',
                    'DEFAULT_DIRECTION':2,'SPEED_FIELD':'',
                    'DEFAULT_SPEED':50,'TOLERANCE':0,
                    'START_POINTS': snap,
                    'TRAVEL_COST2':travel_cost,
                    'INCLUDE_BOUNDS':False,
                    'OUTPUT_LINES':'memory:connected_area'})['OUTPUT_LINES']

    #select streets in the connected area
    processing.run("native:selectbylocation", 
                   {'INPUT': clip_2,
                    'PREDICATE':[0,1,4,5],
                    'INTERSECT': connected_area,
                    'METHOD':0})
    
    TEMP_connected_streets = processing.run("native:saveselectedfeatures",
                   {'INPUT':clip_2,
                    'OUTPUT': 'memory:TEMP_connected_streets'})['OUTPUT']
    
    return snap , TEMP_connected_streets

def vertex_extraction(temp_connected_streets, elev_model, depth_val, snap):
    '''Function to select streets that are connected to the system, and that 
    will therefore have pipes underneath. 
    Extract vertices of streets as locations for nodes,
    add elevation and invert elevation (wih min depth), to nodes
    Assign  attributes to outlet and identify its location within the system'''

    # Extract start and end nodes 
    t_street_node = processing.run("native:extractspecificvertices", 
                   {'INPUT': temp_connected_streets,
                    'VERTICES':'0, -1',
                    'OUTPUT': 'memory:'})['OUTPUT']
    
    t_street_node = processing.run("native:deleteduplicategeometries", 
                   {'INPUT':t_street_node,
                    'OUTPUT':'memory:'})['OUTPUT']

    
    # Add unique ID to nodes as auto-incremental field
    t_street_node = processing.run("native:addautoincrementalfield", {
        'INPUT': t_street_node,
        'FIELD_NAME': 'TEMP_NUM',
        'START': 1000,
        'GROUP_FIELDS': [],
        'SORT_EXPRESSION': '',
        'SORT_ASCENDING': True,
        'SORT_NULLS_FIRST': False,
        'OUTPUT': 'memory:'
    })['OUTPUT']
    
    # Create string field referencing the temp field
    t_street_node = processing.run("native:fieldcalculator", {
        'INPUT': t_street_node,
        'FIELD_NAME': 'NODE_ID', 
        'FIELD_TYPE': 2,
        'FIELD_LENGTH': 20,
        'FIELD_PRECISION': 0,
        'NEW_FIELD': True,
        'FORMULA': 'concat(\'ND_\', "TEMP_NUM")',
        'OUTPUT': 'memory:'
    })['OUTPUT']
    
    t_street_node = processing.run("native:addxyfields", 
           {'INPUT': t_street_node,
            'CRS': t_street_node.crs(),
            'PREFIX':'',
            'OUTPUT': 'memory:'})['OUTPUT']
    
    # Add invert elevation and XY data 
    t_street_node= processing.run("qgis:rastersampling", 
                   {'INPUT': t_street_node,
                    'RASTERCOPY': elev_model,
                    'COLUMN_PREFIX': 'VALUE_',
                    'OUTPUT': 'memory:'})['OUTPUT']

    # Compute invert_elev
    connected_streets_nodes = processing.run("native:fieldcalculator", 
                   {'INPUT': t_street_node,
                    'FIELD_NAME': 'invert_elev',
                    'FIELD_TYPE': 0,  # Float
                    'FIELD_LENGTH': 10,
                    'FIELD_PRECISION': 2,
                    'FORMULA': ' "VALUE_1" - ' + str(depth_val),
                    'OUTPUT':  'memory:'})['OUTPUT']
    
    
    connected_streets_nodes = processing.run("native:renametablefield", 
                   {'INPUT':connected_streets_nodes,
                    'FIELD':'VALUE_1',
                    'NEW_NAME':'Cover_Elevation',
                    'OUTPUT':'memory:'})['OUTPUT']
    fields_to_keep = {'NODE_ID', 'x', 'y', 'invert_elev','Cover_Elevation'}

    # Collect indices of fields to delete
    fields_to_delete = [
        connected_streets_nodes.fields().indexFromName(field.name())
        for field in connected_streets_nodes.fields()
        if field.name() not in fields_to_keep
    ]
    

    # Delete all unwanted fields in one operation
    if fields_to_delete:
        connected_streets_nodes.dataProvider().deleteAttributes(fields_to_delete)
        connected_streets_nodes.updateFields()
        
    outlet_final = processing.run("native:joinattributesbylocation", 
                   {'INPUT': snap,
                    'JOIN': connected_streets_nodes,
                    'PREDICATE':[0,2,3,4,5],
                    'JOIN_FIELDS':['NODE_ID', 'invert_elev'],
                    'METHOD':1,'DISCARD_NONMATCHING':False,'PREFIX':'',
                    'OUTPUT': 'memory:outlet_network'})['OUTPUT']
        
        
        
    #calculation of length
    TEMP2_connected_streets = processing.run("qgis:exportaddgeometrycolumns", 
                   {'INPUT': temp_connected_streets,
                    'CALC_METHOD':0,
                    'OUTPUT':'memory:'})['OUTPUT']
    
    # Add auto-incremental field
    t_street = processing.run("native:addautoincrementalfield", {
        'INPUT': TEMP2_connected_streets,
        'FIELD_NAME': 'TEMP_NUM',
        'START': 1000,
        'GROUP_FIELDS': [],
        'SORT_EXPRESSION': '',
        'SORT_ASCENDING': True,
        'SORT_NULLS_FIRST': False,
        'OUTPUT': 'memory:'
    })['OUTPUT']
    
    # Create string field referencing the numeric field
    connected_streets = processing.run("native:fieldcalculator", {
        'INPUT': t_street,
        'FIELD_NAME': 'LINK_ID', 
        'FIELD_TYPE': 2,
        'FIELD_LENGTH': 20,
        'FIELD_PRECISION': 0,
        'NEW_FIELD': True,
        'FORMULA': 'concat(\'LN_\', "TEMP_NUM")',
        'OUTPUT': 'memory:'
    })['OUTPUT']
    
    # Create string field referencing the numeric field
    connected_streets = processing.run("native:deletecolumn", {
        'INPUT': connected_streets,
        'COLUMN': ['TEMP_NUM'],
        'OUTPUT': 'memory:'
    })['OUTPUT']
    
    return connected_streets_nodes, connected_streets, outlet_final

def create_mst(outlet, connected_streets, street_nodes, epsg):
    '''Function to create a Minimum Spanning Tree using shortest paths'''
    #cacluate XY coordinates of snap outlet for determining the MST
    for feature in outlet.getFeatures():
         
         X_coord=feature["xcoord"]# calculates X coordinate
         Y_coord=feature["ycoord"]# calculates Y coordinate
     
    #calculate shortest path, i.e. MST, using connected street nodes, network and outlet
    TEMP_MST_streets = processing.run("native:shortestpathlayertopoint", 
                   {'INPUT': connected_streets,
                    'STRATEGY':0,'DIRECTION_FIELD':'','VALUE_FORWARD':'',
                    'VALUE_BACKWARD':'','VALUE_BOTH':'','DEFAULT_DIRECTION':2,
                    'SPEED_FIELD':'','DEFAULT_SPEED':50,'TOLERANCE':2,
                    'START_POINTS': street_nodes,
                    'END_POINT': str(X_coord) + ', ' + str(Y_coord) + ' [EPSG:' + str(epsg) +']',
                    'OUTPUT': 'memory:'})['OUTPUT']
    

    
    #create a buffer aroud the MST to define an area for selecting the MST from the orginal connected treets in the area
    TEMP_MST_streets= processing.run("native:buffer", 
                   {'INPUT': TEMP_MST_streets,
                    'DISTANCE':2,'SEGMENTS':5,'END_CAP_STYLE':0,'JOIN_STYLE':0,
                    'MITER_LIMIT':2,'DISSOLVE':False,
                    'OUTPUT': 'memory:'})['OUTPUT']
    
    TEMP_MST_streets= processing.run("native:extractbylocation", 
                                {'INPUT': connected_streets,
                                 'PREDICATE': [6],
                                 'INTERSECT': TEMP_MST_streets,
                                 'OUTPUT': 'memory:'})['OUTPUT']
    
    return TEMP_MST_streets

def MST_streets2(mst,street_nodes):
    '''Continuation of modifying MST:
    Extract nodes from MST to start identifying from_nodes and to_nodes 
    Add node information to MST, copying names from previously identified nodes'''
    start = processing.run("native:extractspecificvertices", 
                   {'INPUT':mst,
                    'VERTICES':'0',
                    'OUTPUT': 'memory:start1'})['OUTPUT']
    
    start = processing.run("qgis:exportaddgeometrycolumns", 
                   {'INPUT':start,
                    'CALC_METHOD':1,
                    'OUTPUT': 'memory:start2'})['OUTPUT']
    
    end = processing.run("native:extractspecificvertices", 
                   {'INPUT': mst,
                    'VERTICES':'-1',
                    'OUTPUT':'memory:end'})['OUTPUT']
    
    end = processing.run("qgis:exportaddgeometrycolumns", 
                   {'INPUT':end,
                    'CALC_METHOD':1,
                    'OUTPUT':'memory:end2'})['OUTPUT']
    
    #add FROM/TO data to the MST network based on results from shortest path
    start  = processing.run("native:joinattributesbylocation", 
                   {'INPUT':start,
                    'JOIN':street_nodes,
                    'PREDICATE':[0,2,3,4,5],'JOIN_FIELDS':[],
                    'METHOD':1,'DISCARD_NONMATCHING':False,
                    'PREFIX':'',
                    'OUTPUT':'memory:'})['OUTPUT']
    
    start = processing.run("native:fieldcalculator", 
                   {'INPUT':start,
                    'FIELD_NAME':'FROM_node','FIELD_TYPE':2,'FIELD_LENGTH':0,
                    'FIELD_PRECISION':0,'FORMULA': 'NODE_ID',
                    'OUTPUT':'memory:'})['OUTPUT']
    
    end = processing.run("native:joinattributesbylocation", 
                   {'INPUT':end,
                    'JOIN':street_nodes,
                    'PREDICATE':[0,2,3,4,5],'JOIN_FIELDS':[],'METHOD':1,
                    'DISCARD_NONMATCHING':False,'PREFIX':'',
                    'OUTPUT': 'memory:'})['OUTPUT']
    
    end = processing.run("native:fieldcalculator", 
                   {'INPUT':end,
                    'FIELD_NAME':'TO_node','FIELD_TYPE':2,'FIELD_LENGTH':0,
                    'FIELD_PRECISION':0,'FORMULA': 'NODE_ID',
                    'OUTPUT': 'memory:'})['OUTPUT']
    
    
    mst2 = processing.run("native:joinattributestable", 
                   {'INPUT':mst,
                    'FIELD':'LINK_ID',
                    'INPUT_2':start,
                    'FIELD_2':'LINK_ID','FIELDS_TO_COPY':['FROM_node'],
                    'METHOD':1,'DISCARD_NONMATCHING':False,'PREFIX':'',
                    'OUTPUT':'memory:'})['OUTPUT']
    
    mst2 = processing.run("native:joinattributestable", 
                   {'INPUT':mst2,
                    'FIELD':'LINK_ID',
                    'INPUT_2':end,
                    'FIELD_2':'LINK_ID','FIELDS_TO_COPY':['TO_node'],
                    'METHOD':1,'DISCARD_NONMATCHING':False,'PREFIX':'',
                    'OUTPUT':'memory:'})['OUTPUT']
    
    return mst2

def corr_direction(link_info, outlet_info):
    '''Correct direction in each pipe: identify From_node and to_node to ensure 
    gravitational flow (deprecated version)'''
    g=nx.from_pandas_edgelist(link_info, "FROM_node", "TO_node", ["LINK_ID", "length"] )

    #determined node id for the outlet
    # print(outlet_info)
    outlet_id= outlet_info["NODE_ID"][0]
    
    #calculate degree of each node
    dg= dict(g.degree)#convert to dictionary due new format of Networkx
    isinstance(dg, dict)#check if it is disctionary
    nx.set_node_attributes(g, dg, "degree")# creates attribute called degree
    
    #create an array so directions can be updated easily
    np_links= link_info.to_numpy()
    # np_nodes= node_info.to_numpy()
    
    #convert dictionary of degrees into listr so it can be iterated over it
    dg_list=list(g.degree())
    
    
    #create a new list with the corrected flow directions, it works by calculating all the simple/shortest paths 
    # from the source nodes (k>1 excluding outlet) and then removes repeated paths
    temp_new_net=[]
    for i in range(0,len(dg_list)):
        if dg_list[i][1]==1 and dg_list[i][0] != outlet_id:
            paths = nx.all_simple_paths(g,dg_list[i][0],outlet_id)
            
            for path in map(nx.utils.pairwise, paths):
                a=list(path)
                temp_new_net.extend(a)
                
    new_net= list(dict.fromkeys(temp_new_net))
    
    #create a new array with the correct directions
    correct_dir=np.copy(np_links)
    correct_dir[:]=np_links[:]
    
    for i in range(0,len(np_links)):
        st= np.where(np_links[:,6]==new_net[i][0])
        ed = np.where(np_links[:,7]== new_net[i][1])
        lst_st=list(st[0])
        lst_end=list(ed[0])
        temp=set(lst_st)
        inters=list(temp.intersection(lst_end))
        if len(inters)==0:
            st_2= np.where(np_links[:,7]==new_net[i][0])
            ed_2= np.where(np_links[:,6]== new_net[i][1])
            lst_st2=list(st_2[0])
            lst_end2=list(ed_2[0])
            temp2=set(lst_st2)
            inters2=list(temp2.intersection(lst_end2))
            if len(inters2)==1:
                correct_dir[inters2,6]=new_net[i][0]
                correct_dir[inters2,7]=new_net[i][1]
    
    #removes forst columns of new direction to simplify file, it keeps the LINK ID and FROM/TO info
    correct_dir=np.delete(correct_dir, np.s_[0:4],1)
    return correct_dir

def correct_dir2(link_info, outlet_info):
    '''Correct direction in each pipe: identify From_node and to_node to ensure
    gravitational flow'''
    g=nx.from_pandas_edgelist(link_info, "FROM_node", "TO_node", ["LINK_ID", "length"] )

    #determined node id for the outlet
    outlet_id= outlet_info["NODE_ID"][0]
    
    #calculate degree of each node
    dg= dict(g.degree)#convert to dictionary due new format of Networkx
    isinstance(dg, dict)#check if it is disctionary
    nx.set_node_attributes(g, dg, "degree")# creates attribute called degree
    
    #create an array so directions can be updated easily
    np_links= link_info.to_numpy()

    
    #convert dictionary of degrees into listr so it can be iterated over it
    dg_list=list(g.degree())
    
    
    #create a new list with the corrected flow directions, it works by calculating all the simple/shortest paths 
    # from the source nodes (k>1 excluding outlet) and then removes repeated paths
    from itertools import chain
    
    # Assuming dg_list and outlet_id are defined
    temp_new_net = set()  # Use a set to avoid duplicates
    
    for node, value in dg_list:
        if value == 1 and node != outlet_id:
            paths = nx.all_simple_paths(g, node, outlet_id)
            # Flatten the list of paths and add to the set
            temp_new_net.update(chain.from_iterable(nx.utils.pairwise(path) for path in paths))
    
    # Convert the set back to a list if needed
    new_net = list(temp_new_net)
    
    #create a new array with the correct directions
    correct_dir=np.copy(np_links)
    correct_dir[:]=np_links[:]
        
    for i in range(len(new_net)):
        start_node = new_net[i][0]
        end_node = new_net[i][1]
    
        # Create boolean masks for the conditions
        mask_st = np_links[:, 6] == start_node
        mask_ed = np_links[:, 7] == end_node
        mask_st_2 = np_links[:, 7] == start_node
        mask_ed_2 = np_links[:, 6] == end_node
    
        # Find indices where both conditions are met
        inters = np.where(mask_st & mask_ed)[0]
        if len(inters) == 0:
            inters2 = np.where(mask_st_2 & mask_ed_2)[0]
            if len(inters2) == 1:
                correct_dir[inters2, 6] = start_node
                correct_dir[inters2, 7] = end_node
    
    #removes forst columns of new direction to simplify file, it keeps the LINK ID and FROM/TO info
    correct_dir=np.delete(correct_dir, np.s_[0:4],1)
    return correct_dir

#%%
def corr_inv_elev(link_info, node_info, corr_dir, min_slope):
    '''Function corrects elevations to ensure gravitational flow, using the min_slope (deprecated)'''
    node_info.drop(['Cover_Elevation'],axis=1,inplace=True)
    #create a new array with the correct elevations
    np_links= link_info.to_numpy()
    np_nodes= node_info.to_numpy()
    correct_elev=np.copy(np_nodes)
    correct_elev[:]=np_nodes[:]

    #initialize counter to see how many values are corrected
    it=0
    
    #initial seacrh for correcting the elevations
    for i in range(0, len(np_links)):
        
        st= np.where(correct_elev[:,0]==corr_dir[i][2])
        ed = np.where(correct_elev[:,0]==corr_dir[i][3])
        st_elev= np_nodes[st,3]
        end_elev= np_nodes[ed,3]
        if end_elev>= st_elev:
            it=it+1
            new_elev= np_nodes[st,3]- min_slope*corr_dir[i][0]
            correct_elev[ed,3]=new_elev
    
    #iterate over previous results until all elevations are corrected, i.e. it=0
    #initialize value to break in case of incosistencies
    br=0
    while it>0:

        it=0
        br=br+1
        for i in range(0, len(np_links)):
    
            st= np.where(correct_elev[:,0]==corr_dir[i][2])
            ed = np.where(correct_elev[:,0]==corr_dir[i][3])
            st_elev= correct_elev[st,3]
            end_elev= correct_elev[ed,3]
    
            if end_elev>= st_elev:
                it=it+1
                new_elev= correct_elev[st,3]- min_slope*corr_dir[i][0]
                correct_elev[ed,3]=new_elev
                
        if br>len(np_nodes)*1000:
            print("CHECK ELEVATIONS")
            break

    #removes forst columns of new direction to simplify file, it keeps the LINK ID and FROM/TO info
    corr_dir=np.delete(corr_dir, np.s_[0],1)  
    return correct_elev, corr_dir

#%%
def find_all_branches(G, outlet_node):
    """
    Find all branches in a directed graph using an iterative approach,
    starting with the longest path to the outlet and then finding branches.
    
    Returns a dictionary where keys are branch levels and values are lists of paths.
    """
    # Copy the graph to avoid modifying the original
    working_graph = G.copy()
    
    # Dictionary to store all branches by levels
    all_branches = {}
    
    # Level 1 is the main branch (longest path to outlet)
    level = 1
    
    # Find the main branch
    main_branch = find_longest_path(working_graph, outlet_node)
    all_branches[level] = [main_branch]
    
    # Remove edges of the main branch from the working graph
    for i in range(len(main_branch) - 1):
        u, v = main_branch[i], main_branch[i + 1]
        if working_graph.has_edge(u, v):
            working_graph.remove_edge(u, v)
    
    # Branch nodes in the main path that might have other branches
    branch_nodes = find_branch_nodes(G, main_branch)
    
    # Process each level
    while branch_nodes:
        level += 1
        all_branches[level] = []
        next_branch_nodes = []
        
        # For each branch node, find branches
        for node in branch_nodes:
            # Find all nodes that can connect to this branch node in the working graph
            connecting_nodes = [n for n in working_graph.nodes if nx.has_path(working_graph, n, node)]
            
            for source in connecting_nodes:
                if source == node:
                    continue  # Skip self-connections
                
                # Find the longest path from source to the branch node
                branch_path = find_longest_path(working_graph, node, source=None)
                
                if branch_path and len(branch_path) > 1:  # Valid path found
                    all_branches[level].append(branch_path)
                    
                    # Remove edges of this branch from working graph
                    for i in range(len(branch_path) - 1):
                        u, v = branch_path[i], branch_path[i + 1]
                        if working_graph.has_edge(u, v):
                            working_graph.remove_edge(u, v)
                    
                    # Add new branch nodes for next level
                    new_branch_nodes = find_branch_nodes(working_graph, branch_path)
                    next_branch_nodes.extend(new_branch_nodes)
        
        # Update branch nodes for next iteration
        branch_nodes = list(set(next_branch_nodes))  # Remove duplicates
        
        # If no new branches were found at this level, break
        if not all_branches[level]:
            del all_branches[level]
            break
    
    return all_branches

def find_longest_path(graph, target, source=None):
    """
    Find the longest path to a target node in the graph.
    If source is specified, find the longest path from that source to the target.
    Otherwise, find the longest path from any source to the target.
    """

    if source:
        # Find the longest path from the specified source to the target
        try:
            paths = list(nx.all_simple_paths(graph, source=source, target=target))
            if not paths:
                return []
            
            # Find path with maximum weight
            max_weight = -1
            longest_path = []
            
            for path in paths:
                weight = 0
                for u, v in zip(path[:-1], path[1:]):
                    weight += edge_weight(u, v, graph[u][v])
                if weight > max_weight:
                    max_weight = weight
                    longest_path = path
            
            return longest_path
            
        except nx.NetworkXNoPath:
            return []
    else:
        # Find sources (nodes with no incoming edges)
        sources = [n for n in graph.nodes if graph.in_degree(n) == 0]
        
        longest_path = []
        max_weight = -1
        
        for source in sources:
            try:
                for path in nx.all_simple_paths(graph, source=source, target=target):
                    # Calculate custom weight
                    weight = 0
                    for u, v in zip(path[:-1], path[1:]):
                        weight += edge_weight(u, v, graph[u][v])
                    if weight > max_weight:
                        max_weight = weight
                        longest_path = path
            except nx.NetworkXNoPath:
                continue
        
        return longest_path

def find_branch_nodes(graph, path):
    """
    Find nodes in the path that have incoming edges in the graph
    that are not part of the path. For a sewer network, we want to
    find nodes where other branches can flow into the path.
    """
    branch_nodes = []
    
    # Create a set of edges in the path for quick lookup
    path_edges = set((path[i], path[i+1]) for i in range(len(path)-1))
    
    # Check each node in the path except the source node (first node)
    for node in path:
        # For a sewer network, we want nodes that have incoming edges not in the path
        # Get all predecessors (nodes with edges pointing to this node)
        predecessors = list(graph.predecessors(node))
        
        # Check if there are incoming edges to this node that are not in the path
        has_branch = False
        for pred in predecessors:
            # If the predecessor is not in the path or the edge is not in our path_edges
            if pred not in path or (pred, node) not in path_edges:
                has_branch = True
                break
        
        if has_branch:
            branch_nodes.append(node)
    
    return branch_nodes

    
def edge_weight(u, v, d):
    # Prioritize importance: higher importance → larger weight
    return d['length'] * (d['importance'] ** 2)  # tweak exponent as needed

def get_links(netw,branch):
    
    edges_in_path = list(zip(branch[:-1], branch[1:]))


    # Get 'link_id' attributes for these edges
    link_ids = [netw.edges[u, v]['link_id'] for u, v in edges_in_path]
    return link_ids

def update_invert_elevations(nodes, pipes, link_id, min_slope):
    # Set index for quick lookup
    link = pipes.loc[link_id]
    from_node, to_node, length = link['FROM_node'], link['TO_node'], link['length']
    
    # depth_from = nodes.loc[from_node,'depth']
    cover_to= nodes.loc[to_node,'Cover_Elevation']

    from_elev = nodes.loc[from_node, 'invert_elev']
    to_elev = nodes.loc[to_node, 'invert_elev']
    
    slope = (from_elev - to_elev)/length
    max_slope = 10
    
    if min_slope>slope:
        # print(True)

        # Calculate the required drop
        required_drop = min_slope * length

        # Set invert elevation of downstream (TO_node) first, then upstream (FROM_node)
        # You can reverse this logic depending on your branch direction
    
        # Ensure TO_node invert_elev is below cover elev
        new_inv_to =  nodes.loc[from_node, 'invert_elev'] - required_drop
        nodes.loc[to_node, 'invert_elev'] = new_inv_to
    
        # Optional: update depth (if required)
        nodes.loc[to_node, 'depth'] = cover_to - new_inv_to
        
    if slope>max_slope:
        '''find the next depth and maybe add a jump'''
        print('slope is too high')

def corr_elev_new(links, nodes, directions, min_slope, outlet):
    '''Function corrects elevations to ensure gravitational flow, using  the min_slope. it prioritizes the main paths,
    and then the branches leading to it'''
    road_map = {
        'primary': 5.0,
        'secondary': 4.0,
        'tertiary': 3.0
        
    }

    # Apply the mapping with a default value of 1.0 for all other cases
    links['FROM_node'] = directions[:, 2]  # Update column 'f' with the third column of the NumPy array
    links['TO_node'] = directions[:, 3]  # Update column 'g' with the fourth column of the NumPy array
    links['highway'] = links['highway'].map(road_map).fillna(1.0).astype(float)

    G = nx.DiGraph()

    # Add edges
    for _, row in links.iterrows():
        from_node = row['FROM_node']
        to_node = row['TO_node']
        link_id=row['LINK_ID']
        length = row['length']
        importance = row['highway']
        
        # Define a composite score to maximize
        G.add_edge(from_node, to_node, 
                   length=length, 
                   importance=importance,
                   link_id = link_id)  # Tune 1000 depending on units

    outlet_node = outlet.iloc[0,3]
    all_paths = find_all_branches(G, outlet_node)
    
    nodes = nodes.set_index('NODE_ID')
    links = links.set_index('LINK_ID')  
        
    links.drop(['full_id', 'osm_id', 'osm_type', 'highway'], inplace=True,axis=1)
    
    for key in sorted(all_paths.keys(), reverse=True):
        
        for sublist in all_paths[key]:
            pipe_ids = get_links(G, sublist)
            # print(pipe_ids)
            for link in pipe_ids:
                
                update_invert_elevations(nodes,links, link, min_slope)

    elevation_dict = nodes['invert_elev'].to_dict()

    # Calculate slope directly
    links['slope'] = (
        (links['FROM_node'].map(elevation_dict) - links['TO_node'].map(elevation_dict)) 
        / links['length'])
    
    nodes = nodes.reset_index()
    links = links.reset_index()
    
    return links, nodes                                      

def get_base_sewer(temp_mst, corr_directions):
    ''' Function to replace attributes to the pipes after corrections'''
    #adds the corrected directions from CSV file and creates file with base sewer network
    mst = processing.run("native:fieldcalculator", 
                   {'INPUT':temp_mst,
                    'FIELD_NAME':'new_id','FIELD_TYPE':2,'FIELD_LENGTH':0,
                    'FIELD_PRECISION':0,'FORMULA':' \"LINK_ID\" ',
                    'OUTPUT':'memory:'})['OUTPUT']

    layer = QgsVectorLayer(f"Point?crs={temp_mst.crs().authid()}", "temp_layer", "memory")

    # Define fields
    layer.dataProvider().addAttributes([
        QgsField("field_1", QVariant.String),
        QgsField("field_2", QVariant.String),
        QgsField("field_3", QVariant.String),

    ])
    layer.updateFields()
    for row in corr_directions:
        # print([row[0], row[1], row[2]])
        feature = QgsFeature()
        feature.setAttributes([row[0], row[1], row[2]])
        # Optionally set geometry if needed, e.g., feature.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(x, y)))
        layer.dataProvider().addFeature(feature)

    # Update the layer's extent
    layer.updateExtents()

    mst = processing.run("native:joinattributestable", 
                   {'INPUT':mst,
                    'FIELD':'new_id',
                    'INPUT_2':layer,
                    'FIELD_2':'field_1','FIELDS_TO_COPY':[],'METHOD':1,
                    'DISCARD_NONMATCHING':False,'PREFIX':'',
                    'OUTPUT':'memory:'})['OUTPUT']

    # Get the index of each field and add it to list.
    fields_to_delete = {'LINK_ID', 'FROM_node', 'TO_node', 'new_id'}
    field_indexes_to_delete = [
        mst.fields().indexFromName(field.name())
        for field in mst.fields()
        if field.name() in fields_to_delete
    ]
    if field_indexes_to_delete:
        mst.dataProvider().deleteAttributes(field_indexes_to_delete)
        mst.updateFields()

    
    mst = processing.run("native:renametablefield", 
                   {'INPUT':mst,
                    'FIELD':'field_1','NEW_NAME':'LINK_ID',
                    'OUTPUT':'memory:'})['OUTPUT']

    mst = processing.run("native:renametablefield", 
                   {'INPUT':mst,
                    'FIELD':'field_2','NEW_NAME':'FROM',
                    'OUTPUT':'memory:'})['OUTPUT']
    base_sewer = processing.run("native:renametablefield", 
                   {'INPUT':mst,
                    'FIELD':'field_3','NEW_NAME':'TO',
                    'OUTPUT':'memory:'})['OUTPUT']
    return base_sewer

def get_sewer_nodes(corrected_elevations, street_nodes):
    ''' Function to update attributes to the nodes after corrections'''
    df = pd.DataFrame(corrected_elevations)
    
    layer = turn_dataframe_to_qgis(df)

    elevations = processing.run("native:fieldcalculator", 
                   {'INPUT':layer,
                    'FIELD_NAME':'new_id2','FIELD_TYPE':2,'FIELD_LENGTH':0,
                    'FIELD_PRECISION':0,'FORMULA':' \"0\" ',
                    'OUTPUT':'memory:'})['OUTPUT']
   

    street_nodes2 = processing.run("native:joinattributestable", 
                   {'INPUT':street_nodes,
                    'FIELD':'NODE_ID',
                    'INPUT_2':elevations,
                    'FIELD_2':'new_id2','FIELDS_TO_COPY':[],'METHOD':1,
                    'DISCARD_NONMATCHING':False,'PREFIX':'',
                    'OUTPUT':'memory:'})['OUTPUT']

    field_mapping = {'3': 'INVERT_ELV'}
    
    base_sewer_nodes = street_nodes2
    provider = base_sewer_nodes.dataProvider()
    
    # Get current field definitions
    fields = provider.fields()
    
    # Modify field names
    for i, field in enumerate(fields):
        if field.name() in field_mapping:
            provider.renameAttributes({i: field_mapping[field.name()]})
    
    base_sewer_nodes.updateFields()
           
    fields_to_delete = {'invert_elev', 'new_id2', '0', '1','2','4','new_id_2'}
    field_indexes_to_delete = [
        base_sewer_nodes.fields().indexFromName(field.name())
        for field in base_sewer_nodes.fields()
        if field.name() in fields_to_delete
    ]
    if field_indexes_to_delete:
        base_sewer_nodes.dataProvider().deleteAttributes(field_indexes_to_delete)
        base_sewer_nodes.updateFields()
    
    # Convert INVERT_ELV from string to float
    base_sewer_nodes = processing.run("native:fieldcalculator", {
        'INPUT': base_sewer_nodes,
        'FIELD_NAME': 'INVERT_EF',  # New field
        'FIELD_TYPE': 0,  # Double
        'FIELD_LENGTH': 20,
        'FIELD_PRECISION': 10,
        'FORMULA': 'to_real("INVERT_ELV")',  # Convert string to float
        'OUTPUT': 'memory:'
    })['OUTPUT']
    
    base_sewer_nodes =  processing.run("native:deletecolumn", {
        'INPUT': base_sewer_nodes,
        'COLUMN': ['INVERT_ELV'],
        'OUTPUT': 'memory:'
    })['OUTPUT']
    
    base_sewer_nodes = processing.run("native:renametablefield", {
        'INPUT':base_sewer_nodes,
        'FIELD':'INVERT_EF',
        'NEW_NAME':'INVERT_ELV',
        'OUTPUT':'memory:'
    })['OUTPUT']
    
    return base_sewer_nodes

'''Section 5'''

def join_designed_data(sewer, pipe_dims):
    '''Function to joins relevant data and update of pipes after design'''
    pipe_dims2 =  pipe_dims.reset_index()
    
    pipe_results = turn_dataframe_to_qgis(pipe_dims2, 'pipe_data')
    
    designed_pipes = processing.run("native:joinattributestable", 
                    {'INPUT':sewer,
                     'FIELD':'LINK_ID',
                     'INPUT_2':pipe_results,
                     'FIELD_2':'LINK_ID','FIELDS_TO_COPY':[],'METHOD':1,
                     'DISCARD_NONMATCHING':False,'PREFIX':'',
                     'OUTPUT':'memory:'})['OUTPUT']
    designed_pipes = processing.run("qgis:deletecolumn", 
                   {'INPUT':designed_pipes,
                    'COLUMN':['1','2','3','link_id_2'],
                    'OUTPUT':'memory:'})['OUTPUT']

    return designed_pipes

def update_outlet(outlet, nodes):
    """
    Add INVERT_ELV field from all_nodes shapefile to outlet_nodes shapefile based on matching NODE_ID values.
    
    Parameters:
    outlet_nodes (str): The shapefile with outlet nodes
    all_nodes (str): The shapefile with all the nodes with invert elevation
    
    Returns:
    QgsVectorLayer: In-memory layer with INVERT_ELV field added
    """
    outlet_with_invert_elv = processing.run("qgis:joinattributestable", {
        'INPUT': outlet,
        'FIELD': 'NODE_ID',
        'INPUT_2': nodes,
        'FIELD_2': 'NODE_ID',
        'FIELDS_TO_COPY': ['Cover_Elev', 'INVERT_ELV'],
        'METHOD': 1,
        'DISCARD_NONMATCHING': False,
        'PREFIX': '',
        'OUTPUT': 'memory:'
    })['OUTPUT']
    
    final_outlet = processing.run("qgis:deletecolumn", {
        'INPUT':outlet_with_invert_elv,
        'COLUMN':['invert_elev'],
        'OUTPUT':'memory:'
    })['OUTPUT']
    
    return final_outlet
