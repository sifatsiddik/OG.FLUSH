import numpy as np
import os

def format_subcatchments(building_layer, street_layer, min_slope_buildings, max_slope_buildings, 
                         min_slope_streets, max_slope_streets):
    """
    Format subcatchment data for SWMM input file by processing building and street layers.
    
    Parameters:
        building_layer (QgsVectorLayer): Building polygons layer with attributes
        street_layer (QgsVectorLayer): Street polygons layer with attributes  
        min_slope_buildings (float): Minimum slope for buildings in %
        max_slope_buildings (float): Maximum slope for buildings in %
        min_slope_streets (float): Minimum slope for streets in %
        max_slope_streets (float): Maximum slope for streets in %
    
    Returns:
        numpy.ndarray: Array with subcatchment data [sub_id, raingage, outfall, area, imperv, width, slope, curb_length]
    """
    subcatchments = []
    
    # Process buildings
    for feature in building_layer.getFeatures():
        sub_id = feature["sub_id"]
        outfall = feature["Out_NODE"]
        area = feature["area"] / 10000  # Convert to hectares
        imperv = feature["imperv"]
        width = max(feature["width_1"], feature["width_2"])
        
        # Handle invalid slope values
        try:
            slope = float(feature["slope_perc"])
            if slope <= 0 or isinstance(slope, str):
                slope = min_slope_buildings
            elif slope > max_slope_buildings:
                slope = max_slope_buildings
        except (ValueError, TypeError):
            slope = min_slope_buildings
        
        subcatchments.append([sub_id, "LO", outfall, area, imperv, width, slope, 0])

    # Process streets
    for feature in street_layer.getFeatures():
        sub_id = feature["sub_id"]
        outfall = feature["Out_NODE"]
        area = feature["area"] / 10000
        imperv = feature["imperv"]
        width = max(feature["width_1"], feature["width_2"])
        
        # Handle invalid slope values
        try:
            slope = float(feature["slope_perc"])
            if slope <= 0 or isinstance(slope, str):
                slope = min_slope_streets
            elif slope > max_slope_streets:
                slope = max_slope_streets
        except (ValueError, TypeError):
            slope = min_slope_streets
        
        subcatchments.append([sub_id, "LO", outfall, area, imperv, width, slope, 0])

    return np.array(subcatchments)

def extract_polygons(building_layer, street_layer):
    """
    Extract polygon coordinates from building and street layers.
    
    Parameters:
        building_layer (QgsVectorLayer): Building polygons layer
        street_layer (QgsVectorLayer): Street polygons layer
    
    Returns:
        list: List of polygon coordinate data [sub_id, x_coordinate, y_coordinate]
    """
    polygons = []

    # Process buildings
    for feature in building_layer.getFeatures():
        sub_id = feature["sub_id"]
        geom = feature.geometry()
        
        if geom.isMultipart():
            polygon_coords = geom.asMultiPolygon()[0]
        else:
            polygon_coords = geom.asPolygon()
            
        for point in polygon_coords[0]:
            polygons.append([sub_id, point.x(), point.y()])

    # Process streets
    for feature in street_layer.getFeatures():
        sub_id = feature["sub_id"]
        geom = feature.geometry()
        
        if geom.isMultipart():
            polygon_coords = geom.asMultiPolygon()[0]
        else:
            polygon_coords = geom.asPolygon()
            
        for point in polygon_coords[0]:
            polygons.append([sub_id, point.x(), point.y()])

    return polygons

def write_swmm_file(output_path, output_filename, subcatchments, polygons, nodes, links, outfall, 
                    node_depth, rainfile_path, is_combined_system, dwf_data=None, groundwater_factor=0.1, 
                    water_consumption=100):
    """
    Write the SWMM input file.
    
    Parameters:
        output_path (str): Path to save the output file
        output_filename (str): Name of the output file
        subcatchments (numpy.ndarray): Subcatchment data
        polygons (list): Polygon coordinate data
        nodes (numpy.ndarray): Node data
        links (numpy.ndarray): Link data
        outfall (numpy.ndarray): Outfall data
        node_depth (float): Node depth
        rainfile_path (str): Full path to the rainfall file
        is_combined_system (bool): True if combined sewer system, False otherwise
        dwf_data (numpy.ndarray): Optional, dry weather flow data
        groundwater_factor (float): Optional, groundwater infiltration factor
        water_consumption (float): Optional, water consumption in L/cap*d
    
    Returns:
        str: Path to the created SWMM input file
    """
    output_file_path = os.path.join(output_path, output_filename)
    
    with open(output_file_path, 'w') as swmm_file:
        # [TITLE]
        swmm_file.write('[TITLE]\n')
        swmm_file.write('Generated Sewer Model\n')
        swmm_file.write('\n')
        
        # [OPTIONS]
        swmm_file.write('[OPTIONS]\n')
        swmm_file.write('FLOW_UNITS LPS\n')
        swmm_file.write('FLOW_ROUTING DYNWAVE\n')
        swmm_file.write('MIN_SLOPE 1\n')
        swmm_file.write('SKIP_STEADY_STATE YES\n')
        swmm_file.write('INFILTRATION GREEN_AMPT\n')
        swmm_file.write('START_DATE 11/16/2016\n')
        swmm_file.write('START_TIME 00:00\n')
        swmm_file.write('END_DATE 11/25/2016\n')
        swmm_file.write('END_TIME 23:55\n')
        swmm_file.write('REPORT_START_DATE 11/20/2016\n')
        swmm_file.write('REPORT_START_TIME 00:00\n')
        swmm_file.write('SWEEP_START  01/01\n')
        swmm_file.write('SWEEP_END 12/31\n')
        swmm_file.write('DRY_DAYS 0\n')
        swmm_file.write('REPORT_STEP  00:05:00\n')
        swmm_file.write('WET_STEP 00:01:00\n')
        swmm_file.write('DRY_STEP 00:01:00\n')
        swmm_file.write('ROUTING_STEP 0:01:00 \n')
        swmm_file.write('ALLOW_PONDING NO\n')
        swmm_file.write('INERTIAL_DAMPING PARTIAL\n')
        swmm_file.write('VARIABLE_STEP 0.75\n')
        swmm_file.write('LENGTHENING_STEP 300\n')
        swmm_file.write('MIN_SURFAREA 0\n')
        swmm_file.write('NORMAL_FLOW_LIMITED  BOTH\n')
        swmm_file.write('SKIP_STEADY_STATE NO\n')
        swmm_file.write('IGNORE_RAINFALL  NO\n')
        swmm_file.write('FORCE_MAIN_EQUATION  H-W\n')
        swmm_file.write('LINK_OFFSETS DEPTH\n')
        swmm_file.write('\n')
        
        # [EVAPORATION]
        swmm_file.write('[EVAPORATION]\n')
        swmm_file.write('CONSTANT 0.0\n')
        swmm_file.write('DRY_ONLY NO\n')
        swmm_file.write('\n')
        
        # [RAINGAGES]
        swmm_file.write('[RAINGAGES]\n')
        swmm_file.write(f'LO VOLUME 0:05 1.0 FILE {rainfile_path} LO MM\n')
        swmm_file.write('\n')
        
        # [SUBCATCHMENTS]
        swmm_file.write('[SUBCATCHMENTS]\n')
        for i in range(len(subcatchments)):
            swmm_file.write(f"{subcatchments[i,0]} {subcatchments[i,1]} {subcatchments[i,2]} "
                           f"{round(float(subcatchments[i,3]),3)} {subcatchments[i,4]} "
                           f"{round(float(subcatchments[i,5]),3)} {subcatchments[i,6]} 0\n")
        swmm_file.write('\n')
        
        # [SUBAREAS]
        swmm_file.write('[SUBAREAS]\n')
        for i in range(len(subcatchments)):
            swmm_file.write(f"{subcatchments[i,0]}  0.012 0.15 1.23 4.22 25 OUTLET\n")
        swmm_file.write('\n')
        
        # [INFILTRATION]
        swmm_file.write('[INFILTRATION]\n')
        for i in range(len(subcatchments)):
            swmm_file.write(f"{subcatchments[i,0]}  4.84 6.5 0.231\n")
        swmm_file.write('\n')
        
        # [JUNCTIONS]
        swmm_file.write('[JUNCTIONS]\n')
        outfall_id = [outfall[i, 3] for i in range(len(outfall))]  # List of outfalls that are included in the nodes
        for i in range(len(nodes)):
            if nodes[i,0] not in outfall_id:
                swmm_file.write(f"{nodes[i,0]}  {round(nodes[i, 4], 2)} {round(nodes[i, 5], 1)} 0 0 0\n")
        
        # [OUTFALLS]
        swmm_file.write('[OUTFALLS]\n')
        for i in range(len(outfall)):
            swmm_file.write(f"{outfall[i,3]}  {round(outfall[i,4],3)} FREE NO\n")
        swmm_file.write('\n')
        
        # [CONDUITS]
        swmm_file.write('[CONDUITS]\n')
        for i in range(len(links)):
            swmm_file.write(f"{links[i,0]} {links[i,2]} {links[i,3]} "
                           f"{round(links[i,1],3)} 0.015 0 0 0 0\n")
        swmm_file.write('\n')
        
        # [XSECTIONS]
        swmm_file.write('[XSECTIONS]\n')
        for i in range(len(links)):
            swmm_file.write(f"{links[i,0]} CIRCULAR {links[i,5]} 0 0 0 1\n")
        swmm_file.write('\n')
        
        # [REPORT]
        swmm_file.write('[REPORT]\n')
        swmm_file.write('SUBCATCHMENTS NONE\n')
        swmm_file.write('NODES ALL\n')
        swmm_file.write('LINKS ALL\n')
        swmm_file.write('\n')
        
        # [DWF] (Dry Weather Flow) - for combined sewer systems
        if is_combined_system and dwf_data is not None:
            swmm_file.write('[DWF]\n')
            for i in range(len(dwf_data)):
                if dwf_data[i,5] > 0:
                    flow_rate = round((1 + groundwater_factor) * dwf_data[i,5] * (water_consumption/86400), 4)
                    swmm_file.write(f"{int(dwf_data[i,1])} FLOW {flow_rate} "
                                   f"\"Hourly_pattern_weekdays\" \"Hourly_pattern_weekend\"\n")
            swmm_file.write('\n')
        
            # [PATTERNS]
            swmm_file.write('[PATTERNS]\n')
            swmm_file.write('Hourly_pattern_weekdays HOURLY     0.7549 0.6616 0.5668 0.5197 0.5465 0.7362\n')
            swmm_file.write('Hourly_pattern_weekdays            1.0463 1.2694 1.3027 1.2621 1.2282 1.2061\n')
            swmm_file.write('Hourly_pattern_weekdays            1.1910 1.1667 1.1358 1.1061 1.0817 1.0646\n')
            swmm_file.write('Hourly_pattern_weekdays            1.0554 1.0497 1.0612 1.0822 1.0236 0.8815\n')
            swmm_file.write('\n')
            swmm_file.write('Hourly_pattern_weekend WEEKEND    0.8787 0.8250 0.7590 0.6152 0.4980 0.5831\n')
            swmm_file.write('Hourly_pattern_weekend            0.8693 1.0798 1.2212 1.3105 1.3155 1.2144\n')
            swmm_file.write('Hourly_pattern_weekend            1.1366 1.0956 1.0364 1.0023 1.0265 1.1003\n')
            swmm_file.write('Hourly_pattern_weekend            1.1665 1.1632 1.1032 1.0502 1.0067 0.9427\n')
            swmm_file.write('\n')
        
        # [COORDINATES]
        swmm_file.write('[COORDINATES]\n')
        for i in range(len(nodes)):
            swmm_file.write(f"{nodes[i,0]} {nodes[i,1]} {nodes[i,2]}\n")
        swmm_file.write('\n')
        
        # [POLYGONS]
        swmm_file.write('[POLYGONS]\n')
        for poly in polygons:
            swmm_file.write(f"{poly[0]} {poly[1]} {poly[2]}\n")
        swmm_file.write('\n')
    
    print("SWMM input file written.")
    
    return output_file_path

def create_swmm_model(config):
    """
    Create a SWMM model from pre-loaded arrays and shapefiles.
    
    Parameters:
        config (dictionary): Configuration parameters (including):
            nodes_array (numpy.ndarray): Array of node data
            links_array (numpy.ndarray): Array of pipe links data
            outfall_array (numpy.ndarray): Array of outfall data
            building_layer (QgsVectorLayer): building polygons
            street_layer (QgsVectorLayer): street polygons
            model_path (str): Path to save SWMM model
            node_depth (float): Optional, depth below ground for node invert elevation (default: 2.5)
            min_slope_buildings (float): Optional, minimum slope for buildings in % (default: 1)
            max_slope_buildings (float): Optional, maximum slope for buildings in % (default: 5)
            min_slope_streets (float): Optional, minimum slope for streets in % (default: 1)
            max_slope_streets (float): Optional, maximum slope for streets in % (default: 5)
            water_consumption (float): Optional, water consumption in L/cap*d (default: 100)
            groundwater_factor (float): Optional, groundwater infiltration factor (default: 0.1)
            model_filename (str): Optional, output filename for SWMM model (default: 'swmm_model.inp')
            rain_filename (str): Optional, filename for rainfall data (default: 'rain_file.dat')
            is_combined_system (bool): Optional, whether combined sewer system (default: False)
            dwf_data (numpy.ndarray): Optional, dry weather flow data (default: None)
            
    Returns:
        str: Path to the created SWMM input file
    """
    # Extract required inputs
    nodes_array = config['nodes_array']
    links_array = config['links_array']
    outfall_array = config['outfall_array']
    building_layer = config['building_layer']
    street_layer = config['street_layer']
    model_path = config['model_path']
    
    # Extract configuration parameters with defaults
    node_depth = config.get('node_depth', 2.5)
    min_slope_buildings = config.get('min_slope_buildings', 1)
    max_slope_buildings = config.get('max_slope_buildings', 5)
    min_slope_streets = config.get('min_slope_streets', 1)
    max_slope_streets = config.get('max_slope_streets', 5)
    water_consumption = config.get('water_consumption', 100)
    groundwater_factor = config.get('groundwater_factor', 0.1)
    model_filename = config.get('model_filename', 'swmm_model.inp')
    rain_filename = config.get('rain_filename', 'rain_file.dat')
    is_combined_system = config.get('is_combined_system', 0)
    dwf_data = config.get('dwf_data')
    
    # Ensure output path exists
    os.makedirs(model_path, exist_ok=True)
    
    # Format subcatchments
    subcatchments = format_subcatchments(
        building_layer, street_layer, 
        min_slope_buildings, max_slope_buildings, min_slope_streets, max_slope_streets)
    
    # Extract polygon coordinates
    polygons = extract_polygons(building_layer, street_layer)
    
    # Write SWMM input file
    try:
        output_file_path = write_swmm_file(
            model_path, model_filename, subcatchments, polygons, nodes_array, links_array, outfall_array,
            node_depth, rain_filename, is_combined_system, dwf_data, groundwater_factor, water_consumption)
        
        return output_file_path
    except Exception as e:
        print(f"Error creating SWMM model: {str(e)}")
        return None
