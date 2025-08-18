# **OG.FLUSH**

## Description
**OG.FLUSH** (**O**pen-access **G**IS-based **F**low **L**ayout for **U**rban **S**ewer **H**ydrodynamics) is a QGIS plugin. It's developed for automatically generating a sewer system based on open-source geodata inputs, ideally the open-access source OpenStreetMaps. The outputs are then used to generate a hydrodynamic sewer system model like the Storm Water Management Model (SWMM).

The conceptual details and methodology can be found at https://www.mdpi.com/2073-4441/15/1/46.

This Tool was developed under the scope of the KlimaKonform project. Further information can be found here at https://klimakonform.uw.tu-dresden.de.

⚠️ This plugin may be considered a work in progress.


<br>

## Inputs
The following inputs are necessary for this plugin.


| Parameter | Requirment | Type | Description | Source |
| --- | --- | --- | --- | --- |
| Study area | Required | ESRI Polygon Shapefile | Boundary defining the area of interest | User-defined |
| Outlet | Required | ESRI Point Shapefile | Single endpoint for the network (e.g., treatment plant or discharge location) | User-defined |
| Buildings | Required | ESRI Polygon Shapefile | Building footprints within the study area | Various (e.g. OSM, Google Open Buildings) |
| Streets | Required | ESRI LineString Shapefile | OSM-based road network guiding the sewer design alignment | OSM |
| Digital elevation model | Required | Raster | Elevation data representing terrain heights | Various (e.g. city portals) |
| Population density | Optional | ESRI Point Shapefile | Population distribution for combined sewer design | Various (e.g. OSM) |


<br>

## Preparation
The following points should be considered to prepare the input files.
1.	Please ensure that all the inputs are in a UTM coordinate system.
2.	Please ensure that the extent of Buildings, Streets, Elevation and Population Density data covers the whole Study Area. One way to achieve this is to get a buffer of the Study Area and extract the data using the buffered area.
3.	Please ensure that all the plugin dependencies are available in QGIS. For more information, see the **External Dependencies** section.


<br>

## External Dependencies
OG.FLUSH depends on three Python libraries for operation, namely NetworkX, NumPy and Pandas. While some of these libraries might already be included in a QGIS environment, they also might not. In the case of such unavailability, the user has to install these libraries to use the plugin. To install these, you can follow the steps given below.

If QGIS has been installed with the official OSGeo4W installer, the OSGeo4W Shell can be used to install the required libraries.  
1.  First, search for "OSGeo4W Shell" in your Windows Start Menu and click to open it.  
2.  At the command prompt, type (or just copy) the following command to check if pip is installed.  
    `python -m pip --version`  
3.  If the console shows pip as already installed, proceed to step 4. Otherwise, type the following command to install pip.  
    `python -m ensurepip --upgrade`  
4.  Once pip is installed, type the following command to install the required libraries.  
    `python -m pip install networkx numpy pandas`  
5.  The console will download and install the missing libraries, and skip the ones already installed.  
6.  Once done, close the shell, restart QGIS and the plugin should work.  

QGIS' own Python Console can be used to install the libraries as well, if pip is already installed.  
1.  In QGIS, go to **Plugins** → **Python Console**. The console should appear, usually at the bottom of the QGIS window.  
2.  After the ">>>", type or copy the following lines one by one and press **Enter** after each line.  
    `import pip`  
    `pip install networkx numpy pandas`  
3.  Similar to the previous method, the console will download and install the necessary libraries.  
4.  Once done, restart QGIS and the plugin should also work.


<br>

## Instructions
1.	Download the plugin ZIP file.  
2.	Install the plugin in QGIS.  
  a.	Go to **Plugins** → **Manage and Install Plugins** → **Install from ZIP**.  
  b.	Select the ZIP file and click ok.  
3.	Once the plugin is installed, click on the icon* on the toolbar.  
    **It currently looks like a blue network. One day, we hope to hire a professional designer.*  
4.	If you don't see the plugin in the interface or on the installed list, try toggling it on and off in the Plugin Manager or restarting QGIS.
5.	Select the shapefiles needed from the dropdown menu. If they are loaded in QGIS, they can appear in their corresponding fields, otherwise, you can browse for them with the **…** button.  
6.	Select the design parameters according to local regulations. There is more information for them in the pop-up messages if you hover your mouse above the fields.  
7.	If you want to design a combined system, click on the checkbox and input the necessary data.  
8.	If you want to add meshness, as in additional pipes in the network for many reasons (e.g. having more storage), put how much in percentage of the ones skipped during the main design.  
9.	Select an output folder for your files.  
10.	Click the “Run” button.  
11.	If QGIS freezes for a couple of minutes, don't worry. Just let it work. Meanwhile, the progress bar should keep you company.  


<br>

## Assumptions and Limitations
1.	All sewers are designed along the streets (technically, under).
2.	The rational method is used for sewer pipe dimensions.
3.	Lower-resolution DEM (> 10 m) might result in a slightly lesser designed network.
4.	Only OSM is supported for Streets (**NOT** Buildings or other inputs). Other open-access street or road sources can be available in a later version.
5.	For now, this plugin works well with UTM coordinate systems. For other coordinate systems, the outlet distance validation might raise issues. Support for the Geographic system are planned for a later version.


<br>

## Authors
Diego Novoa Vazquez  
Julian David Reyes Silva  
Md Sifat Siddik  
  
All affiliated with,  
Urban Hydrology Research Group  
Institute for Urban and Industrial Water Management  
TU Dresden  


<br>

## Contact
diego.novoa_vazquez@tu-dresden.de  
md_sifat.siddik@tu-dresden.de  


<br>

## Important Links:
**Article:** Reyes-Silva, J. D., Novoa, D., Helm, B., & Krebs, P. (2023). An Evaluation Framework for Urban Pluvial Flooding Based on Open-Access Data. Water, 15(1), 46. https://doi.org/10.3390/w15010046  
**Sewer network meshness:** https://doi.org/10.2166/wst.2020.070  
**OpenStreetMaps:** https://planet.openstreetmap.org  
**SWMM:** https://www.epa.gov/water-research/storm-water-management-model-swmm
