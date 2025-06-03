# **OG.FLUSH**

## Description
**OG.FLUSH** (**O**pen-access **G**IS-based **F**low **L**ayout for **U**rban **S**ewer **H**ydrodynamics) is a QGIS plugin. It's developed for automatically generating a sewer system based on open-source geodata inputs, ideally the open-access source OpenStreetMaps. The outputs are then used to generate a hydrodynamic sewer system model like the Storm Water Management Model (SWMM).

The conceptual details and methodology can be found at https://www.mdpi.com/2073-4441/15/1/46.

This Tool was developed under the scope of the KlimaKonform project. Further information can be found here at https://klimakonform.uw.tu-dresden.de/

<br>

## Instructions
The following inputs are necessary for this plugin.


| Parameter | Requirment | Type | Source |
| --- | --- | --- | --- |
| Study area | Required | ESRI Shapefile | User-defined |
| Outlet | Required | ESRI Shapefile | User-defined |
| Buildings | Required | ESRI Shapefile | OSM |
| Streets | Required | ESRI Shapefile | OSM |
| Elevation | Required | Raster | Various (e.g. SRTM) |
| Population density | Optional | ESRI Shapefile | Various (e.g. OSM) |

\
Please note that this is a work in progress for all of its components, including but not limited to: the name, the icons, the tool itself, the graphical user interface, etc. 

1.	Download the plugin ZIP file.  
2.	Install the plugin in QGIS.  
  a.	**Go to Plugins** → **Manage and Install Plugins** → **Install from ZIP**.  
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

## Considerations and Limitations
1.	For consistency and a trouble-free run, please ensure that all the inputs are in the WGS84 UTM coordinate system.

<br>

## Authors/Contact
Diego.Novoa_Vazquez@tu-dresden.de  
Julian_David.Reyes_Silva@tu-dresden.de  
Md_Sifat.Siddik@mailbox.tu-dresden.de

<br>

## Important Links:
**Article:** Reyes-Silva, J. D., Novoa, D., Helm, B., & Krebs, P. (2023). An Evaluation Framework for Urban Pluvial Flooding Based on Open-Access Data. Water, 15(1), 46. https://doi.org/10.3390/w15010046  
**OpenStreetMaps:** https://planet.openstreetmap.org  
**SWMM:** https://www.epa.gov/water-research/storm-water-management-model-swmm
