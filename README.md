# surfbright
Surface brightness python routine for Cloudy models

Input:  
Surfbright can be imported and used to process models from WARPFIELD by providing the '.out' file paths.
Alternatively you may allow it to recursively search a path for all files. 

Output:  
By default, output emission line profiles are saved as fits files using the '.out' file as a template, replacing '.out' with '.fits'

Fits extensions
0: Emission lines per radial bin
1: Radial bin in cm

Header Data:  
Headers include the name of each emission line and their index in the data extension 0.

Units:  
Output units are erg/s/cm-2
