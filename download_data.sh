#!/bin/bash
echo "Downloading views prediction competition files..."
wget -q --show-progress -O ./data/views.zip https://www.dropbox.com/sh/yxk5w04p2e1xtqk/AACU2k5EUOuEeMq2kZ3gpZZwa?dl=0
echo "Unpacking views prediction competition files..."
unzip -q ./data/views.zip -d ./data/views_data/
# GeoBoundaries data
echo "Downloading CGAZ Geoboundaries ADM0 dataset..."
wget -q --show-progress -O ./data/geoBoundariesCGAZ_ADM0.gpkg https://github.com/wmgeolab/geoBoundaries/raw/main/releaseData/CGAZ/geoBoundariesCGAZ_ADM0.gpkg