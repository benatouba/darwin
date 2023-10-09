#!/bin/bash
# -------------------------------------------------------------------------------------------
set -e
# Input

#Choose interpolation method:
#'bil': bilinear
#'nn': nearest neighbour
#'bic': bicubic
METHOD="nn"
# the name of the variable you want to extract
VAR=prcp
# path of the csv files which contains the latitude and longitude of stations
COORDS="./stations.csv"
# path of the nc file you want to extract from
INPUT="./*${VAR}*.nc"
# FIXME: Make LAT_COL and LON_COL actually work
# column of latitues in csv file
LAT_COL=6
# column of longitudes in csv file
LON_COL=7
# -------------------------------------------------------------------------------------------

# ARgument parser
while [[ $# -gt 0 ]]; do
	case $1 in
	-m | --method)
		METHOD="$2"
		shift 2 # past argument
		;;
	-v | --variable)
		VAR="$2"
		shift 2 # past argument
		;;
	-c | --coords)
		COORDS="$2" # save positional arg
		shift 2     # past argument
		;;
	-i | --input)
		INPUT="$2"
		shift 2 # past argument
		;;
	-o | --output)
		OUTPUT="$2"
		shift 2 # past argument
		;;
	# --lon)
	# 	LON_COL="$2"
	# 	shift 2 # past argument
	# 	;;
	# --lat)
	# 	LAT_COL="$2"
	# 	shift 2 # past argument
	# 	;;
	-h | --help)
		echo "The script remaps values of gridded input data to stations in a CSV-File."
		echo "Options:"
		echo "-m | --method:   nn=nearest neighbour; bil=bilinear; bic=bilinear"
		echo "-v | --variable: the variable as named in the input file."
		echo "-i | --input:    input netcdf file path."
		echo "-o | --output:   output file path."
		echo "-c | --coords:   path to a csv-file containing the coordinates to compute"
		echo "--lon:   		     the column number of the lon coordinates in the coords-file (default=2)"
		echo "--lat:           the column number of the lat coordinates in the coords-file (default=3)"
		exit 0
		;;
	-*)
		echo "Unknown option $1"
		exit 1
		;;
	esac
done

# path of the output file
[[ "${OUTPUT-}" = "" ]] && OUTPUT="./${VAR}_${METHOD}.csv"
[[ "${LON_COL-}" = "" ]] && LON_COL=2
[[ "${LAT_COL-}" = "" ]] && LAT_COL=3

set -- "${POSITIONAL_ARGS[@]}" # restore positional parameters

echo "Interpolation Method  = ${METHOD}"
echo "Variable              = ${VAR}"
echo "Input file            = ${INPUT}"
echo "Output file           = ${OUTPUT}"
echo "Stations file         = ${COORDS}"

# read the csv file and extract the lon and lat
declare -a lon
declare -a lat
declare -a coord

stations=($(awk -F "," '{print $1}' "$COORDS"))
lon=($(awk -F "," '{print $3}' "$COORDS"))
lat=($(awk -F "," '{print $2}' "$COORDS"))
# read -ra lon <<<$(awk -F "\"*,\"*" '{print $3}' "$COORDS") # 7th column in the csv contains longitude information
# read -ra lat <<<$(awk -F "\"*,\"*" '{print $2}' "$COORDS") # 6th column in the csv contains latitude information

echo "-----Run Run-----"
for ((i = 0; i < ${#lon[@]}; i++)); do
	coord[i]="lon=${lon[i]}/lat="${lat[i]}
	STATIONS+=","${stations[i]}
done

for ((i = 0; i < ${#coord[@]}; i++)); do
	COORD=${coord[i]}
	if [ "$METHOD" == "bil" ]; then
		cdo remapbil,"$COORD" -selname,"$VAR" "$INPUT" inter.nc
	elif [ "$METHOD" == "nn" ]; then
		cdo remapnn,"$COORD" -selname,"$VAR" "$INPUT" inter.nc
	elif [ "$METHOD" == "bic" ]; then
		cdo remapbic,"$COORD" -selname,"$VAR" "$INPUT" inter.nc
	fi
	cdo info inter.nc | grep -v Date | awk '{ print $9 }' | sed -e 's/:/,/g' >>list3.txt
	paste list3.txt >>sta_"$i".txt # change here for different layer
	rm list3.txt
done
# cdo info inter.nc | grep -v Date | awk '{ print $3 }' | sed -e 's/-/,/g' >>list1.txt
cdo info inter.nc | grep -v Date | awk '{ print $3 }' >>list1.txt
echo "datetime$STATIONS" >"$OUTPUT"
paste -d , list1.txt $(ls -v sta_*) >>"$OUTPUT" # change here for different layer
rm inter.nc list1.txt sta_*.txt

echo "-----Done!!!-----"