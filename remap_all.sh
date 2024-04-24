#!/bin/bash

bash xun_remap.sh -i /home/ben/data/GAR/tutorial/tutorial_d02km_d_2d_prcp_2022.nc -v prcp -m "nn" -c ./stations.csv -o ./tutorial_stations_d_2d_prcp_2022.csv
bash xun_remap.sh -i /home/ben/data/GAR/KF/KF_d02km_d_2d_prcp_2022.nc -v prcp -m "nn" -c ./stations.csv -o ./KF_stations_d_2d_prcp_2022.csv
bash xun_remap.sh -i /home/ben/data/GAR/KF_kfeta-trigger/KF_kfeta-trigger_d02km_d_2d_prcp_2022.nc -v prcp -m "nn" -c ./stations.csv -o ./KF_kfeta-trigger_stations_d_2d_prcp_2022.csv
bash xun_remap.sh -i /home/ben/data/GAR/KFCuP/KFCuP_d02km_d_2d_prcp_2022.nc -v prcp -m "nn" -c ./stations.csv -o ./KFCuP_stations_d_2d_prcp_2022.csv
bash xun_remap.sh -i /home/ben/data/GAR/G3D/G3D_d02km_d_2d_prcp_2022.nc -v prcp -m "nn" -c ./stations.csv -o ./G3D_stations_d_2d_prcp_2022.csv
bash xun_remap.sh -i /home/ben/data/GAR/TG/TG_d02km_d_2d_prcp_2022.nc -v prcp -m "nn" -c ./stations.csv -o ./TG_stations_d_2d_prcp_2022.csv
bash xun_remap.sh -i /home/ben/data/GAR/MYNN/MYNN_d02km_d_2d_prcp_2022.nc -v prcp -m "nn" -c ./stations.csv -o ./MYNN_stations_d_2d_prcp_2022.csv
bash xun_remap.sh -i /home/ben/data/GAR/MYNN-TG/MYNN-TG_d02km_d_2d_prcp_2022.nc -v prcp -m "nn" -c ./stations.csv -o ./MYNN-TG_stations_d_2d_prcp_2022.csv
bash xun_remap.sh -i /home/ben/data/GAR/KF-TG/KF-TG_d02km_d_2d_prcp_2022.nc -v prcp -m "nn" -c ./stations.csv -o ./KF-TG_stations_d_2d_prcp_2022.csv
bash xun_remap.sh -i /home/ben/data/GAR/KF-TG-bpl/KF-TG-bpl_d02km_d_2d_prcp_2022.nc -v prcp -m "nn" -c ./stations.csv -o ./KF-TG-bpl_stations_d_2d_prcp_2022.csv