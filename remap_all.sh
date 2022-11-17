#!/bin/bash

bash xun_remap.sh -i /home/ben/data/GAR/rc_trop_ls_tutorial/rc_trop_ls_tutorial_d02km_d_2d_prcp_2022.nc -v prcp -m "nn" -c ./stations.csv -o ./rc_trop_ls_tutorial_stations_d_2d_prcp_2022.csv
bash xun_remap.sh -i /home/ben/data/GAR/rc_trop_ls_KF/rc_trop_ls_KF_d02km_d_2d_prcp_2022.nc -v prcp -m "nn" -c ./stations.csv -o ./rc_trop_ls_KF_stations_d_2d_prcp_2022.csv
bash xun_remap.sh -i /home/ben/data/GAR/rc_trop_ls_KF_kfeta-trigger/rc_trop_ls_KF_kfeta-trigger_d02km_d_2d_prcp_2022.nc -v prcp -m "nn" -c ./stations.csv -o ./rc_trop_ls_KF_kfeta-trigger_stations_d_2d_prcp_2022.csv
bash xun_remap.sh -i /home/ben/data/GAR/rc_trop_ls_KFCuP/rc_trop_ls_KFCuP_d02km_d_2d_prcp_2022.nc -v prcp -m "nn" -c ./stations.csv -o ./rc_trop_ls_KFCuP_stations_d_2d_prcp_2022.csv
bash xun_remap.sh -i /home/ben/data/GAR/rc_trop_ls_G3D/rc_trop_ls_G3D_d02km_d_2d_prcp_2022.nc -v prcp -m "nn" -c ./stations.csv -o ./rc_trop_ls_G3D_stations_d_2d_prcp_2022.csv
bash xun_remap.sh -i /home/ben/data/GAR/rc_trop_ls_TG/rc_trop_ls_TG_d02km_d_2d_prcp_2022.nc -v prcp -m "nn" -c ./stations.csv -o ./rc_trop_ls_TG_stations_d_2d_prcp_2022.csv
bash xun_remap.sh -i /home/ben/data/GAR/rc_trop_ls_MYNN/rc_trop_ls_MYNN_d02km_d_2d_prcp_2022.nc -v prcp -m "nn" -c ./stations.csv -o ./rc_trop_ls_MYNN_stations_d_2d_prcp_2022.csv
bash xun_remap.sh -i /home/ben/data/GAR/rc_trop_ls_MYNN-TG/rc_trop_ls_MYNN-TG_d02km_d_2d_prcp_2022.nc -v prcp -m "nn" -c ./stations.csv -o ./rc_trop_ls_MYNN-TG_stations_d_2d_prcp_2022.csv
bash xun_remap.sh -i /home/ben/data/GAR/rc_trop_ls_KF-TG/rc_trop_ls_KF-TG_d02km_d_2d_prcp_2022.nc -v prcp -m "nn" -c ./stations.csv -o ./rc_trop_ls_KF-TG_stations_d_2d_prcp_2022.csv
bash xun_remap.sh -i /home/ben/data/GAR/rc_trop_ls_KF-TG-bpl/rc_trop_ls_KF-TG-bpl_d02km_d_2d_prcp_2022.nc -v prcp -m "nn" -c ./stations.csv -o ./rc_trop_ls_KF-TG-bpl_stations_d_2d_prcp_2022.csv
