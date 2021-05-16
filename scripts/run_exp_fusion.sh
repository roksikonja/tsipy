python exp_fusion.py -e="exp_fusion_svgp" -m="svgp"
python exp_fusion.py -e="exp_fusion_localgp" -m="localgp" -n_ind_pts=50 --max_iter=2000
python exp_fusion.py -e="exp_fusion_localgp_500" -m="localgp" -n_ind_pts=500 --max_iter=5000
