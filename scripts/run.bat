python exp_fusion.py -e="exp_fusion_svgp" -m="svgp"
python exp_fusion.py -e="exp_fusion_localgp" -m="localgp" -n_ind_pts=200 --max_iter=2000

python exp_virgo.py -e="exp_virgo"
python exp_virgo.py -e="exp_virgo_500" -n_ind_pts=500 --max_iter=5000
