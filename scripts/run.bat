python demo_degradation.py -e="demo_degradation_exp" -m="exp"
python demo_degradation.py -e="demo_degradation_explin" -m="explin"
python demo_degradation.py -e="demo_degradation_mr" -m="mr"

python demo_fusion.py -e="demo_fusion_svgp" -m="svgp"
python demo_fusion.py -e="demo_fusion_svgp_n_c" -m="svgp" -n -c

python demo_fusion.py -e="demo_fusion_localgp_250" -m="localgp" -n_ind_pts=250 --max_iter=2000 -p_w=0.2 -f_w=0.6
python demo_fusion.py -e="demo_fusion_localgp_500" -m="localgp" -n_ind_pts=500 --max_iter=5000 -p_w=0.2 -f_w=0.6

python exp_acrim.py -e="exp_acrim_svgp" -m="svgp"
python exp_acrim.py -e="exp_acrim_svgp_500" -m="svgp" -n_ind_pts=500 --max_iter=5000

python exp_virgo.py -e="exp_virgo"
