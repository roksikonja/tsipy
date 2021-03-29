python demo_degradation.py --experiment_name="demo_degradation_explin" --degradation_model="explin"
python demo_degradation.py --experiment_name="demo_degradation_mr" --degradation_model="mr"

python demo_fusion.py --fusion_model="svgp"
python demo_fusion.py --experiment_name="demo_svgp_500" --fusion_model="svgp" --num_inducing_pts=500 --max_iter=5000

python demo_fusion.py --fusion_model="localgp" --num_inducing_pts=250 --max_iter=2000 --pred_window=0.2 --fit_window=0.6

python exp_acrim.py --fusion_model="svgp"
python exp_acrim.py --experiment_name="exp_acrim_500" --fusion_model="svgp" --num_inducing_pts=500 --max_iter=5000

python exp_acrim.py --experiment_name="exp_acrim_500" --fusion_model="localgp" --num_inducing_pts=500 --max_iter=5000

python exp_acrim_erbs.py --fusion_model="svgp"
python exp_acrim_erbs.py --experiment_name="exp_acrim_500" --fusion_model="svgp" --num_inducing_pts=500 --max_iter=5000

python exp_acrim_erbs.py --experiment_name="exp_acrim_500" --fusion_model="localgp" --num_inducing_pts=500 --max_iter=5000 --pred_window=1.0 --fit_window=3.0
