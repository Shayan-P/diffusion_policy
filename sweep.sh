# change to train or test if you'd like...
python train.py \
    --config-dir=./diffusion_policy/config/sweep \
    --config-name=sweep_lambda_diffusion_transformer_lowdim_pusht_workspace.yaml \
    --multirun \
    training.seed=42 \
    training.device=cuda:0 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
