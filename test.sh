# python train.py \
#     --config-dir=./diffusion_policy/config \
#     --config-name=train_diffusion_transformer_lowdim_pusht_workspace.yaml \
# #  --config-name=image_pusht_diffusion_policy_cnn.yaml \
#     training.seed=42 \
#     training.device=cuda:0 \
#     hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

python train.py \
    --config-dir=./diffusion_policy/config \
    --config-name=test_diffusion_transformer_lowdim_uncertainty_guidance_pusht_workspace.yaml \
#  --config-name=image_pusht_diffusion_policy_cnn.yaml \
    training.seed=42 \
    training.device=cuda:0 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
