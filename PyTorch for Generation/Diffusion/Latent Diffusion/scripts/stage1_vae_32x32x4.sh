
accelerate launch stage1.py \
  --experiment_name "VAETrainer" \
  --wandb_run_name "vae_bird_origgan" \
  --working_directory "work_dir/vae_birds" \
  --log_wandb \
  --training_config "configs/stage1_vae_train.yaml"