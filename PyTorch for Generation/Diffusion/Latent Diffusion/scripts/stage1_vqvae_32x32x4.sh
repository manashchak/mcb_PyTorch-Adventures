accelerate launch stage1_vqvae_trainer.py \
  --experiment_name "VAETrainer" \
  --wandb_run_name "vqvae_ffhd" \
  --working_directory "work_dir/vqvae_ffhd" \
  --log_wandb \
  --training_config "configs/stage1_vqvae_train.yaml"