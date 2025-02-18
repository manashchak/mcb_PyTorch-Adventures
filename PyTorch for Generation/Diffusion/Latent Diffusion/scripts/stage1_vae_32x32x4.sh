
accelerate launch stage1_vae_trainer.py \
  --experiment_name "VAETrainer" \
  --wandb_run_name "vae_ffhd" \
  --working_directory "work_dir/vae_ffhd" \
  --training_config "configs/stage1_vae_train.yaml" \
  --dataset imagenet \
  --path_to_dataset "/mnt/datadrive/data/ImageNet/train/" \
  --path_to_save_gens "src/imagenet_gen" \
  --log_wandb


