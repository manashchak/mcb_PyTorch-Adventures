accelerate launch stage2_unconditional_diffusion_trainer.py \
  --experiment_name "DiffusionTrainer" \
  --wandb_run_name "diffusion_celebahq" \
  --working_directory "work_dir/diffusion_celebahq" \
  --training_config "configs/stage2_diffusion_train.yaml" \
  --model_config "configs/ldm.yaml" \
  --path_to_vae_backbone "work_dir/vae_celeba/model.safetensors" \
  --dataset celebahq \
  --path_to_dataset "/mnt/datadrive/data/CelebAMask-HQ/CelebA-HQ-img" \
  --path_to_save_gens "src/diffusion_celebahq"


# accelerate launch stage2_unconditional_diffusion_trainer.py \
#   --experiment_name "DiffusionTrainer" \
#   --wandb_run_name "diffusion_celebahq" \
#   --working_directory "work_dir/diffusion_celebahq" \
#   --training_config "configs/stage2_diffusion_train.yaml" \
#   --model_config "configs/ldm.yaml" \
#   --path_to_vae_backbone "work_dir/vae_celeba/model.safetensors" \
#   --dataset conceptual_captions \
#   --path_to_dataset "/mnt/datadrive/data/ConceptualCaptions/hf_train" \
#   --path_to_save_gens "src/diffusion_cc"