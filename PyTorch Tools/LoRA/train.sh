accelerate launch train.py --experiment_name="LoRA" \
                           --hf_dataset food101 \
                           --hf_model_name microsoft/resnet-50 \
                           --working_directory work_dir \
                           --epochs 10 \
                           --batch_size 128 \
                           --gradient_accumulation_steps=1 \
                           --learning_rate 0.001 \
                           --weight_decay 1e-4 \
                           --max_grad_norm=1.0 \
                           --img_size=224 \
                           --num_workers=24 \
                           --rank 8 \
                           --lora_alpha 8 \
                           --use_rslora \
                           --lora_dropout 0.1


