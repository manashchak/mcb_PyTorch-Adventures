### Download Conceptual Captions Images from TSVs 
### You can get the TSV files @ https://ai.google.com/research/ConceptualCaptions/download

### Use img2dataset To Download Images ###
### https://github.com/rom1504/img2dataset
img2dataset --url_list "/mnt/datadrive/data/ConceptualCaptions/cc3m.tsv" \
            --input_format "tsv"\
            --url_col "url" \
            --caption_col "caption" \
            --output_format parquet \
            --output_folder /mnt/datadrive/data/ConceptualCaptions/train \
            --processes_count 1 \
            --thread_count 48 \
            --image_size 256 \
            --resize_mode keep_ratio \
            --enable_wandb True \
            --wandb_project "DownloadCC3MLocal"