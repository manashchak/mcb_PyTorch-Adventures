### Download Conceptual Captions Images from TSVs 
### You can get the TSV files @ https://ai.google.com/research/ConceptualCaptions/download

python prep_cc_dataset.py \
    --path_to_root /mnt/datadrive/data/ConceptualCaptions \
    --path_to_store /mnt/datadrive/data/ConceptualCaptions \
    --num_workers 32 \
    --batch_size 100 \
    --sample_pct 0.2
