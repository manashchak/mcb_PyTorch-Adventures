import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import RobertaForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer
import evaluate
from transformers import TrainerCallback

class LoRALinear(nn.Module):

    """
    This is a basic implementation of the paper LoRA
    https://arxiv.org/pdf/2106.0968

    """
    def __init__(self, 
                 weight, 
                 bias, 
                 rank=8, 
                 lora_alpha=1, 
                 use_rslora=True, 
                 lora_dropout=0.0,
                 b_grad=True):
        
        super().__init__()
        
        self.rank = rank 
        self.lora_alpha = lora_alpha

        ### These Are Our Pretrained Parameters ###
        self.W = weight
        self.b = bias

        ### Change Gradient Flag ###
        self.W.requires_grad = False
        self.b.requires_grad = b_grad

        ### Get In/Out Features (PyTorch Weight Matrix goes Out/In) ###
        out_features, in_features = self.W.shape

        ### Create our Low Rank Matricies ###
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank), requires_grad=True)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features), requires_grad=True)

        ### Initialize lora_A w/ gaussian, B stays as 0s (as described in paper) ###
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        ### Create LoRA Dropout ###
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else lambda x: x

        ### Compute Scaling for LoRA ###
        self.scaling = self.lora_alpha / self.rank**0.5 if use_rslora else self.lora_alpha / self.rank

    def __repr__(self):
        return f"LoRALinear(in_features={self.W.shape[1]}, out_features={self.W.shape[1]}, rank={self.rank})"
    
    def forward(self, x):

        ### Output W/ Original Weights (No Gradients) ###
        orig_output = x @ self.W.T

        ### Low Rank Output (With Gradients) ###
        low_rank_output = ((self.lora_dropout(x) @ self.lora_A) @ self.lora_B) * self.scaling

        ### Sum Outputs ###
        output = orig_output + low_rank_output

        ### Add Bias (if its there) ###
        if self.b is not None:

            output = output + self.b

        return output
    
class LoRAEmbedding(nn.Module):

    """
    This is a basic implementation of the paper LoRA
    https://arxiv.org/pdf/2106.0968

    """
    def __init__(self, 
                 weight, 
                 rank=8, 
                 lora_alpha=1, 
                 use_rslora=True, 
                 padding_idx=None):
        
        super().__init__()
        
        self.rank = rank 
        self.lora_alpha = lora_alpha
        self.padding_idx = padding_idx

        ### These Are Our Pretrained Parameters ###
        self.W = weight

        ### Change Gradient Flag ###
        self.W.requires_grad = False

        ### Get In/Out Features (PyTorch Weight Matrix goes Out/In) ###
        num_embeddings, embedding_dim = self.W.shape

        ### Create our Low Rank Matricies ###
        self.lora_A = nn.Parameter(torch.zeros(num_embeddings, rank), requires_grad=True)
        self.lora_B = nn.Parameter(torch.zeros(rank, embedding_dim), requires_grad=True)

        ### Different than the paper but matches implementation ###
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        ### Compute Scaling for LoRA ###
        self.scaling = self.lora_alpha / self.rank**0.5 if use_rslora else self.lora_alpha / self.rank
    
    def __repr__(self):
        return f"LoRAEmbedding({self.W.shape[0]}, {self.W.shape[1]}, rank={self.rank})"
    
    def forward(self, x):

        ### Output W/ Original Weights (No Gradients) ###
        orig_output = F.embedding(x, self.W, padding_idx=self.padding_idx)

        ### Embed with Low Rank A Embedding Matrix ###
        low_rank_A_output = F.embedding(x, self.lora_A)

        ### Project Back to Embed Dim with Low Rank B ###
        low_rank_output = (low_rank_A_output @ self.lora_B) * self.scaling
    
        ### Sum Outputs ###
        output = orig_output + low_rank_output
        
        return output
    

class LoRAModel(nn.Module):

    def __init__(self, 
                 model, 
                 rank=16, 
                 lora_alpha=1.0,
                 use_rslora=True,
                 target_modules=None,
                 lora_dropout=0.0,
                 initializer_range=1.0,
                 b_grad=True):
        
        super().__init__()
        
        self.model = model
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.use_rslora = use_rslora
        self.lora_dropout = lora_dropout
        self.initializer_range = initializer_range
        self.b_grad = b_grad
        self.target_modules = target_modules

        ### Compute Number of Trainable Parameters Before LoRA ###
        before_params = self._compute_parameters()

        ### Change Layers to LoRA ###
        self._apply_lora(self.model)

        ### Compute Number of Trainable Parameters After LoRA ###
        after_params = self._compute_parameters()

        print(f"Initial Parameters : {before_params} || LoRA Parameters : {after_params} || trainable% : {round(after_params/before_params, 2)}")

    def _apply_lora(self, module):
        
        ### If no target module specified, apply to all linear/embedding layers ###
        if self.target_modules is None:
            convert_to_lora = True

        ### Recursively Go Through Model and Find Linear Layers ###
        for name, child in module.named_children():
            
            ### If our name is in our target modules and the target modules was not None we convert ###            
            if (self.target_modules is not None): 
                if (name in self.target_modules):
                    convert_to_lora = True
                else:
                    convert_to_lora = False
                
            ### If a child of this module is a linear layer, update with our LoRALinear ###
            if isinstance(child, nn.Linear):
                
                if convert_to_lora:

                    ### Create LoRA Layer for This Linear Layer ###
                    lora_layer = LoRALinear(weight=child.weight, 
                                            bias=child.bias, 
                                            rank=self.rank,
                                            lora_alpha=self.lora_alpha, 
                                            use_rslora=self.use_rslora, 
                                            lora_dropout=self.lora_dropout,
                                            b_grad=self.b_grad)
                    
                    ### Replace the linear layer (identified by its name) in this module with our lora layer ###
                    setattr(module, name, lora_layer)

            ### If its an Embedding Layer then We Can Replace With Our Own LoraEmbedding ###
            elif isinstance(child, nn.Embedding):
                
                if convert_to_lora:

                    lora_layer = LoRAEmbedding(child.weight, 
                                               rank=self.rank, 
                                               lora_alpha=self.lora_alpha, 
                                               use_rslora=self.use_rslora, 
                                               padding_idx=child.padding_idx)
                    
                    ### Replace the embedding layer (identified by its name) in this module with our lora layer ###
                    setattr(module, name, lora_layer)

            ### Else, Dig Deeper Into the Module To Search For Linear Layers ###
            else:

                self._apply_lora(child)

    def _compute_parameters(self):

        total = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                total += param.numel()

        return total
    
    def forward(self, *inputs, **kwargs):
        return self.model(*inputs, **kwargs)

if __name__ == "__main__":

    ### Load Model ###
    model = RobertaForSequenceClassification.from_pretrained("FacebookAI/roberta-base", num_labels=5)

    ### Convert Roberta Backbone with LoRA (ignoring classifier) ###
    target_modules = ["query", "key", "value", "dense", "word_embeddings", "position_embeddings"]
    model.roberta = LoRAModel(model.roberta, rank=16, lora_alpha=16, target_modules=target_modules)
    print(model)

    ### Prepare Dataset/DataLoader ###
    dataset = load_dataset("yelp_review_full")
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(5000))
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(5000))

    ### Set Up Training ###
    training_args = TrainingArguments(output_dir="work_dir",
                                      eval_strategy="steps",
                                      eval_steps=25,
                                      per_device_train_batch_size=64,
                                      per_device_eval_batch_size=64,
                                      num_train_epochs=5, 
                                      warmup_ratio=0.05, 
                                      bf16=True,
                                      dataloader_num_workers=32,
                                      report_to="none")

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()