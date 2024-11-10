import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AdamW, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from train.model import Adaptor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
protein_encoder_model = AutoModel.from_pretrained("Bo1015/proteinglm-1b-mlm", 
                                                  trust_remote_code=True).to(device)

tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-13b-v1.5")
llm_model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-13b-v1.5").to(device)

lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05)
llm_model = get_peft_model(llm_model, lora_config)

learning_rate = 1e-5

optimizer = AdamW(
    params=(
        list(protein_encoder_model.parameters()) +  # Protein encoder params
        list(llm_model.parameters()) +  # LLM params
        list(Adaptor.parameters())  # Adaptor params
    ), 
    lr=learning_rate, 
    weight_decay=0.05, 
    betas=(0.9, 0.999)
)

scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=2000, 
                                            num_training_steps=210000)

