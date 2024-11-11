import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from torch.utils.data import Dataset
from train.config import device


class Adaptor(nn.Module):
    def __init__(self, input_dim=2048, output_dim=5120):
        super(Adaptor, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, protein_embedding):
        return self.linear(protein_embedding)
    
class ProteinChat(nn.Module):
    def __init__(self, protein_encoder, adaptor, llm):
        super(ProteinChat, self).__init__()
        self.protein_encoder = protein_encoder
        self.adaptor = adaptor
        self.llm = llm

    def forward(self, protein_seq1, protein_seq2, prompt, tokenizer):
        def checkpoint_fn(module, *inputs):
            return checkpoint.checkpoint(module, *inputs)

        protein_embedding1 = checkpoint_fn(self.protein_encoder, protein_seq1)
        protein_embedding2 = checkpoint_fn(self.protein_encoder, protein_seq2)

        adapted_embedding1 = checkpoint_fn(self.adaptor, protein_embedding1)
        adapted_embedding2 = checkpoint_fn(self.adaptor, protein_embedding2)

        combined_embedding = (adapted_embedding1 + adapted_embedding2) / 2

        prompt_enc = tokenizer(prompt, return_tensors="pt", padding=True).to(combined_embedding.device)
        inputs_embeds = combined_embedding.unsqueeze(0)

        output = self.llm(**prompt_enc, inputs_embeds=inputs_embeds)
    
        return output

class ProteinChatDataset(Dataset):
    def __init__(self, data, tokenizer, prompts, max_length=600):
        self.data = data
        self.tokenizer = tokenizer
        self.prompts = prompts  
        self.max_length = max_length

        # Expand dataset with multiple prompts
        self.expanded_data = []
        for protein_seq1, protein_seq2, interaction_text in self.data:
            for prompt in self.prompts:
                self.expanded_data.append((protein_seq1, protein_seq2, prompt, interaction_text))

    def __len__(self):
        return len(self.expanded_data)  # 4 times the original dataset size

    def __getitem__(self, idx):
        protein_seq1, protein_seq2, prompt, interaction_text = self.expanded_data[idx]

        protein_seq1_enc = self.tokenizer(protein_seq1, return_tensors="pt", padding="max_length", max_length=self.max_length, truncation=True).to(device)
        protein_seq2_enc = self.tokenizer(protein_seq2, return_tensors="pt", padding="max_length", max_length=self.max_length, truncation=True).to(device)
        prompt_enc = self.tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=self.max_length, truncation=True).to(device)
        interaction_text_enc = self.tokenizer(interaction_text, return_tensors="pt", padding="max_length", max_length=self.max_length, truncation=True).to(device)

        return protein_seq1_enc, protein_seq2_enc, prompt_enc, interaction_text_enc
