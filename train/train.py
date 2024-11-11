import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.amp as amp
import matplotlib.pyplot as plt
from train.config import tokenizer, llm_model, protein_encoder_model, optimizer, scheduler, device
from train.model import Adaptor, ProteinChat, ProteinChatDataset
import wandb
from tqdm import tqdm

def load_data_from_csv(file_path):
    df = pd.read_csv(file_path)
    return df

wandb.init(project="ppi")

def train(model, dataloader, optimizer, scheduler, device, num_epochs=5):
    model.train()  
    model = model.to(device)
    epoch_losses = []
    scaler = amp.GradScaler()  # mixed precision training

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        with tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
            for batch_idx, (protein_seq1, protein_seq2, prompt, answer) in enumerate(pbar):
                protein_seq1, protein_seq2 = protein_seq1.to(device), protein_seq2.to(device)
                prompt, answer = prompt.to(device), answer.to(device)

                optimizer.zero_grad()  # clear the gradients of the optimizer

                with amp.autocast():  # Mixed precision training  - autmatic casting to float16
                    outputs = model(protein_seq1, protein_seq2, prompt, tokenizer) # fowrd pass
                    loss = torch.nn.CrossEntropyLoss()(outputs.logits.view(-1, outputs.logits.size(-1)), answer.view(-1))

                scaler.scale(loss).backward()  # backpropagation and gradient scaling
                scaler.step(optimizer)  
                scaler.update()  
                scheduler.step() 
                
                epoch_loss += loss.item()  

                pbar.set_postfix(loss=loss.item())

                if batch_idx % 10 == 0:  # log every 10 batches
                    wandb.log({"loss": loss.item()})


        avg_epoch_loss = epoch_loss / len(dataloader)  
        epoch_losses.append(avg_epoch_loss)  
        print(f"Epoch {epoch} complete. The Average Loss - {avg_epoch_loss}")

        checkpoint_path = f"model_checkpoint_epoch_{epoch + 1}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

        if epoch == num_epochs - 1:
            torch.save(protein_encoder_model.state_dict(), "protein_encoder_final.pt")
            print("Protein encoder saved after final epoch")



    plt.plot(range(num_epochs), epoch_losses, label="Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.savefig("training_loss_curve.png")
    plt.close()

    wandb.log({"training_loss_curve": wandb.Image("training_loss_curve.png")})


if __name__ == "__main__":

    file_path = 'ppi_data/protein_train_data.csv'
    df = load_data_from_csv(file_path)

    prompts = [
            "What is the interaction mechanism between these proteins?",
            "Can you describe how these proteins interact?",
            "Explain the nature of the interaction between these protein sequences.",
            "Provide details on the interaction mechanism of the given protein pair."
        ]

    train_data = []
    for index, row in df.iterrows():
        protein_seq1 = row['protein_seq1']
        protein_seq2 = row['protein_seq2']
        interaction_text = row['interaction_text']
        train_data.append((protein_seq1, protein_seq2, interaction_text))

    dataset = ProteinChatDataset(train_data, tokenizer, prompts)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, pin_memory=True)

    adaptor = Adaptor()
    model = ProteinChat(protein_encoder_model, adaptor, llm_model)

    train(model, dataloader, optimizer, scheduler, device)