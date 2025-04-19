import torch
from tqdm import tqdm
import numpy as np
import os
device=torch.device("cuda")


from datasets import load_dataset

dataset = load_dataset("LabHC/bias_in_bios")

from transformers import AutoTokenizer, AutoModel

access_token = "ADD_YOUR_OWN"

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf",token=access_token)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = AutoModel.from_pretrained("meta-llama/Llama-2-7b-hf",token=access_token).to(device)
model.resize_token_embeddings(len(tokenizer))


def tokenize(example):
    return tokenizer(example["hard_text"], padding=True, return_tensors="pt").to(device)


@torch.no_grad()
def get_embeddings_from_ds(ds, embd_size=4096, layers=[i for  i in range(33)], file_name="llama_embds.obj", infer_batch_size=32, batch_size=500):
    if not os.path.exists(file_name):
        mode="w+"
    else:
        mode = "r+"
    
    n_layers= len(layers)
    embds_file=np.memmap(file_name, dtype=np.float64, shape=(n_layers, len(ds),embd_size) , mode=mode)
    embds_mem=np.empty((n_layers, 0,embd_size))
    i=0
    gender=[]
    prof=[]

    
    for batch in tqdm(ds.iter(batch_size=infer_batch_size), total= int(len(ds)/infer_batch_size)):
        
        gender+=batch["gender"]
        prof+=batch["profession"]
        
        tokens=tokenize(batch)

        last_token_idx = (torch.sum(tokens["attention_mask"], dim=1) -1).unsqueeze(1).unsqueeze(0).expand(n_layers, -1, embd_size).unsqueeze(2)
        
        output = torch.stack(model(**tokens,output_hidden_states=True).hidden_states)[layers]#n_layers x inf_batch x n_tokens x embd_size
        batch_length =output.shape[1]
        extracted_output = torch.gather(output, 2, last_token_idx).view(n_layers, batch_length, -1)
        embds_mem= np.concatenate((embds_mem, extracted_output.cpu().numpy()), axis=1)

        if embds_mem.shape[1] >= batch_size:
            embds_file[:,i:i+embds_mem.shape[1],:] = embds_mem
            embds_file.flush()
            i+=embds_mem.shape[1]
            embds_mem=np.empty((n_layers, 0,embd_size))
    
    print("main loop done")
    embds_file[:,i:,:]=embds_mem
    embds_file.flush()

    return gender,prof

get_embeddings_from_ds(dataset["train"], file_name="llama_train_all_layers.obj")