import torch
import esm
import numpy as np

# loads the ESM2 model and returns the model, alphabet, and batch converter

def load_esm(model_name="esm2_t12_35M_UR50D"):
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    return model, alphabet, batch_converter

# returns the embeddings of the sequence

def get_esm_embeddings(sequence, model, batch_converter, device="cpu"):
    data = [("protein", sequence)]

    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)
    model = model.to(device)

    with torch.no_grad():
        out = model(tokens, repr_layers=[model.num_layers], return_contacts=False)

    # (1, L+2, d)
    reps = out["representations"][model.num_layers][0]

    # remove special tokens [CLS], [EOS]
    residue_embeddings = reps[1:-1]

    return residue_embeddings.cpu().numpy()


if __name__ == "__main__":
    seq = ""

    model, alphabet, batch_converter = load_esm()

    emb = get_esm_embeddings(seq, model, batch_converter)

    print("Sequence length:", len(seq))
    print("Embedding shape:", emb.shape)