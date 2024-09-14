import torch
import numpy as np
import pandas as pd
from transformer_lens import utils


SPACE = "·"
NEWLINE="↩"
TAB = "→"

def replacement_hook(mlp_post, hook, encoder):
    mlp_post_reconstr = encoder(mlp_post)[1]
    return mlp_post_reconstr

def mean_ablate_hook(mlp_post, hook):
    mlp_post[:] = mlp_post.mean([0, 1])
    return mlp_post

def zero_ablate_hook(mlp_post, hook):
    mlp_post[:] = 0.
    return mlp_post

def process_token(model, s):
    if isinstance(s, torch.Tensor):
        s = s.item()
    if isinstance(s, np.int64):
        s = s.item()
    if isinstance(s, int):
        s = model.to_string(s)
    s = s.replace(" ", SPACE)
    s = s.replace("\n", NEWLINE+"\n")
    s = s.replace("\t", TAB)
    return s

def process_tokens(model, l):
    if isinstance(l, str):
        l = model.to_str_tokens(l)
    elif isinstance(l, torch.Tensor) and len(l.shape)>1:
        l = l.squeeze(0)
    return [process_token(model, s) for s in l]

def process_tokens_index(model, l):
    if isinstance(l, str):
        l = model.to_str_tokens(l)
    elif isinstance(l, torch.Tensor) and len(l.shape)>1:
        l = l.squeeze(0)
    return [f"{process_token(model, s)}/{i}" for i,s in enumerate(l)]

def create_vocab_df(model, logit_vec, make_probs=False, full_vocab=None):
    if full_vocab is None:
        full_vocab = process_tokens(model, model.to_str_tokens(torch.arange(model.cfg.d_vocab)))
    vocab_df = pd.DataFrame({"token": full_vocab, "logit": utils.to_numpy(logit_vec)})
    if make_probs:
        vocab_df["log_prob"] = utils.to_numpy(logit_vec.log_softmax(dim=-1))
        vocab_df["prob"] = utils.to_numpy(logit_vec.softmax(dim=-1))
    return vocab_df.sort_values("logit", ascending=False)

def list_flatten(nested_list):
    return [x for y in nested_list for x in y]

def make_token_df(model, tokens, len_prefix=5, len_suffix=1):
    str_tokens = [process_tokens(model, model.to_str_tokens(t)) for t in tokens]
    unique_token = [[f"{s}/{i}" for i, s in enumerate(str_tok)] for str_tok in str_tokens]

    context = []
    batch = []
    pos = []
    label = []
    for b in range(tokens.shape[0]):
        # context.append([])
        # batch.append([])
        # pos.append([])
        # label.append([])
        for p in range(tokens.shape[1]):
            prefix = "".join(str_tokens[b][max(0, p-len_prefix):p])
            if p==tokens.shape[1]-1:
                suffix = ""
            else:
                suffix = "".join(str_tokens[b][p+1:min(tokens.shape[1]-1, p+1+len_suffix)])
            current = str_tokens[b][p]
            context.append(f"{prefix}|{current}|{suffix}")
            batch.append(b)
            pos.append(p)
            label.append(f"{b}/{p}")
    # print(len(batch), len(pos), len(context), len(label))
    return pd.DataFrame(dict(
        str_tokens=list_flatten(str_tokens),
        unique_token=list_flatten(unique_token),
        context=context,
        batch=batch,
        pos=pos,
        label=label,
    ))

def interpret_feature_given_tokens(model, encoder, freqs, feature_id, tokens):
    print(f"Feature {feature_id} freq: {freqs[feature_id].item():.4f}")
    # Let's run the model on some text and then use the autoencoder to process the MLP activations
    _, cache = model.run_with_cache(tokens, stop_at_layer=1, names_filter=utils.get_act_name("post", 0))
    mlp_acts = cache[utils.get_act_name("post", 0)]
    mlp_acts_flattened = mlp_acts.reshape(-1, encoder.d_mlp)
    loss, x_reconstruct, hidden_acts, l2_loss, l1_loss = encoder(mlp_acts_flattened)
    # This is equivalent to:
    # hidden_acts = F.relu((mlp_acts_flattened - encoder.b_dec) @ encoder.W_enc + encoder.b_enc)
    print("hidden_acts.shape", hidden_acts.shape)
    return tokens, hidden_acts

def get_top_activations_df(feature_id, tokens, hidden_acts):
    token_df = make_token_df(tokens)
    token_df["feature"] = utils.to_numpy(hidden_acts[:, feature_id])
    return token_df.sort_values("feature", ascending=False).head(20).style.background_gradient("coolwarm")

def get_logit_effect(model, encoder, feature_id):
    logit_effect = encoder.W_dec[feature_id] @ model.W_out[0] @ model.W_U
    return create_vocab_df(logit_effect).head(40).style.background_gradient("coolwarm")


def interpret_feature(all_tokens, feature_id, start_index, end_index):
    tokens = all_tokens[start_index:end_index]
    tokens, hidden_acts = interpret_feature_given_tokens(feature_id=feature_id, tokens=tokens)
    top_acts_df = get_top_activations_df(feature_id=feature_id , tokens=tokens, hidden_acts=hidden_acts)
    return top_acts_df
