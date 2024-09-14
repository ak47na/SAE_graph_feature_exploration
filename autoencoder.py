from transformer_lens import utils
import torch
import pprint
import torch.nn as nn
import torch.nn.functional as F
from features_utils import replacement_hook, zero_ablate_hook, mean_ablate_hook
from functools import partial
import tqdm
import einops
from jaxtyping import Float, Int
from torch import Tensor
from transformer_lens import HookedTransformer


DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

class AutoEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        d_hidden = cfg["d_mlp"] * cfg["dict_mult"]
        d_mlp = cfg["d_mlp"]
        l1_coeff = cfg["l1_coeff"]
        dtype = DTYPES[cfg["enc_dtype"]]
        torch.manual_seed(cfg["seed"])
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_mlp, d_hidden, dtype=dtype)))
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_hidden, d_mlp, dtype=dtype)))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(d_mlp, dtype=dtype))

        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

        self.d_hidden = d_hidden
        self.l1_coeff = l1_coeff
        self.d_mlp = d_mlp

        self.device = device
        self.to(device)

    def forward(self, x):
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc + self.b_enc)
        x_reconstruct = acts @ self.W_dec + self.b_dec
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).sum(-1).mean(0)
        l1_loss = self.l1_coeff * (acts.float().abs().sum())
        loss = l2_loss + l1_loss
        return loss, x_reconstruct, acts, l2_loss, l1_loss

    @torch.no_grad()
    def remove_parallel_component_of_grads(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj

    # def get_version(self):
    #     return 1+max([int(file.name.split(".")[0]) for file in list(SAVE_DIR.iterdir()) if "pt" in str(file)])

    # def save(self):
    #     version = self.get_version()
    #     torch.save(self.state_dict(), SAVE_DIR/(str(version)+".pt"))
    #     with open(SAVE_DIR/(str(version)+"_cfg.json"), "w") as f:
    #         json.dump(cfg, f)
    #     print("Saved as version", version)

    # def load(cls, version):
    #     cfg = (json.load(open(SAVE_DIR/(str(version)+"_cfg.json"), "r")))
    #     pprint.pprint(cfg)
    #     self = cls(cfg=cfg)
    #     self.load_state_dict(torch.load(SAVE_DIR/(str(version)+".pt")))
    #     return self

    @classmethod
    def load_from_hf(cls, version):
        """
        Loads the saved autoencoder from HuggingFace.

        Version is expected to be an int, or "run1" or "run2"

        version 25 is the final checkpoint of the first autoencoder run,
        version 47 is the final checkpoint of the second autoencoder run.
        """
        if version=="run1":
            version = 25
        elif version=="run2":
            version = 47

        cfg = utils.download_file_from_hf("NeelNanda/sparse_autoencoder", f"{version}_cfg.json")
        pprint.pprint(cfg)
        self = cls(cfg=cfg)
        self.load_state_dict(utils.download_file_from_hf("NeelNanda/sparse_autoencoder", f"{version}.pt", force_is_torch=True))
        return self
    
    @torch.no_grad()
    def get_recons_loss(self, model, all_tokens, model_batch_size, num_batches=5):
        loss_list = []
        for i in range(num_batches):
            tokens = all_tokens[torch.randperm(len(all_tokens))[:model_batch_size]]
            loss = model(tokens, return_type="loss")
            recons_loss = model.run_with_hooks(tokens, return_type="loss", fwd_hooks=[(utils.get_act_name("post", 0), partial(replacement_hook, encoder=self))])
            # mean_abl_loss = model.run_with_hooks(tokens, return_type="loss", fwd_hooks=[(utils.get_act_name("post", 0), mean_ablate_hook)])
            zero_abl_loss = model.run_with_hooks(tokens, return_type="loss", fwd_hooks=[(utils.get_act_name("post", 0), zero_ablate_hook)])
            loss_list.append((loss, recons_loss, zero_abl_loss))
        losses = torch.tensor(loss_list)
        loss, recons_loss, zero_abl_loss = losses.mean(0).tolist()

        print(f"loss: {loss:.4f}, recons_loss: {recons_loss:.4f}, zero_abl_loss: {zero_abl_loss:.4f}")
        score = ((zero_abl_loss - recons_loss)/(zero_abl_loss - loss))
        print(f"Reconstruction Score: {score:.2%}")
        # print(f"{((zero_abl_loss - mean_abl_loss)/(zero_abl_loss - loss)).item():.2%}")
        return score, loss, recons_loss, zero_abl_loss
    
    # Frequency
    @torch.no_grad()
    def get_freqs(self, all_tokens, model, batch_size, num_batches=25):
        act_freq_scores = torch.zeros(self.d_hidden, dtype=torch.float32).to(self.device)
        total = 0
        for i in tqdm.trange(num_batches):
            tokens = all_tokens[torch.randperm(len(all_tokens))[:batch_size]]

            _, cache = model.run_with_cache(tokens, stop_at_layer=1, names_filter=utils.get_act_name("post", 0))
            mlp_acts = cache[utils.get_act_name("post", 0)]
            mlp_acts = mlp_acts.reshape(-1, self.d_mlp)

            hidden = self(mlp_acts)[2]

            act_freq_scores += (hidden > 0).sum(0)
            total+=hidden.shape[0]
        act_freq_scores /= total
        num_dead = (act_freq_scores==0).float().mean()
        print("Num dead", num_dead)
        return act_freq_scores
    
    @torch.no_grad()
    def get_ae_feature_acts(
        self,
        tokens: Int[Tensor, "batch seq"],
        model: HookedTransformer,
        reshape_acts: bool=True,
    ) -> Float[Tensor, "(batch seq) n_hidden_ae"]:
        '''
        '''
        batch_size, seq_len = tokens.shape

        logits, cache = model.run_with_cache(tokens, stop_at_layer=1, names_filter = ["blocks.0.mlp.hook_post"])
        post = cache["blocks.0.mlp.hook_post"]
        assert post.shape == (batch_size, seq_len, model.cfg.d_mlp)
        post_reshaped = einops.rearrange(post, "batch seq d_mlp -> (batch seq) d_mlp")
        assert post_reshaped.shape == (batch_size * seq_len, model.cfg.d_mlp)
        # forward returns loss, x_reconstruct, acts, l2_loss, l1_loss
        acts = self.forward(post_reshaped)[2]
        assert acts.shape == (batch_size * seq_len, self.d_hidden)
        if reshape_acts:
            acts = acts.reshape(batch_size, seq_len, self.d_hidden)
        return acts