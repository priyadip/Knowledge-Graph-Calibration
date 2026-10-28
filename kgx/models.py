"""The three further model families: registry with pinned revisions, loading, FFN access, and the start-of-text token.

Every model has a gated feed-forward block, down_proj(act(gate_proj(x)) * up_proj(x)), so the units, the selection
score and the repaired consumer (down_proj) are defined exactly as for Qwen2.5 and OLMo-2.
- Llama 3.1 8B and Mistral 7B v0.3: SwiGLU, used as loaded.
- Phi-4: SwiGLU with gate and up fused in one matrix (gate_up_proj, gate rows first). Each MLP is replaced by an
  equivalent module with separate gate_proj and up_proj (same weights, same function).

Start-of-text token. Qwen2.5 and OLMo-2 add no special token to a text, and neither does Phi-4. Llama 3.1 and Mistral
prepend a BOS token to every text and expect it at position 0. For these families every token sequence the
model sees starts with BOS (calibration sequences, perplexity windows, zero-shot contexts; generation prompts already
get it from the tokenizer). For families without BOS nothing changes, so the protocol of kgcal is the special case
bos = None.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn

REGISTRY = {
    # Llama 3.1 is gated on the official repo; this copy is checked in download_models.py
    "llama8b": {"repo": "unsloth/Meta-Llama-3.1-8B", "rev": "e9a141a2091ea561b96483212645a2a05e6f99fc",
                "label": "Llama-3.1-8B", "official": "meta-llama/Llama-3.1-8B"},
    "mistral7b": {"repo": "mistralai/Mistral-7B-v0.3", "rev": "caa1feb0e54d415e2df31207e5f4e273e33509b1",
                  "label": "Mistral-7B-v0.3", "official": "mistralai/Mistral-7B-v0.3"},
    "phi4": {"repo": "microsoft/phi-4", "rev": "2db69c1c3e91a05d2c64a3185acfbaf36f744e25",
             "label": "Phi-4", "official": "microsoft/phi-4"},
}


class GatedMLP(nn.Module):
    """down_proj(act_fn(gate_proj(x)) * up_proj(x)), the FFN layout of the other families."""

    def __init__(self, gate_proj: nn.Linear, up_proj: nn.Linear, down_proj: nn.Linear, act_fn):
        super().__init__()
        self.gate_proj, self.up_proj, self.down_proj, self.act_fn = gate_proj, up_proj, down_proj, act_fn

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def _split_fused_mlps(model) -> None:
    """Phi-3 family: gate_up_proj (2N x d, gate first, as in Phi3MLP.forward) -> gate_proj and up_proj."""
    for layer in decoder_layers(model):
        m = layer.mlp
        if not hasattr(m, "gate_up_proj"):
            continue
        W = m.gate_up_proj.weight.data
        N, d = W.shape[0] // 2, W.shape[1]
        gate = nn.Linear(d, N, bias=False, device=W.device, dtype=W.dtype)
        up = nn.Linear(d, N, bias=False, device=W.device, dtype=W.dtype)
        gate.weight.data = W[:N].clone()
        up.weight.data = W[N:].clone()
        layer.mlp = GatedMLP(gate, up, m.down_proj, m.activation_fn)
        m.gate_up_proj.weight.data = torch.empty(0, device=W.device, dtype=W.dtype)   # free the fused copy now
        del m, W


def text_config(model):
    return model.config.get_text_config()


def decoder_layers(model):
    return model.model.layers


def bos_id(tok) -> int | None:
    """The BOS id if the tokenizer starts every text with it, else None (Qwen2.5, OLMo-2, Phi-4)."""
    ids = tok("a")["input_ids"]
    b = tok.bos_token_id
    return b if (b is not None and len(ids) > 0 and ids[0] == b) else None


def place_on_two_gpus(model, n_first: int) -> None:
    """Blocks 0..n_first-1, the embeddings, the final norm and the head on cuda:0; the other blocks on cuda:1.

    Used only when one GPU has no room for the model. Direct GPU-to-GPU copies were corrupted on the machine we used
    (peer access was reported but the copied data were wrong; copies through host memory were exact), so
    every tensor that changes GPU goes through the CPU. A forward pre-hook does this for the inputs of every block, of
    the final norm and of the head, so each block computes exactly what it computes on a single GPU; the head sits on
    cuda:0 so that the library never moves logits between GPUs itself.
    """
    d0, d1 = torch.device("cuda:0"), torch.device("cuda:1")
    layers = decoder_layers(model)
    model.model.embed_tokens.to(d0)
    model.model.rotary_emb.to(d0)
    model.model.norm.to(d0)
    model.lm_head.to(d0)

    def mover(dev):
        def mv(x):
            if torch.is_tensor(x):
                if x.device == dev:
                    return x
                return x.to("cpu").to(dev) if x.is_cuda and dev.type == "cuda" else x.to(dev)
            if isinstance(x, tuple):
                return tuple(mv(y) for y in x)
            return x

        def pre(mod, args, kwargs):
            return tuple(mv(a) for a in args), {k: mv(v) for k, v in kwargs.items()}
        return pre
    for i, layer in enumerate(layers):
        dev = d0 if i < n_first else d1
        layer.to(dev)
        layer.register_forward_pre_hook(mover(dev), with_kwargs=True)
    model.model.norm.register_forward_pre_hook(mover(d0), with_kwargs=True)
    model.lm_head.register_forward_pre_hook(mover(d0), with_kwargs=True)


def load(tag: str, device: str = "cuda"):
    """Model in bf16 on `device`, tokenizer (pad = eos if missing, as in kgcal), and a provenance dict."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    spec = REGISTRY[tag]
    tok = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["rev"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    split = os.environ.get("KGX_SPLIT") == "1"
    model = AutoModelForCausalLM.from_pretrained(spec["repo"], revision=spec["rev"], dtype=torch.bfloat16).eval()
    if split:
        _split_fused_mlps(model)                              # on the CPU, then the blocks are placed
        place_on_two_gpus(model, int(os.environ.get("KGX_SPLIT_FIRST", "19")))
    else:
        model = model.to(device)
        _split_fused_mlps(model)
    meta = {"tag": tag, "repo": spec["repo"], "rev": spec["rev"], "official": spec["official"],
            "arch": type(model).__name__, "bos": bos_id(tok), "split": split}
    return model, tok, meta
