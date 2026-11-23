"""The early-exit layer encoder must return exactly what the full forward's hidden_states[layer] returned."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.mark.parametrize("layer", [1, 2, 4])
def test_early_exit_matches_full_forward_hidden_state(layer):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from wts.encoders import LLMLayerEncoder
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    model = transformers.LlamaForCausalLM(cfg).eval()

    class Tok:
        pad_token, eos_token = "<pad>", "<pad>"

        def __call__(self, texts, **kw):
            ids = [[(ord(c) % 60) + 1 for c in t][:kw.get("max_length", 64)] for t in texts]
            n = max(map(len, ids))
            pad = [[0] * (n - len(x)) + x for x in ids]                      # left padding
            mask = [[0] * (n - len(x)) + [1] * len(x) for x in ids]
            return transformers.BatchEncoding({"input_ids": torch.tensor(pad), "attention_mask": torch.tensor(mask)})

    enc = LLMLayerEncoder(model_name="tiny", layer=layer, device="cpu", model=model, tokenizer=Tok())
    texts = ["hello world", "a much longer piece of text here", "x"]
    got = enc.raw_many(texts, batch_size=2)
    ref = []
    with torch.no_grad():
        for i in range(0, len(texts), 2):
            b = Tok()(texts[i:i + 2], max_length=1024)
            hs = model(**b, output_hidden_states=True).hidden_states[layer]
            ref += [hs[j][b["attention_mask"][j].bool()].float().numpy() for j in range(hs.shape[0])]
    assert len(got) == len(ref)
    for g, r in zip(got, ref):
        np.testing.assert_allclose(g, r, rtol=1e-5, atol=1e-6)


def test_last_position_logits_match_full_forward():
    """LlamaGuardScorer scores from logits_to_keep=1; it must equal the full forward's last-position logits."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from wts.baselines import LlamaGuardScorer
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    model = transformers.LlamaForCausalLM(cfg).eval()
    ids = torch.randint(1, 64, (1, 20))
    g = LlamaGuardScorer(model_name="tiny", device="cpu", _m=model)
    with torch.no_grad():
        ref = model(ids).logits[0, -1].float()
    torch.testing.assert_close(g._last_logits(ids), ref, rtol=1e-5, atol=1e-6)


def test_gcg_memory_savers_match_full_forward_values_and_gradients():
    """E4 GCG: target-only logits and early-exit layer states give the same loss and the same suffix gradient."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from wts.encoders import hidden_at_layer
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    model = transformers.LlamaForCausalLM(cfg).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    E = model.get_input_embeddings().weight
    prefix, suffix, tgt = torch.randint(1, 64, (6,)), torch.randint(1, 64, (5,)), torch.randint(1, 64, (4,))

    def loss(new):
        one_hot = torch.nn.functional.one_hot(suffix, 64).to(E.dtype).requires_grad_(True)
        full = torch.cat([E[prefix], one_hot @ E, E[tgt]], 0)[None]
        unit = torch.cat([E[prefix], one_hot @ E], 0)[None]
        if new:
            lg = model(inputs_embeds=full, logits_to_keep=len(tgt) + 1, use_cache=False).logits[0, :-1]
            hs = hidden_at_layer(model, 2, inputs_embeds=unit)[0]
        else:
            t0 = full.shape[1] - len(tgt)
            lg = model(inputs_embeds=full).logits[0][t0 - 1:-1]
            hs = model(inputs_embeds=unit, output_hidden_states=True).hidden_states[2][0]
        val = torch.nn.functional.cross_entropy(lg.float(), tgt) + hs.pow(2).mean()
        return val, torch.autograd.grad(val, one_hot)[0]

    (v_old, g_old), (v_new, g_new) = loss(False), loss(True)
    torch.testing.assert_close(v_new, v_old, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(g_new, g_old, rtol=1e-5, atol=1e-6)
