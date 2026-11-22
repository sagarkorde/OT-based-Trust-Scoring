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
