"""Causal correctness: position t never uses information from positions > t."""
import torch

from src.attention import MultiHeadAttention, causal_mask


def test_mask_blocks_future():
    m = causal_mask(5)
    assert m.tolist() == [[j <= i for j in range(5)] for i in range(5)]


def test_future_attention_weights_are_zero():
    torch.manual_seed(0)
    mha = MultiHeadAttention(16, 4).eval()
    mha.store_weights = True
    mha(torch.randn(2, 6, 16))
    future = ~causal_mask(6)
    assert (mha.last_weights[..., future] == 0).all()


def test_weights_not_stored_by_default():
    mha = MultiHeadAttention(16, 4).eval()
    mha(torch.randn(2, 6, 16))
    assert mha.last_weights is None


def test_changing_future_tokens_leaves_past_logits_unchanged(tiny_model):
    torch.manual_seed(0)
    x = torch.randint(0, 11, (2, 12))
    x2 = x.clone()
    x2[:, 7:] = torch.randint(0, 11, (2, 5))
    with torch.no_grad():
        a, _ = tiny_model(x)
        b, _ = tiny_model(x2)
    assert torch.allclose(a[:, :7], b[:, :7], atol=1e-6)
    assert not torch.allclose(a[:, 7:], b[:, 7:])


def test_prefix_loss_gradients_ignore_future(tiny_model):
    """Gradients of a loss built only from prefix logits do not depend on future tokens."""
    torch.manual_seed(0)
    x = torch.randint(0, 11, (2, 12))
    x2 = x.clone()
    x2[:, 7:] = (x2[:, 7:] + 3) % 11
    targets = torch.randint(0, 11, (2, 7))

    def grads(inp):
        tiny_model.zero_grad()
        logits, _ = tiny_model(inp)
        loss = torch.nn.functional.cross_entropy(logits[:, :7].reshape(-1, 11), targets.reshape(-1))
        loss.backward()
        return [p.grad.clone() for p in tiny_model.parameters() if p.grad is not None]

    for ga, gb in zip(grads(x), grads(x2)):
        assert torch.allclose(ga, gb, atol=1e-6)


def test_generation_is_causal(tiny_model):
    """The next sampled token depends only on the tokens already generated."""
    prompt = torch.randint(0, 11, (1, 5))
    gen = torch.Generator().manual_seed(0)
    out = tiny_model.generate(prompt, 5, temperature=0.0, generator=gen)
    logits, _ = tiny_model(out[:, :-1])
    assert torch.equal(logits[0].argmax(-1)[4:], out[0, 5:])


def test_leak_mode_sees_the_future(tiny_cfg):
    """The no-mask model (leakage demonstration) does use future tokens."""
    from dataclasses import replace

    from src.gpt import GPT

    torch.manual_seed(0)
    model = GPT(replace(tiny_cfg, causal=False)).eval()
    x = torch.randint(0, 11, (1, 12))
    x2 = x.clone()
    x2[:, 7:] = (x2[:, 7:] + 3) % 11
    with torch.no_grad():
        assert not torch.allclose(model(x)[0][:, :7], model(x2)[0][:, :7])
