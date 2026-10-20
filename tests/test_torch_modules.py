import torch
from anima36.attnres import BlockAttnRes
from anima36.attention import LSLinearAttention, StandardSelfAttention, TextCrossAttention
from anima36.ffn import SwiGLU
from anima36.lsconv import LSConv
from anima36.model import Anima36Core
from anima36.optim import make_adamw
from anima36.text_conditioning import LayerwiseTextProjector, stage_features_for_block

def test_lsconv_forward_backward():
    module = LSConv(dim=16, groups=8)
    x = torch.randn(2, 16, 8, 8, requires_grad=True)
    y = module(x)
    assert y.shape == x.shape and torch.isfinite(y).all()
    y.mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

def test_attention_shapes_and_gradients():
    x = torch.randn(2, 16, 32, requires_grad=True)
    linear = LSLinearAttention(32, num_heads=4, kv_stride=2)
    standard = StandardSelfAttention(32, num_heads=4)
    assert linear(x, (4, 4)).shape == x.shape
    assert standard(x).shape == x.shape
    (linear(x, (4, 4)).mean() + standard(x).mean()).backward()
    assert x.grad is not None

def test_cross_attention_mask_and_shape():
    x = torch.randn(2, 16, 32)
    context = torch.randn(2, 7, 24)
    mask = torch.tensor([[1,1,1,1,0,0,0],[1,1,1,1,1,1,0]], dtype=torch.bool)
    y = TextCrossAttention(32, 24, 4)(x, context, mask)
    assert y.shape == x.shape and torch.isfinite(y).all()

def test_attnres_and_swiglu():
    x = torch.randn(2, 5, 32)
    assert BlockAttnRes(32, 4)(x, [x, x]).shape == x.shape
    ffn = SwiGLU(32, 64)
    assert ffn(x).shape == x.shape
    assert ffn.gate_up_proj.out_features == 128

def test_layerwise_text_projector():
    projector = LayerwiseTextProjector(24, 32)
    features = {key: torch.randn(2, 7, 24) for key in ("shallow", "middle", "deep")}
    assert all(value.shape == (2, 7, 32) for value in projector(features).values())
    assert stage_features_for_block(0) == ("middle", "deep")
    assert stage_features_for_block(12) == ("shallow", "middle", "deep")
    assert stage_features_for_block(35) == ("shallow", "deep")

def test_small_full_model_forward_backward():
    model = Anima36Core(latent_channels=4, dim=32, depth=4, num_heads=4,
                        context_dim=24, patch_size=2, ffn_dim=64, kv_stride=2,
                        gradient_checkpointing=True)
    latents = torch.randn(2, 4, 8, 8, requires_grad=True)
    text = torch.randn(2, 6, 24)
    out = model(latents, torch.tensor([10., 20.]), text)
    assert out.shape == latents.shape and torch.isfinite(out).all()
    out.mean().backward()
    assert latents.grad is not None and torch.isfinite(latents.grad).all()

def test_adamw_helper_cpu():
    model = torch.nn.Linear(4, 3)
    optimizer = make_adamw(model.parameters(), lr=1e-3, fused=None)
    loss = model(torch.randn(2, 4)).square().mean()
    loss.backward()
    optimizer.step()

def test_checkpointing_matches_eager_gradients():
    torch.manual_seed(7)
    eager = Anima36Core(latent_channels=4, dim=32, depth=4, num_heads=4,
                        context_dim=24, patch_size=2, ffn_dim=64, kv_stride=2,
                        gradient_checkpointing=False)
    checked = Anima36Core(latent_channels=4, dim=32, depth=4, num_heads=4,
                          context_dim=24, patch_size=2, ffn_dim=64, kv_stride=2,
                          gradient_checkpointing=True)
    checked.load_state_dict(eager.state_dict())
    eager.train()
    checked.train()
    x1 = torch.randn(1, 4, 8, 8, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    text = torch.randn(1, 6, 24)
    timestep = torch.tensor([12.])
    y1 = eager(x1, timestep, text)
    y2 = checked(x2, timestep, text)
    torch.testing.assert_close(y1, y2, rtol=1e-5, atol=1e-5)
    y1.square().mean().backward()
    y2.square().mean().backward()
    torch.testing.assert_close(x1.grad, x2.grad, rtol=1e-4, atol=1e-5)
    for p1, p2 in zip(eager.parameters(), checked.parameters()):
        if p1.grad is not None or p2.grad is not None:
            assert p1.grad is not None and p2.grad is not None
            torch.testing.assert_close(p1.grad, p2.grad, rtol=1e-4, atol=1e-5)
