"""Prototype: MInference-style vertical-slash sparse prefill attention for ONE layer,
implemented with torch FlexAttention (no minference package, no flash-attn, no custom CUDA).

Semantics follow MInference (Jiang et al. 2024) `vertical_and_slash_kernel`:
  1. Pattern search with the last `last_q`=64 queries: A = softmax(q_last k^T * scale) (causal on
     the last 64x64 block).  vertical score = column sums of A; slash score = sums of A along each
     diagonal (offset d = q_pos - k_pos).  First 30 columns and last 100 diagonals are forced on.
  2. Keep top-`nv` columns and top-`ns` diagonals per query head.
  3. Token mask: k <= q  and  (vert[h, k] or slash[h, q - k]).
Execution: the token mask is converted analytically (no N x N materialisation) into a FlexAttention
BlockMask (BLOCK x BLOCK tiles, default 128).  Two modes:
  exact : every touched tile is "partial" -> mask_mod applied, identical semantics to MInference.
  block : touched off-diagonal tiles are "full" (whole tile attended, a superset of the token mask
          -- denser, but no mask_mod work); diagonal tiles causal.  Cheaper per tile.
(MInference's own Triton kernel is in between: slash tiles are dense 64x64 blocks, vertical columns
are gathered exactly; FlexAttention cannot gather columns, so scattered verticals cost whole tiles.)

Drop-in for src/fastpath.py line `o = F.scaled_dot_product_attention(q, k, v, is_causal=True, ...)`:
    o = vs_attention(q, k, v, scale=att.scaling, nv=..., ns=...)
q: [B, H, N, D], k/v: [B, KV, N, D] (GQA handled by enable_gqa).

Usage:  python src/minf_proto.py [--device cuda|cpu] [--lens 4096,8192,16384] [--nv 1000 --ns 6096]
"""
import argparse, math, time
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, BlockMask

_flex_compiled = None
# one compile per (mask_mod code, N); variable-length prompts need bucketing/padding or dynamic=True
torch._dynamo.config.recompile_limit = 64


def get_flex(compile_):
    global _flex_compiled
    if not compile_:
        return flex_attention
    if _flex_compiled is None:
        _flex_compiled = torch.compile(flex_attention, dynamic=False)
    return _flex_compiled


@torch.no_grad()
def search_pattern(q, k, scale, nv, ns, last_q=64, n_sink=30, n_local=100):
    """Returns vert [B,H,N] bool (selected key columns) and slash [B,H,N] bool (selected offsets d=q-k)."""
    B, H, N, D = q.shape
    KV = k.shape[1]
    g = H // KV
    kk = k.repeat_interleave(g, 1) if g > 1 else k
    lq = min(last_q, N)
    s = torch.einsum('bhmd,bhnd->bhmn', q[:, :, -lq:].float(), kk.float()) * scale  # [B,H,lq,N]
    qpos = torch.arange(N - lq, N, device=q.device)
    kpos = torch.arange(N, device=q.device)
    s.masked_fill_(kpos[None, :] > qpos[:, None], float('-inf'))
    a = s.softmax(-1)
    vscore = a.sum(-2)                                           # [B,H,N]
    vscore[..., :n_sink] = float('inf')
    off = (qpos[:, None] - kpos[None, :]).clamp(min=0)            # [lq,N] offset d, >=0 where valid
    sscore = torch.zeros(B, H, N, device=q.device, dtype=a.dtype)
    sscore.scatter_add_(-1, off.reshape(1, 1, -1).expand(B, H, -1), a.reshape(B, H, -1))
    sscore[..., :n_local] = float('inf')
    vert = torch.zeros(B, H, N, dtype=torch.bool, device=q.device)
    vert.scatter_(-1, vscore.topk(min(nv, N), -1).indices, True)
    slash = torch.zeros(B, H, N, dtype=torch.bool, device=q.device)
    slash.scatter_(-1, sscore.topk(min(ns, N), -1).indices, True)
    return vert, slash


@torch.no_grad()
def build_block_mask(vert, slash, BS=128, mode='exact'):
    """Analytic tile mask: tile (qb,kb) is touched iff kb<=qb and (some vertical column in kb, or some
    selected offset d in [(qb-kb)*BS-BS+1, (qb-kb)*BS+BS-1])."""
    B, H, N = vert.shape
    nb = (N + BS - 1) // BS
    pad = nb * BS - N
    dev = vert.device
    colblk = F.pad(vert, (0, pad)).view(B, H, nb, BS).any(-1)            # [B,H,nb]
    cs = F.pad(slash.int().cumsum(-1), (1, 0))                            # cs[i] = #selected d < i
    delta = torch.arange(nb, device=dev)
    lo = ((delta - 1) * BS + 1).clamp(0, N)
    hi = ((delta + 1) * BS).clamp(0, N)                                   # exclusive upper bound
    diag = (cs[..., hi] - cs[..., lo]) > 0                                # [B,H,nb] per delta
    qb = torch.arange(nb, device=dev)[:, None]
    kb = torch.arange(nb, device=dev)[None, :]
    causal_blk = kb <= qb
    dmat = (qb - kb).clamp(min=0)
    touched = causal_blk & (colblk[:, :, None, :] | diag[:, :, dmat])     # [B,H,nb,nb]

    vert_i, slash_i = vert, slash

    if mode == 'exact':
        def mask_mod(b, h, qi, ki):
            return (ki <= qi) & (vert_i[b, h, ki] | slash_i[b, h, (qi - ki).clamp(min=0)])
        partial, full = touched, None
    else:
        def mask_mod(b, h, qi, ki):
            return ki <= qi
        diag_blk = (qb == kb)
        partial, full = touched & diag_blk, touched & ~diag_blk

    def to_idx(m):
        num = m.sum(-1).to(torch.int32)
        idx = m.to(torch.int8).argsort(dim=-1, descending=True, stable=True).to(torch.int32)
        return num, idx
    kn, ki_ = to_idx(partial)
    if full is not None:
        fn, fi = to_idx(full)
    else:
        fn = fi = None
    bm = BlockMask.from_kv_blocks(kn, ki_, fn, fi, BLOCK_SIZE=BS, mask_mod=mask_mod, seq_lengths=(N, N))
    density = touched.float().sum() / causal_blk.float().sum() / (B * H)
    return bm, touched, density.item()


def vs_attention(q, k, v, scale, nv=1000, ns=6096, BS=128, mode='exact', compile_=True):
    vert, slash = search_pattern(q, k, scale, nv, ns)
    bm, _, dens = build_block_mask(vert, slash, BS, mode)
    o = get_flex(compile_)(q, k, v, block_mask=bm, scale=scale, enable_gqa=(q.shape[1] != k.shape[1]))
    return o, dens


def dense_ref_mask(vert, slash, touched, BS, mode):
    """Materialised [B,H,N,N] bool mask for correctness checks (small N only)."""
    B, H, N = vert.shape
    qi = torch.arange(N, device=vert.device)[:, None]
    ki = torch.arange(N, device=vert.device)[None, :]
    causal = ki <= qi
    if mode == 'exact':
        d = (qi - ki).clamp(min=0)
        return causal & (vert[:, :, None, :] | slash[:, :, d])
    t = touched.repeat_interleave(BS, -1).repeat_interleave(BS, -2)[..., :N, :N]
    return causal & t


def make_qkv(B, H, KV, N, D, dtype, dev, structured=True, seed=0):
    """Random q/k/v; if structured, add attention sinks + a few 'retrieval' columns + locality so the
    pattern search has something to find (pure random attention is nearly uniform)."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    q = torch.randn(B, H, N, D, generator=g)
    k = torch.randn(B, KV, N, D, generator=g)
    v = torch.randn(B, KV, N, D, generator=g)
    if structured:
        u = torch.randn(D, generator=g); u = u / u.norm()
        q = q + 2.0 * u                                   # all queries share a direction
        hot = torch.randint(0, N, (16,), generator=g)
        k[:, :, :4] += 4.0 * u                            # sinks
        k[:, :, hot] += 3.0 * u                           # a few strong columns ("needles")
        pos = torch.arange(N, dtype=torch.float32)
        freqs = 1.0 / (10000 ** (torch.arange(0, D, 2).float() / D))
        ang = pos[:, None] * freqs[None]
        rot = torch.cat([ang.cos(), ang.sin()], -1)     # shared positional component -> locality
        q = q + 1.5 * rot
        k = k + 1.5 * rot
    return (q.to(dev, dtype), k.to(dev, dtype), v.to(dev, dtype))


def timeit(fn, reps=10, warm=3, dev='cuda'):
    for _ in range(warm):
        fn()
    if dev == 'cuda':
        torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn()
        if dev == 'cuda':
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2] * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--lens', default='')
    ap.add_argument('--H', type=int, default=32)
    ap.add_argument('--KV', type=int, default=8)
    ap.add_argument('--D', type=int, default=128)
    ap.add_argument('--nv', type=int, default=1000)
    ap.add_argument('--ns', type=int, default=6096)
    ap.add_argument('--BS', type=int, default=128)
    ap.add_argument('--cpu_compile', action='store_true')
    ap.add_argument('--dtype', default='fp16', help='TITAN V (sm70): Triton 3.3 fails on fp16 flex -> use fp32')
    args = ap.parse_args()
    dev = args.device
    print('torch', torch.__version__, 'device', dev,
          torch.cuda.get_device_name() if dev == 'cuda' else '', flush=True)
    dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}[args.dtype] if dev == 'cuda' else torch.float32
    compile_ = dev == 'cuda' or args.cpu_compile  # eager flex ignores block sparsity (applies mask_mod only)

    # ---- 1. correctness: flex block-sparse vs dense SDPA with the same materialised mask ----
    Nc = 2048 if dev == 'cuda' else 1024
    Hc, KVc = (8, 2)
    q, k, v = make_qkv(1, Hc, KVc, Nc, args.D, dtype, dev)
    scale = args.D ** -0.5
    nv_c, ns_c = 64, 128
    vert, slash = search_pattern(q, k, scale, nv_c, ns_c)
    kr, vr = k.repeat_interleave(Hc // KVc, 1), v.repeat_interleave(Hc // KVc, 1)
    o_dense = F.scaled_dot_product_attention(q, kr, vr, is_causal=True, scale=scale)
    for mode in ('exact', 'block'):
        bm, touched, dens = build_block_mask(vert, slash, args.BS, mode)
        o = get_flex(compile_)(q, k, v, block_mask=bm, scale=scale, enable_gqa=True)
        m = dense_ref_mask(vert, slash, touched, args.BS, mode)
        o_ref = F.scaled_dot_product_attention(q.float(), kr.float(), vr.float(), attn_mask=m, scale=scale)
        err_impl = (o.float() - o_ref).abs().max().item()
        rel_approx = ((o.float() - o_dense.float()).norm() / o_dense.float().norm()).item()
        tokdens = (m.float().sum() / (Hc * Nc * (Nc + 1) / 2)).item()
        print(f'[correct] N={Nc} mode={mode:5s} tile_density={dens:.3f} token_density={tokdens:.3f} '
              f'max|flex-maskedSDPA|={err_impl:.2e}  rel_err_vs_dense={rel_approx:.3f}', flush=True)

    if dev != 'cuda':
        print('CPU run: skipping speed benchmark'); return

    # ---- 2. speed: dense SDPA causal vs vertical-slash (search + mask build + flex) ----
    lens = [int(x) for x in args.lens.split(',')] if args.lens else [4096, 8192, 16384]
    print(f'backend check: flash sdp enabled={torch.backends.cuda.flash_sdp_enabled()} '
          f'mem_eff={torch.backends.cuda.mem_efficient_sdp_enabled()}', flush=True)
    for N in lens:
        q, k, v = make_qkv(1, args.H, args.KV, N, args.D, dtype, dev, seed=1)
        try:
            t_sdpa = timeit(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale, enable_gqa=True))
        except torch.OutOfMemoryError:  # fp32 on sm70 falls back to the O(N^2)-memory math backend
            t_sdpa = float('nan'); torch.cuda.empty_cache()
        flex = get_flex(True)
        from torch.nn.attention.flex_attention import create_block_mask
        cbm = create_block_mask(lambda b, h, qi, ki: ki <= qi, None, None, N, N, device=dev, BLOCK_SIZE=args.BS)
        t_flex_dense = timeit(lambda: flex(q, k, v, block_mask=cbm, scale=scale, enable_gqa=True))
        t_search = timeit(lambda: search_pattern(q, k, scale, args.nv, args.ns))
        vert, slash = search_pattern(q, k, scale, args.nv, args.ns)
        t_build = timeit(lambda: build_block_mask(vert, slash, args.BS, 'exact'))
        row = f'[speed] N={N:6d} sdpa_dense={t_sdpa:8.2f}ms flex_dense_causal={t_flex_dense:8.2f}ms search={t_search:6.2f}ms build={t_build:6.2f}ms'
        for mode in ('exact', 'block'):
            bm, _, dens = build_block_mask(vert, slash, args.BS, mode)
            t_k = timeit(lambda: flex(q, k, v, block_mask=bm, scale=scale, enable_gqa=True))
            tot = t_search + t_build + t_k
            row += f' | {mode}: dens={dens:.3f} kernel={t_k:7.2f}ms total={tot:7.2f}ms speedup_vs_sdpa={t_sdpa / tot:.2f}x vs_flexdense={t_flex_dense / tot:.2f}x'
        print(row, flush=True)
        del q, k, v
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
