"""Feasibility: post-hoc closed-form YOCO-fication of Qwen2.5-1.5B-Instruct.

Sequence of n tokens, context cutoff c:
  context C = positions 1 .. c-2      (computed only through layers 0..j-1)
  queries Q = {0} ∪ [c-1, n)          (full depth; pos 0 = attention sink, c-1 = last prompt token)
For each upper layer l >= j, context keys/values (pre-RoPE) come from:
  predict : [rms(h_j), 1] @ W_j  (ridge, closed form, one solve for all upper layers)
  naive   : k_proj_l(input_layernorm_l(h_j))   (SwiftKV-style reuse without training)
  drop    : context tokens removed from upper-layer attention (eviction)
  oracle  : exact k_l, v_l (validates the custom attention path; should give KL ~ 0)
Metrics at query positions c-1 .. n-2 vs the exact model: KL(exact||variant), top-1 agreement.
"""
import os, argparse, json, math, os, sys, time
import numpy as np, pyarrow.parquet as pq, torch
import torch.nn.functional as F

ROOT = os.environ.get("CFKV_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HUB = os.path.expanduser(os.environ.get("CFKV_HF_HUB", "~/.cache/huggingface/hub"))   # directory holding the Hugging Face model cache
DEV = "cuda"


def snap(name):
    d = f"{HUB}/{name}/snapshots"
    return f"{d}/{os.listdir(d)[0]}"


MODEL = snap("models--Qwen--Qwen2.5-1.5B-Instruct")
WT = snap("datasets--Salesforce--wikitext") + "/wikitext-103-raw-v1"


def wiki_windows(tok, fname, max_rows, n_win, seq_len, seed):
    texts = pq.read_table(f"{WT}/{fname}", columns=["text"]).column("text").to_pylist()
    if max_rows:
        texts = texts[:max_rows]
    enc = tok([t for t in texts if t.strip()], add_special_tokens=False)["input_ids"]
    ids = np.fromiter((x for s in enc for x in s), dtype=np.int64)
    nw = len(ids) // seq_len
    ch = np.sort(np.random.default_rng(seed).choice(nw, size=min(n_win, nw), replace=False))
    return [ids[i * seq_len:(i + 1) * seq_len].tolist() for i in ch]


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), -1)


def rope(x, cos, sin):  # x [1,h,m,d], cos/sin [m,d]
    return x * cos[None, None] + rotate_half(x) * sin[None, None]


def rms(h):
    h = h.float()
    return h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)


def kv_pre(layer, h):
    x = layer.input_layernorm(h)
    return layer.self_attn.k_proj(x), layer.self_attn.v_proj(x)


@torch.no_grad()
def exact_pass(model, ids):
    out = model.model(input_ids=ids, output_hidden_states=True, use_cache=False)
    return out.hidden_states, model.lm_head(out.last_hidden_state)


@torch.no_grad()
def variant_logits(model, ids, c, j, mode, hs=None, W=None):
    cfg = model.config
    L, H, KV = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads
    hd = cfg.hidden_size // H
    layers = model.model.layers
    n = ids.shape[1]
    pos = torch.arange(n, device=DEV)[None]
    h = hs[j] if hs is not None else None  # exact lower layers == HF hidden_states[j]
    cos, sin = model.model.rotary_emb(h, pos)
    cos, sin = cos[0], sin[0]
    Qi = torch.cat([torch.zeros(1, dtype=torch.long, device=DEV), torch.arange(c - 1, n, device=DEV)])
    Ci = torch.arange(1, c - 1, device=DEV)
    hQ = h[:, Qi]
    hjC = h[0, Ci]
    if mode == "predict":
        pred = ((torch.cat([rms(hjC), torch.ones(len(Ci), 1, device=DEV)], 1)) @ W).to(h.dtype)
    for l in range(j, L):
        layer = layers[l]; att = layer.self_attn
        x = layer.input_layernorm(hQ)
        q = rope(att.q_proj(x).view(1, -1, H, hd).transpose(1, 2), cos[Qi], sin[Qi])
        kq = rope(att.k_proj(x).view(1, -1, KV, hd).transpose(1, 2), cos[Qi], sin[Qi])
        vq = att.v_proj(x).view(1, -1, KV, hd).transpose(1, 2)
        if mode == "drop":
            K, V, kpos = kq, vq, Qi
        else:
            if mode == "naive":
                kc, vc = kv_pre(layer, hjC)
            elif mode == "oracle":
                kc, vc = kv_pre(layer, hs[l][0, Ci])
            else:
                o = (l - j) * 2 * KV * hd
                kc, vc = pred[:, o:o + KV * hd], pred[:, o + KV * hd:o + 2 * KV * hd]
            kc = rope(kc.reshape(1, -1, KV, hd).transpose(1, 2), cos[Ci], sin[Ci])
            vc = vc.reshape(1, -1, KV, hd).transpose(1, 2)
            K, V, kpos = torch.cat([kc, kq], 2), torch.cat([vc, vq], 2), torch.cat([Ci, Qi])
        allowed = (kpos[None, :] <= Qi[:, None])[None, None]
        g = H // KV
        o_ = F.scaled_dot_product_attention(q, K.repeat_interleave(g, 1), V.repeat_interleave(g, 1),
                                            attn_mask=allowed, scale=att.scaling)
        a = hQ + att.o_proj(o_.transpose(1, 2).reshape(1, -1, H * hd))
        hQ = a + layer.mlp(layer.post_attention_layernorm(a))
    return model.lm_head(model.model.norm(hQ))[0, 1:-1]  # query positions c-1 .. n-2


def compare(lx, lv):
    lpx, lpv = F.log_softmax(lx.float(), -1), F.log_softmax(lv.float(), -1)
    kl = (lpx.exp() * (lpx - lpv)).sum(-1)
    return kl, (lx.argmax(-1) == lv.argmax(-1)).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--js", default="6,10,14,18,22")
    ap.add_argument("--alphas", default="1e-6,1e-4,1e-2")
    ap.add_argument("--n_eval", type=int, default=64)
    ap.add_argument("--n_calib_wiki", type=int, default=128)
    ap.add_argument("--n_calib_chat", type=int, default=768)
    args = ap.parse_args()
    run_dir = f"{ROOT}/results/raw/{args.run_id}"; assert not os.path.exists(run_dir); os.makedirs(run_dir)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=DEV).eval()
    model.requires_grad_(False)
    torch.set_grad_enabled(False)
    L = model.config.num_hidden_layers
    js = [int(x) for x in args.js.split(",")]

    # ---- data: (ids, c) pairs. wiki: 1024 context + 256 query. chat: context = prompt + first half of response
    def chat(split, k):
        recs = torch.load(f"{ROOT}/data/gen_{split}.pt")
        out = []
        for r in recs:
            s, p = r["prompt_ids"] + r["gen_ids"], len(r["prompt_ids"])
            if len(s) - p >= 32:
                out.append((s, p + (len(s) - p) // 2))
            if len(out) == k:
                break
        return out
    calib = [(s, len(s)) for s in wiki_windows(tok, "train-00000-of-00002.parquet", 40000, args.n_calib_wiki, 1280, 1)] \
        + [(s, len(s)) for s, _ in chat("calib", args.n_calib_chat)]
    val = [(s, 1024) for s in wiki_windows(tok, "validation-00000-of-00001.parquet", None, 8, 1280, 2)] + chat("val", 8)
    evals = {"wiki_test": [(s, 1024) for s in wiki_windows(tok, "test-00000-of-00001.parquet", None, args.n_eval, 1280, 3)]}
    for sp in ["test_alpaca", "mtbench", "gsm8k", "mbpp"]:
        evals[sp] = chat(sp, args.n_eval)
    print("data:", len(calib), "calib seqs;", {k: len(v) for k, v in evals.items()}, flush=True)

    # ---- closed-form fit: one ridge solve per j for all upper layers' pre-RoPE K,V at context positions
    tf = time.time()
    G = {j: None for j in js}; C = {j: None for j in js}; npos = 0
    layers = model.model.layers
    for s, c in calib:
        ids = torch.tensor([s], device=DEV)
        hs, _ = exact_pass(model, ids)
        sl = slice(1, c - 1)
        npos += c - 2
        for j in js:
            X = torch.cat([rms(hs[j][0, sl]), torch.ones(c - 2, 1, device=DEV)], 1).double()
            Y = torch.cat([torch.cat(kv_pre(layers[l], hs[l][0, sl]), 1) for l in range(j, L)], 1).double()
            G[j] = X.T @ X if G[j] is None else G[j] + X.T @ X
            C[j] = X.T @ Y if C[j] is None else C[j] + X.T @ Y
    torch.cuda.synchronize(); fit_sec = time.time() - tf
    print(f"accumulated grams over {npos} context positions in {fit_sec:.0f}s", flush=True)

    def solve(j, a):
        A = G[j] + a * torch.diagonal(G[j]).mean() * torch.eye(G[j].shape[0], device=DEV, dtype=torch.float64)
        return torch.cholesky_solve(C[j], torch.linalg.cholesky(A)).float()

    def run_set(seqs, j, mode, W=None):
        kls, ags = [], []
        for s, c in seqs:
            ids = torch.tensor([s], device=DEV)
            hs, lx = exact_pass(model, ids)
            lv = variant_logits(model, ids, c, j, mode, hs=hs, W=W)
            kl, ag = compare(lx[0, c - 1:len(s) - 1], lv)
            kls.append(kl); ags.append(ag)
        kl, ag = torch.cat(kls), torch.cat(ags)
        return {"kl": kl.mean().item(), "kl_p90": kl.quantile(0.9).item(), "top1": ag.mean().item()}

    res = {"run_id": args.run_id, "model": "Qwen2.5-1.5B-Instruct", "L": L, "fit_sec_all_js": fit_sec,
           "n_calib_positions": npos, "results": {}}
    # oracle check at one j: custom attention path with exact KV must reproduce the exact model
    orc = run_set(evals["wiki_test"][:8], js[len(js) // 2], "oracle")
    res["oracle_check"] = orc
    print("ORACLE (should be ~0 KL, ~1.0 top1):", orc, flush=True)
    for j in js:
        ts = time.time()
        best = None
        for a in map(float, args.alphas.split(",")):
            W = solve(j, a)
            r = run_set(val, j, "predict", W)
            if best is None or r["kl"] < best[1]:
                best = (a, r["kl"], W)
        a, _, W = best
        solve_sec = time.time() - ts
        rj = {"alpha": a, "select_sec": solve_sec, "upper_layer_fraction_skipped_for_context": (L - j) / L}
        for sp, seqs in evals.items():
            rj[sp] = {m: run_set(seqs, j, m, W if m == "predict" else None) for m in ("predict", "naive", "drop")}
        res["results"][j] = rj
        print(f"j={j:2d} (skip {(L-j)/L:.0%} of layers for context) a={a:g} | " + " | ".join(
            f"{sp}: top1 P {rj[sp]['predict']['top1']:.3f} N {rj[sp]['naive']['top1']:.3f} D {rj[sp]['drop']['top1']:.3f}"
            f" / KL P {rj[sp]['predict']['kl']:.3f} N {rj[sp]['naive']['kl']:.3f} D {rj[sp]['drop']['kl']:.3f}"
            for sp in evals), flush=True)
        json.dump(res, open(f"{run_dir}/metrics.json", "w"), indent=1)
    res["runtime_sec"] = time.time() - t0; res["gpu"] = torch.cuda.get_device_name(0)
    json.dump(res, open(f"{run_dir}/metrics.json", "w"), indent=1)
    with open(f"{ROOT}/results/registry.jsonl", "a") as f:
        f.write(json.dumps({"run_id": args.run_id, "summary": {j: {sp: {m: v[sp][m]["top1"] for m in v[sp]}
                                                                   for sp in evals} for j, v in res["results"].items()}}) + "\n")
    print("DONE", res["runtime_sec"])


if __name__ == "__main__":
    main()
