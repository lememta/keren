#!/usr/bin/env python3
"""Export a GPT-2 style transformer to StableHLO MLIR with hierarchy preserved.

JAX inlines everything into one flat function. To preserve the model's logical
structure (embedding, layer_norm, attention, mlp, lm_head), we lower each
component separately and emit a module with multiple func.func entries
connected by call ops.
"""

import re
import os
import argparse
import jax
import jax.numpy as jnp
import numpy as np


# ── Pure-function components (no Flax, just JAX) ──────────────────────

def layer_norm(x, scale, bias, eps=1e-6):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return scale * (x - mean) / jnp.sqrt(var + eps) + bias


def embedding_lookup(table, ids):
    return table[ids]


def causal_self_attention(x, w_qkv, b_qkv, w_proj, b_proj, n_head):
    B, T, C = x.shape
    head_dim = C // n_head

    qkv = x @ w_qkv + b_qkv
    q, k, v = jnp.split(qkv, 3, axis=-1)

    q = q.reshape(B, T, n_head, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(B, T, n_head, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(B, T, n_head, head_dim).transpose(0, 2, 1, 3)

    scale = jnp.sqrt(jnp.float32(head_dim))
    attn = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / scale

    mask = jnp.tril(jnp.ones((T, T)))
    attn = jnp.where(mask == 0, jnp.float32(-1e9), attn)
    attn = jax.nn.softmax(attn, axis=-1)

    out = jnp.matmul(attn, v)
    out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
    out = out @ w_proj + b_proj
    return out


def mlp(x, w_fc, b_fc, w_proj, b_proj):
    h = x @ w_fc + b_fc
    # GELU approximation
    h = h * 0.5 * (1.0 + jnp.tanh(
        jnp.sqrt(2.0 / jnp.pi) * (h + 0.044715 * h ** 3)))
    return h @ w_proj + b_proj


def transformer_block(x, ln1_s, ln1_b, w_qkv, b_qkv, w_attn_proj, b_attn_proj,
                       ln2_s, ln2_b, w_fc, b_fc, w_mlp_proj, b_mlp_proj, n_head):
    normed = layer_norm(x, ln1_s, ln1_b)
    x = x + causal_self_attention(normed, w_qkv, b_qkv, w_attn_proj, b_attn_proj, n_head)
    normed = layer_norm(x, ln2_s, ln2_b)
    x = x + mlp(normed, w_fc, b_fc, w_mlp_proj, b_mlp_proj)
    return x


def lm_head(x, w, b):
    return x @ w + b


# ── StableHLO extraction helpers ──────────────────────────────────────

def extract_all_funcs(stablehlo_text):
    """Extract all func.func definitions from lowered StableHLO text."""
    lines = stablehlo_text.split('\n')
    funcs = []
    current = []
    inside = False
    brace_depth = 0
    for line in lines:
        if not inside and 'func.func' in line:
            inside = True
            brace_depth = 0
            current = []
        if inside:
            current.append(line)
            brace_depth += line.count('{') - line.count('}')
            if brace_depth <= 0 and len(current) > 1:
                funcs.append('\n'.join(current))
                inside = False
                current = []
    return funcs


def lower_component(fn, args, name):
    """Lower a JAX function to StableHLO and extract all func bodies.

    JAX may generate helper functions (e.g. @tril, @_where) alongside @main.
    We extract all of them, rename @main to the given name, and prefix
    helpers with the component name to avoid collisions across components.
    """
    lowered = jax.jit(fn).lower(*args)
    text = lowered.as_text("stablehlo")
    funcs = extract_all_funcs(text)

    # Identify helper function names (everything that's not @main)
    helper_names = []
    for f in funcs:
        m = re.search(r'func\.func\s+(?:public\s+|private\s+)?@(\w+)', f)
        if m and m.group(1) != 'main':
            helper_names.append(m.group(1))

    # Build the combined output: rename @main, prefix helpers
    parts = []
    for f in funcs:
        # Rename helpers to avoid cross-component collisions
        for hname in helper_names:
            f = re.sub(r'@' + re.escape(hname) + r'(?=[\s(])', f'@{name}__{hname}', f)
        # Rename @main to @<name>
        f = f.replace('func.func public @main', f'func.func @{name}', 1)
        # Strip mhlo/jax attributes
        f = re.sub(r'\s*\{mhlo\.layout_mode\s*=\s*"[^"]*"\}', '', f)
        f = re.sub(r'\s*\{jax\.result_info\s*=\s*"[^"]*",\s*mhlo\.layout_mode\s*=\s*"[^"]*"\}', '', f)
        parts.append(f)

    return '\n\n'.join(parts)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Export GPT-2 StableHLO with hierarchy')
    parser.add_argument('--output', default='/home/lememta/keren/examples/gpt2.mlir')
    parser.add_argument('--n-head', type=int, default=4)
    parser.add_argument('--n-embd', type=int, default=128)
    parser.add_argument('--n-layer', type=int, default=2)
    parser.add_argument('--seq-len', type=int, default=16)
    parser.add_argument('--vocab-size', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=1)
    args = parser.parse_args()

    B = args.batch_size
    T = args.seq_len
    C = args.n_embd
    V = args.vocab_size
    H = args.n_head
    FF = 4 * C

    rng = jax.random.PRNGKey(42)

    # Create dummy tensors for each component
    x_3d = jnp.zeros((B, T, C), dtype=jnp.float32)
    scale_1d = jnp.ones((C,), dtype=jnp.float32)
    bias_1d = jnp.zeros((C,), dtype=jnp.float32)
    ids = jnp.zeros((B, T), dtype=jnp.int32)
    table_tok = jnp.zeros((V, C), dtype=jnp.float32)
    table_pos = jnp.zeros((T, C), dtype=jnp.float32)

    # Attention weights
    w_qkv = jnp.zeros((C, 3 * C), dtype=jnp.float32)
    b_qkv = jnp.zeros((3 * C,), dtype=jnp.float32)
    w_attn_proj = jnp.zeros((C, C), dtype=jnp.float32)
    b_attn_proj = jnp.zeros((C,), dtype=jnp.float32)

    # MLP weights
    w_fc = jnp.zeros((C, FF), dtype=jnp.float32)
    b_fc = jnp.zeros((FF,), dtype=jnp.float32)
    w_mlp_proj = jnp.zeros((FF, C), dtype=jnp.float32)
    b_mlp_proj = jnp.zeros((C,), dtype=jnp.float32)

    # LM head weights
    w_lm = jnp.zeros((C, V), dtype=jnp.float32)
    b_lm = jnp.zeros((V,), dtype=jnp.float32)

    # -- Lower each component separately --
    print("Lowering components...")

    fn_embed = lower_component(
        embedding_lookup, (table_tok, ids), 'token_embedding')

    fn_pos_embed = lower_component(
        embedding_lookup, (table_pos, jnp.arange(T, dtype=jnp.int32)), 'position_embedding')

    fn_layer_norm = lower_component(
        layer_norm, (x_3d, scale_1d, bias_1d), 'layer_norm')

    # For attention, we need to pass n_head as a static argument
    def attn_wrapper(x, w_qkv, b_qkv, w_proj, b_proj):
        return causal_self_attention(x, w_qkv, b_qkv, w_proj, b_proj, H)

    fn_attention = lower_component(
        attn_wrapper, (x_3d, w_qkv, b_qkv, w_attn_proj, b_attn_proj),
        'causal_self_attention')

    fn_mlp = lower_component(
        mlp, (x_3d, w_fc, b_fc, w_mlp_proj, b_mlp_proj), 'mlp_block')

    def block_wrapper(x, ln1_s, ln1_b, w_qkv, b_qkv, w_ap, b_ap,
                      ln2_s, ln2_b, w_fc, b_fc, w_mp, b_mp):
        return transformer_block(x, ln1_s, ln1_b, w_qkv, b_qkv, w_ap, b_ap,
                                  ln2_s, ln2_b, w_fc, b_fc, w_mp, b_mp, H)

    fn_block = lower_component(
        block_wrapper,
        (x_3d, scale_1d, bias_1d, w_qkv, b_qkv, w_attn_proj, b_attn_proj,
         scale_1d, bias_1d, w_fc, b_fc, w_mlp_proj, b_mlp_proj),
        'transformer_block')

    fn_lm_head = lower_component(
        lm_head, (x_3d, w_lm, b_lm), 'lm_head')

    # -- Also lower the full forward pass for reference --
    def full_forward(tok_table, pos_table, input_ids,
                     # Per-layer weights (layer 0)
                     ln1_s_0, ln1_b_0, w_qkv_0, b_qkv_0, w_ap_0, b_ap_0,
                     ln2_s_0, ln2_b_0, w_fc_0, b_fc_0, w_mp_0, b_mp_0,
                     # Per-layer weights (layer 1)
                     ln1_s_1, ln1_b_1, w_qkv_1, b_qkv_1, w_ap_1, b_ap_1,
                     ln2_s_1, ln2_b_1, w_fc_1, b_fc_1, w_mp_1, b_mp_1,
                     # Final layer norm + lm head
                     ln_f_s, ln_f_b, w_lm, b_lm):
        tok_emb = tok_table[input_ids]
        pos_ids = jnp.arange(input_ids.shape[1], dtype=jnp.int32)
        pos_emb = pos_table[pos_ids]
        x = tok_emb + pos_emb

        x = transformer_block(x, ln1_s_0, ln1_b_0, w_qkv_0, b_qkv_0, w_ap_0, b_ap_0,
                               ln2_s_0, ln2_b_0, w_fc_0, b_fc_0, w_mp_0, b_mp_0, H)
        x = transformer_block(x, ln1_s_1, ln1_b_1, w_qkv_1, b_qkv_1, w_ap_1, b_ap_1,
                               ln2_s_1, ln2_b_1, w_fc_1, b_fc_1, w_mp_1, b_mp_1, H)

        x = layer_norm(x, ln_f_s, ln_f_b)
        logits = x @ w_lm + b_lm
        return logits

    fn_forward = lower_component(
        full_forward,
        (table_tok, table_pos, ids,
         scale_1d, bias_1d, w_qkv, b_qkv, w_attn_proj, b_attn_proj,
         scale_1d, bias_1d, w_fc, b_fc, w_mlp_proj, b_mlp_proj,
         scale_1d, bias_1d, w_qkv, b_qkv, w_attn_proj, b_attn_proj,
         scale_1d, bias_1d, w_fc, b_fc, w_mlp_proj, b_mlp_proj,
         scale_1d, bias_1d, w_lm, b_lm),
        'gpt2_forward')

    # -- Assemble the module --
    module_lines = [
        'module @gpt2 {',
        '',
        '// ── Token & Position Embedding ──',
        fn_embed,
        '',
        fn_pos_embed,
        '',
        '// ── Layer Normalization ──',
        fn_layer_norm,
        '',
        '// ── Causal Self-Attention ──',
        fn_attention,
        '',
        '// ── Feed-Forward MLP ──',
        fn_mlp,
        '',
        '// ── Transformer Block (attention + mlp with residuals) ──',
        fn_block,
        '',
        '// ── Language Model Head ──',
        fn_lm_head,
        '',
        '// ── Full Forward Pass ──',
        fn_forward,
        '',
        '}  // module @gpt2',
    ]

    output = '\n'.join(module_lines)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        f.write(output)

    n_lines = output.count('\n')
    n_funcs = output.count('func.func')
    n_ops = output.count('stablehlo.')
    print(f"\nExported GPT-2 ({args.n_layer}L, {H}H, {C}D, vocab={V})")
    print(f"  Functions: {n_funcs}")
    print(f"  StableHLO ops: ~{n_ops}")
    print(f"  Lines: {n_lines}")
    print(f"  Output: {args.output}")


if __name__ == '__main__':
    main()
