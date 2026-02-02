#!/usr/bin/env python3
"""Generate random input values for the GPT-2 StableHLO example.

Produces a JSON file matching the @gpt2_forward signature:
  31 arguments with shapes matching the exported model.
"""

import json
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='/home/lememta/keren/examples/gpt2_inputs.json')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # GPT-2 config from the export script
    C = 128      # n_embd
    V = 256      # vocab_size
    T = 16       # seq_len
    B = 1        # batch_size
    H = 4        # n_head
    FF = 4 * C   # 512

    # Arguments in order matching @gpt2_forward:
    arg_specs = [
        ("tok_table",   (V, C)),       # %arg0: tensor<256x128xf32>
        ("pos_table",   (T, C)),       # %arg1: tensor<16x128xf32>
        ("input_ids",   (B, T)),       # %arg2: tensor<1x16xi32>  (int)
        # Layer 0
        ("ln1_scale_0", (C,)),         # %arg3
        ("ln1_bias_0",  (C,)),         # %arg4
        ("w_qkv_0",     (C, 3*C)),     # %arg5
        ("b_qkv_0",     (3*C,)),       # %arg6
        ("w_attn_proj_0", (C, C)),     # %arg7
        ("b_attn_proj_0", (C,)),       # %arg8
        ("ln2_scale_0", (C,)),         # %arg9
        ("ln2_bias_0",  (C,)),         # %arg10
        ("w_fc_0",      (C, FF)),      # %arg11
        ("b_fc_0",      (FF,)),        # %arg12
        ("w_mlp_proj_0", (FF, C)),     # %arg13
        ("b_mlp_proj_0", (C,)),        # %arg14
        # Layer 1
        ("ln1_scale_1", (C,)),         # %arg15
        ("ln1_bias_1",  (C,)),         # %arg16
        ("w_qkv_1",     (C, 3*C)),     # %arg17
        ("b_qkv_1",     (3*C,)),       # %arg18
        ("w_attn_proj_1", (C, C)),     # %arg19
        ("b_attn_proj_1", (C,)),       # %arg20
        ("ln2_scale_1", (C,)),         # %arg21
        ("ln2_bias_1",  (C,)),         # %arg22
        ("w_fc_1",      (C, FF)),      # %arg23
        ("b_fc_1",      (FF,)),        # %arg24
        ("w_mlp_proj_1", (FF, C)),     # %arg25
        ("b_mlp_proj_1", (C,)),        # %arg26
        # Final LN + LM head
        ("ln_f_scale",  (C,)),         # %arg27
        ("ln_f_bias",   (C,)),         # %arg28
        ("w_lm",        (C, V)),       # %arg29
        ("b_lm",        (V,)),         # %arg30
    ]

    inputs = []
    for name, shape in arg_specs:
        if name == "input_ids":
            # Random token IDs
            arr = rng.integers(0, V, size=shape).tolist()
        elif "scale" in name:
            # Layer norm scales ~ 1.0
            arr = (1.0 + 0.01 * rng.standard_normal(shape)).tolist()
        elif "bias" in name:
            # Biases ~ 0
            arr = (0.01 * rng.standard_normal(shape)).tolist()
        else:
            # Weight matrices: Xavier-ish init
            fan_in = shape[-1] if len(shape) > 1 else shape[0]
            std = 1.0 / np.sqrt(fan_in)
            arr = (std * rng.standard_normal(shape)).tolist()
        inputs.append(arr)

    # Round floats to reduce file size
    def round_nested(obj, decimals=4):
        if isinstance(obj, float):
            return round(obj, decimals)
        if isinstance(obj, list):
            return [round_nested(x, decimals) for x in obj]
        return obj

    inputs = round_nested(inputs)

    with open(args.output, 'w') as f:
        json.dump(inputs, f)

    size_kb = len(json.dumps(inputs)) / 1024
    print(f"Generated {len(inputs)} input tensors")
    print(f"  Output: {args.output}")
    print(f"  Size: {size_kb:.0f} KB")
    for name, shape in arg_specs:
        print(f"  {name}: {shape}")


if __name__ == '__main__':
    main()
