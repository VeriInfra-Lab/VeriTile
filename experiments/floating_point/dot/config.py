"""Fixed local matrix-dot contraction profile; statistical settings are shared."""

PROFILE = {
    "rule": "DOT-ADD-TO-DOTACC",
    "input": "fp32",
    "accumulator": "fp32",
    "output": "fp32",
    "dot_input_precision": "tf32",
    "tile": [32, 32, 32],
    "groups_per_part": 8192,
    "distribution": {"family": "normal", "mean": 1.0, "std": 1.0},
    "launch": {"num_warps": 4, "num_stages": 3, "enable_fp_fusion": False},
    "observation": "one independent whole tile; mean signed output ULP shift; maximum absolute oracle error",
    "oracle": "FP64 products and left-to-right K sum, followed by FP64 live-accumulator add",
}
