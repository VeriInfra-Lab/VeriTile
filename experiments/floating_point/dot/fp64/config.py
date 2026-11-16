"""FP64 operands, accumulator and result; unchanged Normal distribution/gates."""
PROFILE = {
    'rule': 'DOT-ADD-TO-DOTACC',
    'input': 'fp64', 'accumulator': 'fp64', 'output': 'fp64',
    'dot_input_precision': 'ieee', 'tile': [32, 32, 32],
    'groups_per_part': 8192,
    'distribution': {'family': 'normal', 'mean': 1.0, 'std': 1.0},
    'launch': {'num_warps': 4, 'num_stages': 3, 'enable_fp_fusion': False},
    'observation': 'one independent whole tile; mean signed output ULP shift; maximum absolute oracle error',
    'oracle': 'double-double products and K sum, followed by the live accumulator; exact rational CPU audit',
}
