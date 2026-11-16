"""Current two-gates settings; each configuration runs once to its fixed budget.

Execution shape bounds memory, not the statistical sample count. Only the three
primary FP32 cancellation-sensitive rules use the validated weighted sampler.
"""

SETTINGS = {
    "samples": 2**32,
    "near_samples": 2**25,
    "execution_shape": [1024, 1024],
    "seed": 20261114,
    "workers": 3,
    "audit_parts": 16,
    "bias_tolerance_ulp": 0.05,
    "bias_se_multiplier": 5.0,
    "peak_ratio_limit": 10.0,
    "near_half_width": 2.0**-12,
    "rejection_envelope": 1.001,
}
