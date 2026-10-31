# Python source provenance

The reference map selects the Python file whose complete JIT bodies are checked against each Lean source. It is a source-correspondence record, not a proof-completion receipt.

Except for VectorAdd and TritonBenchVectorAddition, these native Python examples were newly authored from the examples’ independent mathematical specifications; no recovered historical Python origin is claimed. The corresponding source files identify the numerical and invocation choices. The Lean bodies preserve these Python bodies, including casts, masks and loops.

| Example | Selected Python | Origin |
| --- | --- | --- |
| AdamUpdateGridLaunch | [adam_update_triton.py](../tritonbench_g/adam_update_triton/adam_update_triton.py) | New native Python from the independent mathematical specification |
| FlatVectorAdd | [flat_vector_add.py](FlatVectorAdd/flat_vector_add.py) | New native Python from the independent mathematical specification |
| FloatDTypeAdd | [float_dtype_add.py](FloatDTypeAdd/float_dtype_add.py) | New native Python from the independent mathematical specification |
| FloatDTypeSoftmax | [float_dtype_softmax.py](FloatDTypeSoftmax/float_dtype_softmax.py) | New native Python from the independent mathematical specification |
| FusedLayerNorm | [native_kernels.py](FusedLayerNorm/native_kernels.py) | New native Python from the independent mathematical specification |
| FusedSiLU | [fused_silu.py](FusedSiLU/fused_silu.py) | New native Python from the independent mathematical specification |
| FusedSwiglu | [native_kernels.py](FusedSwiglu/native_kernels.py) | New native Python from the independent mathematical specification |
| HyperConnectionsDepth | [native_kernels.py](HyperConnectionsDepth/native_kernels.py) | New native Python from the independent mathematical specification |
| HyperConnectionsWidth | [native_kernels.py](HyperConnectionsWidth/native_kernels.py) | New native Python from the independent mathematical specification |
| LogExp | [log_exp.py](LogExp/log_exp.py) | New native Python from the independent mathematical specification |
| OnlineSoftmax | [native_kernels.py](OnlineSoftmax/native_kernels.py) | New native Python from the independent mathematical specification |
| RowWiseMax | [row_wise_max.py](RowWiseMax/row_wise_max.py) | New native Python from the independent mathematical specification |
| RowWiseSum | [row_wise_sum.py](RowWiseSum/row_wise_sum.py) | New native Python from the independent mathematical specification |
| SoftmaxReciprocal | [softmax_reciprocal.py](SoftmaxReciprocal/softmax_reciprocal.py) | New native Python from the independent mathematical specification |
| SoftmaxStable | [softmax_stable.py](SoftmaxStable/softmax_stable.py) | New native Python from the independent mathematical specification |
| StableLogSumExp | [stable_log_sum_exp.py](StableLogSumExp/stable_log_sum_exp.py) | New native Python from the independent mathematical specification |
| TritonBenchVectorAddition | [vector_addition.py](../tritonbench_g/vector_addition/vector_addition.py) | Selected repaired TritonBench task source |
| VectorAdd | [vector_add.py](VectorAdd/vector_add.py) | Retained historical Triton source |
| Welford | [native_kernels.py](Welford/native_kernels.py) | New native Python from the independent mathematical specification |
