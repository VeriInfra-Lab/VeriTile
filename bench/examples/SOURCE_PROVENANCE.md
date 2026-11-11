# Python source provenance

The reference map selects the Python file whose complete JIT bodies are checked against each Lean source. It is a source-correspondence record, not a proof-completion receipt.

Except for VectorAdd and TritonBenchVectorAddition, these native Python examples were newly authored from the examples’ independent mathematical specifications; no recovered historical Python origin is claimed. The corresponding source files identify the numerical and invocation choices. The Lean bodies preserve these Python bodies, including casts, masks and loops.

| Example | Selected Python | Origin |
| --- | --- | --- |
| AdamUpdateGridLaunch | [AdamUpdateGridLaunch.py](AdamUpdateGridLaunch/AdamUpdateGridLaunch.py) | New native Python from the independent mathematical specification |
| FlatVectorAdd | [FlatVectorAdd.py](FlatVectorAdd/FlatVectorAdd.py) | New native Python from the independent mathematical specification |
| FloatDTypeAdd | [FloatDTypeAdd.py](FloatDTypeAdd/FloatDTypeAdd.py) | New native Python from the independent mathematical specification |
| FloatDTypeSoftmax | [FloatDTypeSoftmax.py](FloatDTypeSoftmax/FloatDTypeSoftmax.py) | New native Python from the independent mathematical specification |
| FusedLayerNorm | [FusedLayerNorm.py](FusedLayerNorm/FusedLayerNorm.py) | New native Python from the independent mathematical specification |
| FusedSiLU | [FusedSiLU.py](FusedSiLU/FusedSiLU.py) | New native Python from the independent mathematical specification |
| FusedSwiglu | [FusedSwiglu.py](FusedSwiglu/FusedSwiglu.py) | New native Python from the independent mathematical specification |
| HyperConnectionsDepth | [HyperConnectionsDepth.py](HyperConnectionsDepth/HyperConnectionsDepth.py) | New native Python from the independent mathematical specification |
| HyperConnectionsWidth | [HyperConnectionsWidth.py](HyperConnectionsWidth/HyperConnectionsWidth.py) | New native Python from the independent mathematical specification |
| LogExp | [LogExp.py](LogExp/LogExp.py) | New native Python from the independent mathematical specification |
| OnlineSoftmax | [OnlineSoftmax.py](OnlineSoftmax/OnlineSoftmax.py) | New native Python from the independent mathematical specification |
| RowWiseMax | [RowWiseMax.py](RowWiseMax/RowWiseMax.py) | New native Python from the independent mathematical specification |
| RowWiseSum | [RowWiseSum.py](RowWiseSum/RowWiseSum.py) | New native Python from the independent mathematical specification |
| SoftmaxReciprocal | [SoftmaxReciprocal.py](SoftmaxReciprocal/SoftmaxReciprocal.py) | New native Python from the independent mathematical specification |
| SoftmaxStable | [SoftmaxStable.py](SoftmaxStable/SoftmaxStable.py) | New native Python from the independent mathematical specification |
| StableLogSumExp | [StableLogSumExp.py](StableLogSumExp/StableLogSumExp.py) | New native Python from the independent mathematical specification |
| TritonBenchVectorAddition | [TritonBenchVectorAddition.py](https://github.com/VeriInfra-Lab/VeriTileBench/blob/main/bench/examples/TritonBenchVectorAddition/TritonBenchVectorAddition.py) | Selected repaired TritonBench task source |
| VectorAdd | [VectorAdd.py](VectorAdd/VectorAdd.py) | Retained historical Triton source |
| Welford | [Welford.py](Welford/Welford.py) | New native Python from the independent mathematical specification |
