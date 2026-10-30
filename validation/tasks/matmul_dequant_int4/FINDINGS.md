# Findings and current evidence

- The original packed-weight helper seed was unspecified. The retained accepted repair uses explicit int32 zeros; both JIT function ASTs and their configuration lists remain unchanged.
- The source explicitly selects FP16 for the production s1 path. The existing FP32 standalone/helper profiles are preserved. No default-precision guess overrides that source choice.
- Singleton transpose/contiguous outputs can have strides `(1,rows)`. Actual strides remain in the checked contract and formed-address proofs; no normalization copies were introduced.
- Public Real and native endpoints now return full tensor identities/metadata and frames. Native inputs are original absolute-address bytes; Real inputs are independently quantified arbitrary Real states. Historical component proofs are retained as helper modules.
- Both public readers compile independently and neither reaches the other through imports. The complete local closure has 39 Lean modules and passes standard-base axiom audits on the public endpoints.
- The CPU/mocked-CUDA harness passes 66 dequant/s1 admissions, 20 helper cases, 34 fused configuration/branch cases, 65 rejections and library-failure device restoration. It pins both kernel ASTs and exercises the original helper operations.
- Native division by the Python scalar 15 and Tensor/Tensor division remain separate numerical obligations. CPU arithmetic checks do not validate the CUDA implementations of either operation.
- No current GPU execution, offline compiler artifact generation, installed Torch plan observation, or compiler lane-map extraction is claimed. Historical receipt counts and unavailable external artifact claims have been removed from the README.
- The remaining boundaries are explicit ordinary Tensor/runtime/compiler/library mappings and the inherited 26 scalar plus one fixed K16 numerical family. There is no whole-output or variable-K dot correspondence premise.
