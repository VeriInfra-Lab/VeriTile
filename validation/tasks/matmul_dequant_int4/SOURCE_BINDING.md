# Source binding

Original source: `bench/tritonbench_g/matmul_dequant_int4/matmul_dequant_int4.py`.
Candidate: `matmul_dequant_int4_repaired.py` in this directory.

The harness compares complete function ASTs, including decorators/configurations, against the original. SHA-256 of `ast.dump(function, include_attributes=False)`:

| Function | Digest |
| --- | --- |
| `matmul4_kernel` | `ae218558a9105e12b5d0c11987abe60a8353540f7e5e952b73f85ab2bdf032bc` |
| `dequantize_kernel` | `fd254ee85699d77ea358e25427a0329c0cf3ea966d087169ca735caa5a09290d` |

The Lean semantics are source-specific scalarized recurrences and ordinary Tensor-operation semantics. They are not a generated Python/Triton AST interpreter or a certified compiler. The CPU AST harness provides a separate bounded source/control check; the Lean theorems establish the stated recurrences, storage interpretation and independent targets.

| Source operation | Formal connection |
| --- | --- |
| Literal standalone pointer strides, packed loads and masks | `SourceExecution.value/run`, `Strides.active_packed`, `SourceExecution.complete_run_eq`; `Direct.LaunchFacts` derives equality and bounds from the checked contract. |
| Eight standalone configuration occurrences and rectangular launch | `Source.dequantConfigs`, `Direct.dequantConfig/dequantGrid/dequantOrder`; all active lanes, unique ownership and coverage are proved. |
| Signed integer nibble difference, conversion and scale product | `dequantValue`, `Native.dequant16/dequant32`; independent base-16 target in `Spec.lean`. |
| Original `torch.ones`, tuning trials, final standalone launch | `Pipeline.ones/onesBytes`, current-state `Autotune.dequantTrials` and `Pipeline.trialsBytes`; final results reference original stable inputs. |
| Exact optional output and `torch.mm(a, fp_b, out=out)` | `Host.Arguments.output`, `Host.call`, `Direct.globalS1`; `Binding.libraryMetadata` fills actual strides/device and consumes `Library.Implementation.select`. |
| All 17 fused configurations, grouped program IDs, pointer increments | `Source.fusedConfigs`, `GroupedGrid`, `fusedGrid`, `Direct.rawOrder`, `Direct.rawDomain`, and `MachineDomain.fused_formed`. |
| Both fused `NO_GROUPS` branches, zero scaling, half multiply/subtract or FMA, dot and output cast | `fusedValue`, `Native.fused`, `FusedNative.call`; `Binding.Compiler.compile` supplies the selected fixed K16 ISA family and lane fusion interpretation for each trial/configuration. |
| Quantizer transpose/contiguous/view/device, reductions, zero clipping, elementwise arithmetic | `Helpers.input/groupStats`, `HelperReal`, ordinary Torch tensor-operation binding; actual helper code executes in the harness. Original operations are retained. |
| Range divided by Python scalar 15 versus Tensor/Tensor division | `Helpers.scaleDivide15` maps to `Division.scalarDenom15`; `Helpers.div` maps to `Division.tensor`, independently for FP16/FP32. |
| Quantizer packing from explicit zero, return transpose/contiguous | `Helpers.sourceWeightWord/sourceZeroWord`, base-16 packing theorems, `Strides.quantReturned`, `HelperHost.QuantAllocation`. |
| Original physical memory and actual returned tensor bases | `Direct.bind` maps original absolute bytes; concrete `Physical.halfCodec/wordCodec` have proved round trips; `Direct.globalDequant/globalS1/globalQuant/rawCall` publish only actual output spans and prove byte/input/caller frames. |
| Primitive checks and helper content validation | `Contract.lean` mirrors metadata, shape, device, disjointness, formed-index/pointer and computed-scale checks; returned native API theorems derive acceptance from successful wrapper results. |

The Real endpoints range over arbitrary independent Real input states. They do not define Real inputs by decoding FP16/FP32 bytes. Packed int32 inputs retain their exact finite payload semantics. Real quantization is totalized in Lean and proves the source arithmetic interpretation; native successful quantization additionally derives its finite-input/positive-computed-scale guards.

The numerical frontier contains only scalar conversions/arithmetic/extrema/clips/round-clamp and a fixed K16 ISA component. Compiler physical-register/lane mapping and installed Torch matrix/reduction plan selection are ordinary external implementation mappings. Those selections are consumed by the calls; no mapping is established by assuming completed output equality. This repair does not claim that the consistency scalar plan is selected by an installed Torch build.
