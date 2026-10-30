# Fused rotary embedding with a checked static tile

This separate candidate adds one live-path guard to the canonical repair:
`full_block = triton.next_power_of_2(width)` must be at most `1048576`.
The original `tokens == 0` return precedes this guard, pointer checks, metadata
copies and launch. The full Python module AST is unchanged after removing this
single guard. Both K-cache layouts and the entire JIT body are preserved.

`Kernels.lean` contains the complete original and candidate raw Python function
bodies. Nine pointer dtypes, 27 runtime integer arguments and four constexpr
arguments are declared and bound outside each body. The selected original JIT
and wrapper are unchanged; `candidate_kernels.py` is newly authored from the
existing shared-Q-offset Lean candidate, with explicit provenance.
`jit-correspondence.json` checks both complete bodies, casts, branches and calls. `Proofs/Execution`
constructs successful execution without an execution-success premise.
`Proofs/Program`, `Natural`, `Scatter`, and `Connection` derive its exact masked
Q/K/V stores and connect each program to the original proved source model.
`Proofs/Grid` threads every program's resulting memory through the full grid.
The public `Correct.rotaryIO` declares typed Q/K/V, coefficient and int32
metadata inputs, the three complete logical output families, in-place Q,
indirect cache windows, and the complete source launch schedule.
`Correct.fused_rotary_real` uses the shared `ProgramIO.ImplementsOn` judgment
and an independent split-half formula. `Config.Valid` supplies kernel geometry
and write separation; `PublicIO.metadataDomain` pins the actual metadata
channels to that configuration without constraining unused table values.

`FPEquiv.rotaryNativeIO`, retained in `Misc/NativeCorrect.lean`, has the same
memory/schedule interface with unrestricted
FP32 word inputs and per-output Q/K/V observation. Its independent word formula
retains the different Q/K multiply/FMA orders and V's raw copy. Primitive
instruction semantics remain the existing local model boundary; no numerical
transformation or native-to-Real equality was added.

`Proofs/RawMemory`, `RawProgram` and `RawGrid` connect the actual source to the
shared schedule on arbitrary typed initial memory. They reuse the existing
source execution theorem, preserve exact untouched cells and their type tags,
and prove every Q/K/V output has its source storage type. `IOLayout`, `IOBounds`
and `IOResults` connect public inputs, actual access bounds, mathematical
results and frame; `PublicIO` composes them into the shared judgment.
The former raw statements are ordinary helper theorems in `RealExecution`
and `NativeExecution`.

`TensorBridge.fused_rotary_real` and `TensorBridge.fused_rotary_native` retain the
complete admitted Tensor applications, including byte/readback frames, return
identity, caller device and frontend tile bounds. The Real application reuses
the preserved `RealExecution` source theorem; the native application uses the same existing raw
source result through `NativeSource.complete_bytes`, preserving its explicit
word-to-byte correspondence. Primary entries do not import the bridge.

See `KernelScope.md` for the admitted domain and local model boundaries.
The original repair and its frozen frontend counterexample are preserved.


The 2026-10-21 actual pair compares `Source.kernel` with the new complete
ordinary quotation `Source.kernelSharedQOffset`. No distinct historical JIT
candidate existed. The new directly related candidate changes only the second
Q-half address:

```
off_q1 = off_q + dim_range1 * q_dim_stride
→ off_q1 = off_q0 + (HEAD_DIM // 2) * q_dim_stride
```

Since `off_q0 = off_q + dim_range0*q_dim_stride` and
`dim_range1 = dim_range0 + HEAD_DIM/2`, the signed-integer identity is exact.
The half dimension is constexpr; `q_dim_stride` remains a runtime integer
argument. Reusing the first-half address removes the second per-lane stride
multiplication. No measured speedup is claimed.
`Proofs/SourcePair.lean` checks the complete initial statement block and identical
suffix, proving full execution equality and source-safety preservation for
arbitrary structural Algebra. It requires no floating arithmetic law.

`FPEquiv.kernel_equivalent rest p c valid` uses shared
`ProgramIO.EquivalentOn` for the actual original/candidate complete grid.
Both executions are safe and succeed, all three active typed Q/K/V output
families agree, and both exterior frames hold. The original Config.Valid and
metadataDomain are unchanged, including unused table entries, arbitrary FP32
input words, in-place Q, K/V conditional leader programs, both K-cache layouts,
independent coefficient pitches, original even widths, full/padded dimension masks and empty grids.
All Q/K multiplication and FMA orders and V raw copies remain intact. No
experimental numerical rule or new host/hardware premise is introduced.
Single-source native declarations remain ordinary results in
`Misc/NativeCorrect.lean`; the independent Real formula remains unchanged.

Optional reader and application modules are in `Misc/`: [Misc/TensorBridge.lean](Misc/TensorBridge.lean), [Misc/NativeCorrect.lean](Misc/NativeCorrect.lean). Their declaration namespaces and proof statements are preserved. The main `Contract.lean`, `Correct.lean` and `FPEquiv.lean` entries remain at the task root.

`IO.lean` is the canonical `PublicIO.io` data description. `Correct.rotaryIO` only specializes its Real interpretation. The shared invocation ABI and native pair variants are in `InvocationIO.lean` and `PairIO.lean`; layout, safety and equivalence proofs remain in `Proofs/`.
