# triton_linear_activation — separate whole-source repair

The repaired Python source preserves conditional FP16 AMP computation and fixes loss of writes to the caller's FP32 saved buffer with a private holder and explicit scratch copyback. It also fixes ragged store ownership, false wrapped-gather hints, constexpr literal compatibility, empty launches, rank-three output reshaping and CUDA device scope. All 98 original configurations and default compiler arithmetic choices remain.

The public caller domain is canonical FP32 CUDA X[M,K] or X[B,S,K], W[N,K], optional bias[N] and saved[M,N], same device, zero offsets, no lazy flags, distinct nonempty allocations and the concrete source signed32 bounds checked by the wrapper. Empty M/N, K=0, all named activations and identity fallback are admitted.

Kernels.lean exposes the complete operational host pipeline. Correct.lean proves the independent dense Real result, every saved cell, exact caller frame and host metadata. FPEquiv.lean proves the selected raw-word reference and native event composition; its only numerical equalities are nine scalar/cast leaves and one family of actual fixed K8/K16 ISA result lanes. The conditional AMP branch preserves half internal saved rounding before FP32 copyback.

Proofs/Geometry, Config, Bounds, Machine and SourceLink cover every configuration's exact grouped schedule and formed addresses. Arithmetic proves all K terms. Source, Grid, Events, Stores and EventGrid derive current-memory execution and stores. Copy, Materialize, HostTypes, HostData, HostRun and HostReturn connect local casts/stores to caller-visible output, saved, ctx and frame. Misc/Reader.lean instantiates a valid domain for both ranks, both modes and all flags/configs and checks axiom footprints.

Native scalar/ISA conformance remains unvalidated, including exceptional bit patterns. The selected compiler evidence is Triton 3.7.1 SM90, 208 offline profiles, with default TF32 dot and default fusion. No GPU numerical execution occurred. The Python/Triton-to-Lean binding and ordinary local runtime/memory interface are explicit interpretation boundaries. Backward is absent in the original API and is not invented. See the producer FINAL-CONTRACT.md and FPValueScope.md for the exact numerical scope.

The public execution constructs device-return metadata; the separate deviceCall restoration lemma is not called by execute/nativeResult. Byte corollaries project wordBytes from abi.observe rather than executing a separate raw-byte interface. Actual Python device-context behavior and the correspondence of observed words/bytes to tensor storage remain ordinary API interpretation boundaries.

All public specifications include arbitrary finite autotune trial histories. Proofs/Autotune proves that the final selected launch absorbs every prior trial's output/saved writes while preserving input memory, without assuming numerical equality between configurations.
