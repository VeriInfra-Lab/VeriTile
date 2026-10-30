# Token attention Llama2: complete Q/K floating-point score stage

The three specifications in `FPCorrect.lean` execute the complete original
score stage from independent initial memory. They cover arbitrary dimensions,
strides, program IDs, group count, sequence metadata, and scale; the source's
`block_mask` selects zero or one loop iteration. No finite-input, positive-scale,
or fixed-size hypothesis was added. The sole layout premise is injectivity of
addresses of active output lanes. The old mathematical and IO specifications
and their source file are unchanged.

`FPTypedSource.typed .fp32` uses physical `.fp32` Q/K/output cells. Its explicit
same-format load-register and output casts preserve the original operations.
The checked `source_projection` erases those dtype/cast annotations and matches
the entire old source, including all 13 prelude statements, the dynamic loop,
B_Loc gather, K/output masks, and both multiplications. This is a control/address
projection; the software lemmas separately establish the numeric casts.
Initial registers and undef callbacks are arbitrary: the prelude and body assign
all consumed registers before use. Metadata remains the old accepted `.nat`
address model, not a newly proved fixed-width signed-int32 address interpreter.

For an active block and token lane i, let t = pid2*BN+i and
w = max_input_len-seq_len+t. The lane is active exactly when w<max_input_len.
The independent reference reads Q at pid0*sqbs+pid1*sqh+d*sqd, gathers location
B_Loc[sblb*pid0+sbls*w], and reads K at location*skbs+(pid1/groups)*skh+d*skd.
Inactive K lanes use explicit +0. For every d in [0,BD), it computes
P[d]=mul_cfg(Q[d],K[i,d]), then R=treeValue_cfg(plan[BD],P), followed by
score=mul_cfg(R,scaleBits). Every input leaf and every explicit zero padding leaf
is retained. No floating associativity, distributivity or zero-identity law is
assumed. The reference is a function of initial memory, independent of exec.
Readback and memory frame are proved for all lanes and regions; an inactive
block preserves the entire memory.

The model roles are deliberately separate:

- `token_attn_llama2_fp_backend_correctness`: arbitrary `Algebra`, arbitrary
  Real scale, all original accepted dimensions/branches. Every reached cast,
  multiplication, literal conversion and reduction is named in the reference.
- `token_attn_llama2_fp_software_correctness`: concrete fp32 bit mul/add,
  explicit width-indexed reduction trees and physical fp32 load/store identity.
  `encode : Real -> Word` is an explicit host-scalar encoding parameter.
  `score_software` removes every fallback `rest` callback from the result.
  This full arbitrary-Real contract remains backend-relative for encoding.
- `token_attn_llama2_fp_rational_correctness`: arbitrary rational q is converted
  by executable direct `FP.round .fp32 cfg q`. This is a parametric software bit
  theorem over all q, dimensions, layouts, words, and supplied valid trees,
  including exceptional Q/K bit values. It specializes the scalar domain;
  it does not replace or narrow the arbitrary-Real specification.

`rationalEncoder cfg q` is a source-specific binding of the only nonzero scalar
literal reached by this source. It is not claimed to be a general Real-to-fp32
converter: zero padding is handled independently, and the actual scale is q.
Every finite nonzero host binary64 numerical value has such an exact rational;
the original Real parameter cannot distinguish host -0 from +0 or represent
host NaN/infinity. Connecting runtime scalar bits to this model is explicit
native binding work, not an assumed accuracy result for `1/sqrt(D)`.

Native obligations remain: runtime scale argument dtype/encoding; the actual
Triton reduction schedule and padding (including layout/warp dependence); the
chosen scalar rounding/FTZ/DAZ/NaN semantics and instruction/fusion behavior;
and Q/K/output payload plus metadata/address correspondence. A supplied software
reduction tree is not asserted to be the GPU tree. `Config.ieeeValue` selects
nearest-even with input/result flushing disabled. The proof never equates
separately rounded multiplication/reduction to a fused implementation.

The ordinary `token_attn_llama2_fp_bit_success` is only a regression witness:
BD=2, BN=2, B_Loc=3 (K offsets 6/7), output base=5, one active and one masked
lane. Q=(1,2^-25), K=(1,1), and host scale=1+2^-25. Direct conversion and the
explicit seeded add tree both round to fp32 1, and the final output is bits
0x3f800000, while the exact rational score is (1+2^-25)^2 != 1. The masked
output and Q memory are preserved. This fixed witness does not upgrade the
arbitrary-Real or native classification.
