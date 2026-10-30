# Lightning attention: accepted-stage floating value proofs

The old `LightningAttention.lean` is byte-for-byte unchanged. Its three public
statements cover three forward semantic stages and a supplied-gradient store;
the original complete forward, intra-backward and inter-backward kernels appear
only in lowering clauses. This repair preserves those boundaries. It does not
claim a numeric proof of the `NUM_BLOCK` / `NUM_CBLOCK` drivers, carry production,
or backward gradient production.

## Source and scope

`FPSource.lean` contains the complete accepted KV update, OInter producer,
forward block body, and GradPre store. `FPShape.lean` proves each entire erased
surface equal to its old accepted source. The bodies preserve every index,
program ID, address, stride, causal predicate, tail load mask and store mask.
The four actual executions consume physical `.fp32` cells and preserve explicit
load-to-fp32-register / register-to-storage casts. These instantiate the original
Python test profile (`lightning_attention.py:485–487`, all torch.float32). The
`.real` channel used inside the fp32 compute context is a register carrier, not
mathematical real arithmetic or native fp64 storage. The generic DTYPE syntax is
not a claim that other physical formats have a concrete conversion proof.

The source's signed `index = off_block[:,None] - off_block[None,:] >= 0` was already
represented by the accepted body as `j <= r`; this repair retains that existing
adaptation. The old literal Nat-subtraction forward surface remains only a
lowering statement. No FP comparison or NaN assumption is used for this mask.

All four `*_execution_memory` theorems start from an arbitrary initial typed
state, establish actual successful execution, and identify the full resulting
memory with an independent store fold. They require no output assumption,
non-NaN premise, nonempty dimension, region separation, or address injectivity.
Readback/frame adds only the accepted layouts: KV/OInter row-major layouts are
proved injective from Fin bounds; the body uses `columns <= valueDim`; the
GradPre face retains output-address injectivity (the old quantified assumption
implies the new per-state form). Region aliases are allowed: all source loads
precede the stage's single store and refer to the initial memory.

## Numerical specifications

Let `D_p(A,B)` be the explicitly supplied dot backend at precision `some fp32`,
default options `{}`, and the actual matrix dimensions. It is total availability
of the primitive, not an equality assumption about its output. The independent
`word*` functions have these exact operations:

- KV: `add32(C[a,c], D_p(KTrans,V)[a,c])`.
- OInter: `D_p(Q,C)[r,c]`.
- Body: Q/K/V are the original masked loads with +0 padding; C is the supplied
  initial carry. Compute `raw = D_p(Q,Ktrans)`,
  `score[r,j] = if j <= r then raw[r,j] else +0`,
  `intra = D_p(score,V)`, `inter = D_p(Q,C)`, and
  `out[r,c] = add32(intra[r,c], inter[r,c])`. Only active time rows are stored.
- Gradient store: for each active lane, copy the initial GradPre word at the
  identical source/output address; signed zero and NaN payload bits are retained.

`software` closes all scalar addition, zero construction and same-format casts
used here. `*_software` proves the results independent of all remaining fields
of `rest`. The carry and staged K/V are arbitrary input buffers. The old exact-real
`kvClosed` invariant and its mathematical successor identity remain old theorems;
they are not silently asserted to describe a rounded carry or a native dot.

Three `*_fp32_dot_values` public specifications keep `D` explicit and therefore
have role **backend_value_contract**. Three separate `*_sequential_bits` public
specifications instantiate `D` with a fully defined software schedule: initialize
+0, visit k in increasing order, round `mul32(A[i,k],B[k,j])`, then round
`add32(acc,product)`. This is a parametric binary32 interpreter profile over all
initial bit states and dimensions, not just a fixed example. The fourth bit
specification, `gradient_payload_bits`, needs no numeric law at all. The five
ordinary fixed witnesses are supporting evidence, not the reason for these roles.

## Native binding still required

For the three arithmetic stages, bind the actual compiled default `tl.dot`
implementation on its actual matrix inputs, dimensions, precision and options.
This includes default input precision / TF32 policy, tensor-core or SIMT choice,
accumulator initialization, reduction order, FMA fusion, exceptional values and
FTZ. No equality between native dot and the increasing-k separately rounded
software schedule is assumed. Compiler contraction of `prev + dot(...)` or the
two output contributions also needs a source-to-compiled graph binding; a fused
dot-accumulator is not equated with separate dot/add. Scalar `FP.Config`, same
physical-format load/store behavior, and metadata/address machine bounds must
match the selected native profile. Exact Nat addresses do not by themselves prove
native integer overflow freedom. No exp/div primitive occurs in these accepted
stages, so no exp/div experiment is added.

No GPU numerical execution is claimed by this package. Existing input/layout and
stage limitations are not new missing numerical experiments.

## Nonempty software checks

The 2-by-2 dot examples use nonzero program IDs, nonscalar matrices and nonzero
carry. Full output formula checks yield KV `[20,33;52,69]`, OInter
`[40,60;70,100]`, and causal body `[42,63;80,113]`. Successful source runs read back
output cells and preserve an input region. A separate one-active-row body proves
masked-tail preservation; the gradient-tail witness copies `0x80000000` and
`0x7fc01234` exactly while preserving the masked output row. All concrete checks
use kernel reduction (`decide +kernel`), not `native_decide` or an external axiom.
