# Finite-precision source and stage correctness

The primary `source_one_block_fp32` entry interprets the complete original
three-loop body at XBLOCK=1, 0<rnumel<=RBLOCK. The proof executes each source
range once and specifies the complete memory result using an independent
ordered mean/rstd/normalization graph. It retains the original [1,RBLOCK]
reduction layout, both initial +0 accumulations, repeated casts, actual
undefined masked loads, every arithmetic operation, both barriers and all
three output stores. The type-erasure bridge compares it directly with the
original full source transcription specialized only to XBLOCK=1.

This distinction matters: the older accepted `reduce_slice` had removed
initial +0 additions and repeated casts, filled masked loads with zero, and
inserted a variance mask. Those transformations were mathematical modeling
choices. This repair adds no floating-point correctness entry for that transformed
reduction. The primary source entry uses the actual operations instead.

The primary source graph computes and stores its rounded mean, then reads
pass two from that updated memory, computes and stores rstd, then reads the
normalization inputs from the twice-updated memory. It therefore retains
actual alias behavior and the source's register mean/rstd. No premise assumes
that a stored or materialized final answer is already correct.

The original source supplies no `other` on masked reduction loads. Its tail
values are unspecified and enter the reductions. The source theorem retains
the structural state's undefined values; it does not replace them with zero
or claim the ideal mathematical LayerNorm for such tails. For rnumel=RBLOCK,
all reduction lanes are active and this undefined-value boundary disappears.
A native interpretation of partial blocks must account for its actual
undefined-lane behavior; it cannot infer tail safety from the old Real proof.

The separately accepted normalization chunk boundary materializes mean/rstd
as input channels. Its generic theorem gives independent output values and
the entire masked-store frame for any chunk index. Its software instance is
a closed binary32 ScalarOps computation: subtract, multiply twice, add, and
the actual casts. A nonempty example has one active lane and one masked tail;
adding 2^-25 to 1 stores exactly 1 while the tail remains unchanged. This
explicitly differs from identity rounding over real numbers.

The full-source primitive profile makes scalar arithmetic, Nat-to-fp32
conversion, the nearest-even encoding 0x3727c5ac of 1e-05, and an ascending
index fp32 sum from positive zero concrete. The standalone epsilon assignment
is recorded by the frontend as `literal none real`; the local profile binds
only that exact literal to its fp32 encoding, without interpreting arbitrary
untagged real arithmetic as fp32. `roundedSourceMemory` is an independent
word-level formula containing all mean/rstd/output operations; it does not
use unused rest callbacks. Rsqrt remains explicit, so the full-source
primitive entry is a backend-value contract, not a closed bit interpreter.

Native binding must establish storage/address representation, conversion and
rounding/FTZ settings, FMA contraction behavior, both actual reduction trees,
and the source's libdevice.rsqrt implementation including special values and
approximation. The epsilon encoding and integer conversion configuration must
match the compiled kernel. Retained barriers do not by themselves prove
cross-program synchronization or launch scheduling.

The original accepted single-block stage and arbitrary normalization chunk
are represented at fp32, with the above correction separating true source
from the former derived reduction. Multi-block register accumulation loops,
other storage profiles, host allocation and autograd remain outside these
contracts. No native GPU functional-correctness claim is made.
