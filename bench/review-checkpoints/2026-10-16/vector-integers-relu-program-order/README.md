# Vector integer grids, floating program order and original ReLU scope

This batch adds fourteen Lean modules and three public specifications across
two tasks. Cumulative campaign extensions now contain23 contracts across10
tasks. **Whole-original-program acceptance remains0/183; all183 are pending.**
Every original stage, dtype and branch is still required. Numerical validation
may be deferred only with explicit scope; missing structural coverage remains
open. Earlier public declarations, their metadata and source views are retained.

`vector_addition_integer_grid_correctness` covers all81 ordered canonical
integer/bool pairs. It constructs actual complete execution, using shared
fixed-width cast expressions and concrete BitVec promotion, modular addition
and conversion to x's output dtype. Bool loads compare a raw logical int cell
with zero; bool+bool wraps at unsigned i1 before storing signed i8. Bool output
from other integer combinations stores the converted byte, without normalizing
every nonzero byte to one. No floating callback occurs in this value graph.

The complete ascending grid gives the initial-memory result and full frame,
including empty inputs and partial final tiles, under positive natural block
size. Every program reads preceding actual memory. Flat tile disjointness
permits equal logical input/output region names. Int/Nat source cells remain
logical decoded values; native widths, representability, tags, storage bytes,
strides, shifted/mixed-width views and finite-width addresses remain unbound.
The frozen ordinary theorem is preserved; a sixth, thin source registers it
as a public specification with unchanged premises. No new closed-interpreter
audit command or native conformance result is claimed.

The two vector dtype batches prove106/217 promotion-success pairs:25 floating
and81 integer/bool. Remaining logical execution coverage is111 pairs:72 mixed
integer/float,11 fp64 floating and28 fnuz pairs. The new exact-source frontend
matrix observes189 generated TTIRs and100 errors across289 pairs on the pinned
sm90 configuration. The promotion rule separately observes217 successes and72
errors. All64 pairs involving fnuz are rejected by this frontend configuration,
including28 promotion successes. These observations are not backend compilation,
GPU execution or universal error proofs. All earlier13 probe observations stay
unchanged in the preceding checkpoint.

`vector_addition_typed_program_order_correctness` extends the same25 floating
pairs to any permutation of the complete grid. The runner executes the supplied
order on current memory with fresh program registers. An offset's quotient by
block size identifies its owner; a duplicate-free processed prefix cannot have
written the next owner's inputs. Its independent membership-based reference
equals the prior floating-grid specification at every address. Repeated IDs
are excluded because in-place logical aliases can otherwise change inputs.
Floating add, including fp32, and distinct-format casts remain arbitrary. No
new arithmetic law is assumed, and native instruction interleavings remain open.

`relu_typed_grid_correctness` covers twelve decoded logical profiles: eight
signed/unsigned width labels, Bool, fp16, bf16 and fp32, with fp32 output. The
original source writes only when pid==0. For block1024, every output offset
at or beyond1024 remains untouched, including caller lengths beyond the four
five-element harness cases. Every later program still executes its original
loads and comparison. The tail reference is arbitrary memory immediately
before launch, after allocation and metadata callbacks; it is not zero-filled.

ReLU retains ordered comparison and select rather than rewriting them to max.
fp16 compares/selects in fp16 then widens; bf16 widens for comparison, selects
in bf16 and then widens. The concrete fp32 software helper checks NaNs, signed
zero, infinities and all20 harness literal words; selected negative zero is
preserved. It does not instantiate the public arbitrary comparison, zero and
conversion callbacks. Integer/bool labels use decoded Int/Nat/Bool cells; raw
native widths and bool i8 normalization still need structural correspondence.

ReLU's fallible host helper preserves x.device access, empty_like's fp32/device
request, output.numel, the unused num_blocks calculation, launch and return.
It composes actual typed-grid success after preceding primitive outcomes.
The four constructor/call/dictionary-insertion cases and result_gold assignment
retain their source order and error propagation. logicalLaunch consumes dtype
and regions while ignoring physical metadata. Its unsupportedLogicalProfile
error is an interpreter boundary, not a native compiler rejection theorem.
Native allocator, accessor, tensor construction and Python execution remain
unbound. fp64 frontend generation succeeds but has no typed execution proof;
four fp8 frontend errors and other dtype domains remain structural work.

Masked loads in both originals omit `other`; inactive values remain unspecified.
No active-output proof assumes inactive values are zero or finite. The shared
DSL source views select only unchanged original add_kernel/relu_kernel. Manual
typed ASTs are reviewed separately. ReLU's dtype/cast-erasure proof covers its
three floating materializations; no integer erasure or native lowering proof
is inferred. No original source bug or absent guard was repaired.

[root-review.json](root-review.json) binds the exact sources, explicit stage/dtype
metadata and independent reviews. The three frozen producer batches passed all
13 module compilations,63 named axiom checks and10 noncircularity checks. The
separate integer public wrapper adds one successful compilation and axiom check.
The modules, dependency hashes and captured evidence were checked before
integration. Temporary .olean files, dependency symlinks and drafts are excluded;
each source file is copied only once into its normal repository location.

[numerical-followup.json](numerical-followup.json) preserves all eight preceding
rules and adds three narrow ReLU comparison/literal/conversion followups.
Vector integer execution and program order introduce no numerical assumptions.
The producer's pending mixed-conversion note is retained as descriptive future
work; it does not close missing mixed dtype execution. New followups are not
automatically accepted unified-queue scope bindings.

Final registered-build, sharding-test and global-gate outcomes are recorded in
[validation.json](validation.json). Historical and exact51518a8c comparator
replays exclude all fourteen modules in this batch. Earlier210 benchmark-tool
test results are reused only after verifying unchanged tool/test inputs.

The actual registered Lean build passed7,294 jobs with exit0. Initial catalog
generation detected three unchanged predecessor source hashes absent from the
new root-review input map. Their exact prior hashes were added; no source,
statement, scope or baseline changed. Both catalog attempts are retained in
validation evidence. The successful build did not need repetition for this
metadata-only correction.

The actual 25 audit-sharding tests passed. Global audit completed with eight
passed gates and three failed gates: all183 original-program scopes remain
unaccepted, and the host runner cannot locate the official comparator for
either its availability check or library replay. Source structure, baseline,
catalog reproduction, legacy-source binding, value metadata, deferred-numeric
registration and unchanged inputs all passed. This is recorded as a partial
extension validation, not a successful whole-program acceptance run.
