# Host CUDA reduction and eager-autograd boundary

The repaired source and its fusion are unchanged. The host remains eager Torch:
O/Z/DQ/DK/DV scratch tensors use sum(0), and normalized-output backward reduces
its broadcast cotangent over V. HostReduction.lean does not replace torch.sum
with the unadopted serial-loop proposal.

The installed Reduce.cuh and SharedReduceOps.h hashes are pinned in
evidence/torch-reduction-source.json and host_schedule.py. The selected family is
CUDA FP32 sum_functor<float,float,float> using
func_wrapper<float>(an ordered float-add lambda), vt0=4, input_vec_size=4, with FP32 accumulator
and output, max512 threads and warp32. ROCm, other reduction functors, atomics,
and unsafe compiler reassociation do not fall under this source family.

For each actual host reduction, the ordinary runtime/compiler contract supplies:

1. Ordered TensorIterator subiterators after its can_use_32bit_indexing recursion,
   with coalesced/permuted shapes, input byte strides, input allocation pointer,
   output coordinate, reduction interval start, accumulate flag and CUDA device
   multiprocessor/thread properties. The reduced dimension, when it survives singleton coalescing, is the one source
   shard/V axis; the input offset calculator maps index i to the corresponding
   source element. Output metadata and subiterator visitation are ordinary
   TensorIterator indexing facts. They contain no tensor values or result claim.
2. host_schedule.py recomputes setReduceConfig from these integers, checks any
   captured ReduceConfig for equality, and produces TorchHost.Launch metadata.
   It refuses unknown dimensionality, non-FP32 strides, subiterators that still
   exceed int32 element/byte indexing, invalid launch dimensions, or incomplete
   input coverage. It does not choose a covered tree for an unrecognized run.
3. The CUDA compiler preserves this typed scalar-add dependency graph: zero
   initializes four accumulators, each ops.reduce/ops.combine is one ordered
   scalar addition, func_wrapper_t project/cast is same-FP32 identity, and all movement,
   shuffle, shared/global staging and synchronization obey ordinary instruction
   and allocated-memory semantics. LLVM/compiler correctness for this pinned
   ordinary C++ source boundary is explicit, not a formal Lean parser theorem.
   Each scalar instruction's numerical correspondence is a local obligation.

HostReduction.lean expands the integer routes into a SumPlan with every zero
seed and addition retained. Scalar threads use four independently strided
accumulators, then combine 0+1,+2,+3. Vector-input threads retain the per-row
pointer alignment header, aligned four-element groups and scalar tail. Local
blocks reduce X then Y with decreasing offsets. Global reduction reads staging
CTA slots in increasing per-owner order, then reduces Y then X; whichever CTA
finishes last does not change that expression. TensorIterator split reductions
retain ordered old-output+next-launch accumulation. Selection.checked verifies
leaf permutation to all source indices, and Lean derives the Real sum theorem.
No whole-sum, dot, tensor, execution-success or final-memory numerical premise
occurs in this boundary or checker.

The normalized dZ selector is per(bh,t), since row alignment can change its
expression even for a fixed V. Scratch reduction uniformity is a separate
source/config obligation: Machine.HostScratchBounds proves that a Q/K scratch
reduction large enough to require int32 byte splitting has at most4 shards.
The final uniformity review records the exact split/config argument: reduction
axis splits take priority while there are at least two inputs; singleton
coalescing can remove the reduction dimension entirely. Remaining output splits
then use the same one-input scalar tree.
Empty source calls execute no reduction and need no actual host launch receipt.

The eager normalization DAG retains the actual forward den=Z+eps allocation.
The installed libtorch backward disassembly and dispatcher trace agree on
O/den, previous/den, neg(dY), multiplication, and dY/den, followed by the V sum.
HostNative.singleBackward uses f.denominator; it does not recompute den in FP.
Ordinary Tensor allocation, saved-tensor versioning and contiguous cotangent
materialization preserve their specified logical elements. Finite memory read
footprints, write coverage and frames are established separately in Lean.

The nine synthetic metadata controls and five mutation controls validate the
selector structurally. They are not GPU executions or captures of actual CUDA
launch metadata. Future actual invocations require their matching descriptor
binding. No native numerical conformance has been claimed.
