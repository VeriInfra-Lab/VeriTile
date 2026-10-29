import bench.tritonbench_g.kcache_copy_triton.Proofs.SignedRawBitCopy
namespace VeriTile.Bench.TritonBenchG.KcacheCopyTriton.SignedRawBits
open VeriTile.Triton FP.Structural
open TypedCopy (Storage)
open SignedCopy (Parameters)
open RawBits
noncomputable section

/-- Complete arbitrary-n_tokens signed/split-x source, for all five width-explicit raw
storage formats. The source's parameters, signed metadata, aliases, zero width,
zero strides, arbitrary register seed and undef function are unrestricted.
The only function parameters are memory/launch state, not numerical callbacks.
This is word-addressed software storage, not native/GPU memory conformance. -/
specification all_tokens_five_format_raw_bit_correctness (f : MachineFormat) (p : Parameters)
    (m : Memory (BitVec 32)) (seed : State (Carrier (BitVec 32))) :
    ∃ final, FP.Structural.exec (TypedCopy.copyBackend inert) (SignedCopy.source f.storage p)
      (state m seed) = some final ∧
      final.mem=encodeMemory (reference f.storage p m seed.pids) ∧
      decodeMemory final.mem=reference f.storage p m seed.pids ∧
      (∀ i : TileIndex [p.width], final.mem p.KCache (outputOffset p m seed.pids i) =
        encode (readStorage f.storage (m p.K (inputOffset p seed.pids i)))) ∧
      (∀ r o, (r≠p.KCache ∨ ∀ i : TileIndex [p.width], o≠outputOffset p m seed.pids i) →
        final.mem r o=encode (m r o)) ∧
      (∀ r o, Canonical (final.mem r o)) :=
  complete f.storage p m seed

/-- Every payload, including signaling NaN bit patterns and both zero signs,
survives the complete source. This corollary states initial input contents;
it does not assume any final result. -/
theorem every_machine_payload (f : MachineFormat) (p : Parameters)
    (m : Memory (BitVec 32)) (seed : State (Carrier (BitVec 32)))
    (i : TileIndex [p.width]) (b : BitVec f.width)
    (hinput : m p.K (inputOffset p seed.pids i)=payload f b) :
    (SignedCopy.execution inert f.storage p (state m seed)).val.mem p.KCache
      (outputOffset p m seed.pids i) = encode (payload f b) := by
  simpa only [hinput,payload_storage] using execution_readback inert f.storage p m seed i

/-- The old contiguous-head cache layout is the original grid's single
split-x program with zero split stride. This equation instantiates the full
source reference without dropping the sequence metadata producer. -/
theorem legacy_layout_input (p : Parameters) (ids : Nat → Nat)
    (hgrid : ids 2 = 0) (i : TileIndex [p.width]) :
    inputOffset p ids i = ids 0*p.strideKt + ids 1*p.strideKh + i.1.val*p.strideKd := by
  simp [inputOffset, hgrid]

theorem legacy_layout_output {R : Type} (p : Parameters) (m : Memory R)
    (ids : Nat → Nat) (hgrid : ids 2 = 0) (i : TileIndex [p.width]) :
    outputOffset p m ids i = SignedCopy.unsigned64 (blockId p m ids)*p.strideKcb +
      ids 1*p.strideKch +
      SignedCopy.unsigned64 ((past p m ids).tmod (p.blockSize : Int))*p.strideKcs + i.1.val := by
  simp [outputOffset, hgrid]

/-- Unlike Nat subtraction, this preserves the negative token shift needed
by prefill n>1. It also explains how all n positions address their last n slots. -/
theorem token_shift_eq_mod_sub (p : Parameters) (ids : Nat → Nat) :
    SignedCopy.tokenShift p ids = ((ids 0 % p.nTokens : Nat) : Int) - p.nTokens := by
  have h := Nat.mod_add_div (ids 0) p.nTokens
  have hi := congrArg (fun x : Nat => (x : Int)) h
  push_cast at hi
  dsimp [SignedCopy.tokenShift, SignedCopy.sequence]
  simp only [mul_add, mul_one]
  omega

#axiomsClean all_tokens_five_format_raw_bit_correctness
#axiomsClean every_machine_payload
#axiomsClean legacy_layout_input
#axiomsClean legacy_layout_output
#axiomsClean token_shift_eq_mod_sub
end
end VeriTile.Bench.TritonBenchG.KcacheCopyTriton.SignedRawBits
