import bench.tritonbench_g.context_attn_nopad.Proofs.IOLogicalReads

namespace VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.IOLogical
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnNopad Host Program
open IOData
noncomputable section
set_option maxHeartbeats 4000000
variable {α : Type}

theorem iterate_congr (c : Call) (r : Regions) (o : FPBackend.Ops α) (idx : GridIndex c.grid)
    (left right : Memory α) (same : SameReads c r left right) (n stop : Nat)
    (hs : stop ≤ ((c.launch idx).block+1)*128) (old : FPReference.Acc c.layout α) :
    FPReference.iterate c.layout o c.scale (arguments c r idx left) (n*128 : Nat) stop old=
      FPReference.iterate c.layout o c.scale (arguments c r idx right) (n*128 : Nat) stop old := by
  conv_lhs => rw [FPReference.iterate]
  conv_rhs => rw [FPReference.iterate]
  simp only [show c.layout.BN=128 from rfl,dif_neg (by decide : (128:Nat)≠0),Nat.cast_ofNat]
  split_ifs with active
  · have hn : n*128<stop := by exact_mod_cast active
    rw [advance_congr c r o idx left right same n (lt_of_lt_of_le hn hs)]
    have add : ((n*128 : Nat) : Int)+128=((n+1)*128 : Nat) := by push_cast; ring
    rw [add]
    exact iterate_congr c r o idx left right same (n+1) stop hs _
  · rfl
termination_by stop-n*128
decreasing_by omega

theorem output_congr (c : Call) (r : Regions) (o : FPBackend.Ops α) (idx : GridIndex c.grid)
    (left right : Memory α) (same : SameReads c r left right) :
    output c r o idx left=output c r o idx right := by
  unfold output FPReference.output
  have hm : FPReference.blockMask c.layout (arguments c r idx left) ≤ 1 := by
    unfold FPReference.blockMask
    split <;> omega
  have hs : FPReference.blockMask c.layout (arguments c r idx left)*((c.launch idx).block+1)*128 ≤
      ((c.launch idx).block+1)*128 :=
    Nat.mul_le_mul_right 128 (by simpa using Nat.mul_le_mul_right ((c.launch idx).block+1) hm)
  have h := iterate_congr c r o idx left right same 0 _ hs (FPReference.initial c.layout o.algebra)
  simpa only [Nat.zero_mul,Nat.cast_zero] using congrArg
    (fun acc : FPReference.Acc c.layout α => fun i => o.algebra.cast (some .fp32) .real .real (acc.weighted i)) h

/-- Scalar input projection only. Metadata comes from its existing signed
observation and no equality of unused memory is required. -/
theorem source_arguments [Inhabited α] (c : Call) (r : Regions) (idx : GridIndex c.grid)
    (s : State α) (metadata : FPHostSource.Metadata c r s) :
    FPReference.inputs r (FPHost.atProgram c idx s)=
      arguments c r idx (fun region offset => (s.mem region offset).read .real) := by
  have hs := metadata.start (c.launch idx).batch (launch_batch_lt c idx)
  have hl := metadata.length (c.launch idx).batch (launch_batch_lt c idx)
  unfold FPReference.inputs FPHost.atProgram arguments
  congr 1

#axiomsClean output_congr
#axiomsClean source_arguments
end
end VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.IOLogical
