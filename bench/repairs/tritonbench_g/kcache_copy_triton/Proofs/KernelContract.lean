import bench.repairs.tritonbench_g.kcache_copy_triton.Proofs.Memory
import VeriTile.Triton.Semantics.State
namespace VeriTile.Bench.Repairs.KcacheCopyTriton
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

theorem logicalState_binding (m : Inputs) (n : Nat) (b : Bool) (l : Lengths) (t : Tables)
    (metadata : Memory Word) (payload : Nat → ℝ) (scratch : BlockState)
    (binding : TensorBinding m n b l t id metadata) :
    KernelMetadataBinding m n b l t (logicalState m metadata payload scratch) := by
  constructor
  · intro q hq
    simpa [logicalState,BlockState.readMemValue,BlockState.readMemTyped] using congrArg BitVec.toInt (binding.lengths q hq)
  · intro q c hq hc
    simpa [logicalState,BlockState.readMemValue,BlockState.readMemTyped] using congrArg BitVec.toInt (binding.table q c hq hc)

theorem logicalState_payload (m : Inputs) (metadata : Memory Word) (payload : Nat → ℝ)
    (scratch : BlockState) (i : Nat) :
    (logicalState m metadata payload scratch).readMem "key" i=payload (address m .key i) ∧
    (logicalState m metadata payload scratch).readMem "cache" i=payload (address m .cache i) := by
  simp [logicalState,BlockState.readMem,BlockState.readMemValue,BlockState.readMemTyped]

/-- The signed source shift is exactly the independent last-n-token position. -/
theorem position_arithmetic (m : Inputs) (n : Nat) (b : Bool) (l : Lengths) (t : Tables)
    (h : InputContract m n b l t) (s : Nat) (hs : s<(config m n b).count) :
    (l (s/n)).toInt+((s:Int)-(n:Int)*(((s/n:Nat):Int)+1)) =
      (past (config m n b) l s:Int) := by
  have bounds := h.bounds.1 _ (sequence_bound m n b l t h s hs)
  change (n:Int)≤(l (s/n)).toInt ∧ _ at bounds
  have nonnegative : 0≤(l (s/n)).toInt := by omega
  have enough : n≤(l (s/n)).toInt.toNat := by omega
  have partition : s/n*n+s%n=s := by simpa only [Nat.mul_comm] using Nat.div_add_mod s n
  have partitionInt : (s:Int)=((s/n:Nat):Int)*(n:Int)+((s%n:Nat):Int) := by
    exact_mod_cast partition.symm
  change _=(((l (s/n)).toInt.toNat-n+s%n:Nat):Int)
  rw [Int.natCast_add,Int.ofNat_sub enough,Int.toNat_of_nonneg nonnegative,partitionInt]
  ring

theorem output_expanded (m : Inputs) (n : Nat) (b : Bool) (l : Lengths) (t : Tables)
    (h : InputContract m n b l t) (e : Lane) (he : ValidLane (config m n b) e) :
    blockId (config m n b) l t e.tile.sequence * ((config m n b).heads*(config m n b).blockSize*(config m n b).dims) +
      e.tile.head*((config m n b).blockSize*(config m n b).dims) +
      e.tile.split*(if (config m n b).newLayout then (config m n b).blockSize*(config m n b).width else 0) +
      slot (config m n b) l e.tile.sequence*(config m n b).width + e.lane =
        outputIndex (config m n b) l t e.tile.sequence e.tile.head (component (config m n b) e) := by
  have geometry := split_geometry m n b l t h
  have div : component (config m n b) e/(config m n b).width=e.tile.split := by
    simpa [component,Nat.add_comm,Nat.div_eq_of_lt he.2] using
      Nat.add_mul_div_right e.lane e.tile.split geometry.1
  have mod : component (config m n b) e%(config m n b).width=e.lane := by
    simp [component,Nat.add_mod,Nat.mod_eq_of_lt he.2]
  unfold outputIndex
  cases b with
  | «false» =>
    have split : e.tile.split=0 := by have hv := he.1.2.2;change e.tile.split<1 at hv;omega
    simp only [config,ite_false,Bool.false_eq_true,KvCacheCopy.cacheIndex] at *
    simp only [component,split,Nat.zero_mul,Nat.zero_add]
    ring
  | «true» =>
    have new : (config m n Bool.true).newLayout=Bool.true := rfl
    simp only [new,ite_true,KvCacheCopy.cacheIndex,div,mod]
    rw [←geometry.2]
    ring

#axiomsClean logicalState_binding
#axiomsClean logicalState_payload
#axiomsClean position_arithmetic
#axiomsClean output_expanded
end
end VeriTile.Bench.Repairs.KcacheCopyTriton
