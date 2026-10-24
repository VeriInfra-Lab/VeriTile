import bench.repairs.tritonbench_g.kv_cache_copy.Proofs.Memory

namespace VeriTile.Bench.Repairs.KvCacheCopy
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

/-- The source's shifted arange has exactly the translated zero-based lanes. -/
theorem source_range (q width : Nat) :
    List.range' (q*width) ((q+1)*width-q*width)=
      (List.range width).map (fun i => q*width+i) := by
  simp only [Nat.add_mul,Nat.one_mul,Nat.add_sub_cancel_left,List.range'_eq_map_range]

theorem logicalState_binding (m : Inputs) (b : Bool) (l : Lengths) (t : Tables)
    (metadata : Memory Word) (payload : Nat → ℝ) (scratch : BlockState)
    (binding : TensorBinding m b l t id metadata) :
    KernelMetadataBinding m b l t (logicalState m metadata payload scratch) := by
  constructor
  · intro q hq
    simpa [logicalState,BlockState.readMemValue,BlockState.readMemTyped] using congrArg BitVec.toInt (binding.lengths q hq)
  · intro q c hq hc
    simpa [logicalState,BlockState.readMemValue,BlockState.readMemTyped] using congrArg BitVec.toInt (binding.table q c hq hc)

theorem logicalState_payload (m : Inputs) (metadata : Memory Word) (payload : Nat → ℝ)
    (scratch : BlockState) (key : Bool) (i : Nat) :
    (logicalState m metadata payload scratch).readMem (inputRegion key) i=payload (address m (inputRole key) i) ∧
    (logicalState m metadata payload scratch).readMem (outputRegion key) i=payload (address m (outputRole key) i) := by
  cases key <;> simp [logicalState,inputRegion,outputRegion,inputRole,outputRole,
    BlockState.readMem,BlockState.readMemValue,BlockState.readMemTyped]

theorem output_expanded (m : Inputs) (b : Bool) (l : Lengths) (t : Tables)
    (h : InputContract m b l t) (e : Lane) (he : ValidLane (config m b) e) :
    (if e.key then
      blockId (config m b) l t e.tile.sequence*((config m b).heads*(config m b).blockSize*(config m b).dims)+
      e.tile.head*((config m b).blockSize*(config m b).dims)+
      e.tile.split*(if (config m b).newLayout then (config m b).blockSize*(config m b).width else 0)+
      slot (config m b) l e.tile.sequence*(config m b).width+e.lane
    else
      blockId (config m b) l t e.tile.sequence*((config m b).heads*(config m b).blockSize*(config m b).dims)+
      e.tile.head*((config m b).blockSize*(config m b).dims)+
      slot (config m b) l e.tile.sequence*(config m b).dims+component (config m b) e)=
      outputIndex (config m b) l t e.key e.tile.sequence e.tile.head (component (config m b) e) := by
  have geometry := split_geometry m b l t h
  have div : component (config m b) e/(config m b).width=e.tile.split := by
    simpa [component,Nat.add_comm,Nat.div_eq_of_lt he.2] using
      Nat.add_mul_div_right e.lane e.tile.split geometry.1
  have mod : component (config m b) e%(config m b).width=e.lane := by
    simp [component,Nat.add_mod,Nat.mod_eq_of_lt he.2]
  cases key : e.key with
  | «false» =>
    simp only [Bool.false_eq_true,ite_false,outputIndex,Bool.false_and,cacheIndex]
    ring
  | «true» =>
    simp only [ite_true,outputIndex,Bool.true_and]
    cases b with
    | «false» =>
      have split : e.tile.split=0 := by have hv := he.1.2.2;change e.tile.split<1 at hv;omega
      simp only [config,ite_false,Bool.false_eq_true,cacheIndex] at *
      simp only [component,split,Nat.zero_mul,Nat.zero_add]
      ring
    | «true» =>
      have new : (config m Bool.true).newLayout=Bool.true := rfl
      simp only [new,ite_true,cacheIndex,div,mod]
      rw [←geometry.2]
      ring

#axiomsClean source_range
#axiomsClean logicalState_binding
#axiomsClean logicalState_payload
#axiomsClean output_expanded
end
end VeriTile.Bench.Repairs.KvCacheCopy
