import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.Preparation

/-! Exact logical input coordinates and removal of zero-padded feature lanes.
The input functions read the actual independent role strides. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.LogicalInputs
open VeriTile.Triton Memory
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 800000
set_option linter.unusedSimpArgs false

def inputs (mem : BlockState) (p : Parameters) (offsets : BaseOffsets := {}) : RealProofs.Inputs where
  query := fun z h n d => mem.readMem p.Q (RealProofs.address p.queryBatch p.queryHead p.queryRow p.queryDim z h n d+offsets.query)
  key := fun z h n d => mem.readMem p.K (RealProofs.address p.keyBatch p.keyHead p.keyRow p.keyDim z h n d+offsets.key)
  value := fun z h n d => mem.readMem p.V (RealProofs.address p.valueBatch p.valueHead p.valueRow p.valueDim z h n d+offsets.value)
  queryScale := fun n => mem.readMem p.QS (n+offsets.queryScale)
  keyScale := fun n => mem.readMem p.KS (n+offsets.keyScale)

theorem padded_sum (D BD : Nat) (hD : D ≤ BD) (f : Nat → ℝ) :
    (∑ d : Fin BD, if d.val < D then f d.val else 0) = ((List.range D).map f).sum := by
  rw [Fin.sum_univ_eq_sum_range (fun d => if d < D then f d else 0)]
  have hs : (∑ d ∈ Finset.range BD, if d < D then f d else 0) =
      ∑ d ∈ Finset.range D, if d < D then f d else 0 := by
    symm
    apply Finset.sum_subset (Finset.range_mono hD)
    intro d hd hn
    simp only [Finset.mem_range] at hn
    simp [hn]
  rw [hs]
  simp only [Finset.sum_ite_of_true (fun d hd => Finset.mem_range.mp hd)]
  clear hs hD
  induction D with
  | zero => simp
  | succ D ih => simpa [Finset.sum_range_succ,List.range_succ] using congrArg (fun x : ℝ => x + f D) ih

theorem score_at (mem : BlockState) (c : Config) (BD start : Nat) (row : Fin 128) (key : Fin 64)
    (hD : c.p.headDim ≤ BD) (hr : c.startM*128+row.val < c.p.context)
    (hk : start+key.val < c.p.context) (ha : start % 64 = 0) :
    Preparation.score mem c 128 64 BD start row key =
      RealProofs.score c.p.heads c.p.context c.p.headDim c.z c.h (c.startM*128+row.val)
        (start+key.val) (inputs mem c.p c.offsets) := by
  have hq : (c.startM*128+row.val)/128 = c.startM := by omega
  have hks : (start+key.val)/64 = start/64 := by omega
  have hsum : (∑ d : Fin BD, qvalue mem c 128 BD (row,d,PUnit.unit) *
      kvalue mem c BD 64 start (d,key,PUnit.unit)) =
      ((List.range c.p.headDim).map (fun d =>
        (inputs mem c.p c.offsets).query c.z c.h (c.startM*128+row.val) d *
        (inputs mem c.p c.offsets).key c.z c.h (start+key.val) d)).sum := by
    rw [← padded_sum c.p.headDim BD hD]
    apply Finset.sum_congr rfl
    intro d _
    by_cases hd : d.val < c.p.headDim
    · simp only [qvalue,kvalue,hr,hk,hd,and_self,ite_true]
      simp [inputs,queryPointer,keyPointer,qbase,kbase,RealProofs.address,
        Nat.add_comm,Nat.add_left_comm,Nat.add_assoc]
    · simp [qvalue,kvalue,hr,hk,hd]
  simp only [Preparation.score,VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealNumeric.scores,
    hsum,RealProofs.score,inputs,hq,hks,qsbase,ksbase]
  simp [Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]

theorem value_at (mem : BlockState) (c : Config) (BD start : Nat) (key : Fin 64) (lane : Fin BD)
    (hk : start+key.val < c.p.context) (hd : lane.val < c.p.headDim) :
    vvalue mem c 64 BD start (key,lane,PUnit.unit) =
      (inputs mem c.p c.offsets).value c.z c.h (start+key.val) lane.val := by
  simp [vvalue,hk,hd,inputs,valuePointer,vbase,RealProofs.address,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.LogicalInputs
