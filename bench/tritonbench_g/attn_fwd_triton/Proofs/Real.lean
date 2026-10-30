import bench.tritonbench_g.attn_fwd_triton.Sources.Kernels
import bench.tritonbench_g.attn_fwd_triton.Proofs.Mathematical_Real
import Mathlib.Tactic

/-! Reuse the original attention proof dependency graph without modifying it.
Its full execution/frame theorem applies to the original finite-penalty source,
not to this repaired source. The valid online arithmetic lemma below is reused
from the same graph. The independent target below is composed with complete
candidate execution, bounded flat memory and host return in later helpers. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.RealProofs
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealFormula
noncomputable section

abbrev Parameters := VeriTile.Bench.TritonBenchG.AttnFwdTriton.RealProofs.Parameters

structure Inputs where
  query : Nat → Nat → Nat → Nat → ℝ
  key : Nat → Nat → Nat → Nat → ℝ
  value : Nat → Nat → Nat → Nat → ℝ
  queryScale : Nat → ℝ
  keyScale : Nat → ℝ

def score (H N D z h m n : Nat) (xs : Inputs) : ℝ :=
  ((List.range D).map (fun d => xs.query z h m d * xs.key z h n d)).sum *
    xs.queryScale ((z*H+h)*((N+127)/128)+m/128) *
    xs.keyScale ((z*H+h)*((N+63)/64)+n/64)

/-- Independent logical prefix attention. No running state or source execution
occurs here. Scale coordinates preserve the original packed-flat block order. -/
def pairs (H N D z h m d : Nat) (xs : Inputs) : List (ℝ × ℝ) :=
  (List.range (m+1)).map (fun n => (score H N D z h m n xs, xs.value z h n d))

def expected (H N D z h m d : Nat) (xs : Inputs) : ℝ :=
  let ps := pairs H N D z h m d xs
  (ps.map (fun sv => pow2 sv.1 * sv.2)).sum / (ps.map (fun sv => pow2 sv.1)).sum

/-- Pure mathematical reuse, for any partition of the causal input pairs.
This proves no candidate program execution, index bound or memory property. -/
theorem online_formula (H N D z h m d : Nat) (xs : Inputs)
    (blocks : List (List (ℝ × ℝ)))
    (partition : blocks.flatten = pairs H N D z h m d xs) :
    let state := blocks.foldl summarize initial
    state.numerator / state.denominator = expected H N D z h m d xs := by
  dsimp only
  rw [block_updates_eq_closed, initialized_result_normalized, partition]
  simp [unshiftedSum, expected]

theorem query_grid_coverage (N m : Nat) (hm : m < N) :
    m / 128 < (N+127)/128 ∧ m % 128 < 128 ∧
      (m / 128)*128+m%128 = m := by omega

theorem query_grid_unique (m pid lane : Nat) (hl : lane < 128)
    (he : pid*128+lane=m) : pid=m/128 ∧ lane=m%128 := by omega

theorem head_grid_coverage (B H z h : Nat) (hz : z < B) (hh : h < H) :
    z*H+h < B*H := by
  have hz' : z+1 ≤ B := by omega
  calc
    z*H+h < z*H+H := Nat.add_lt_add_left hh _
    _ = (z+1)*H := by ring
    _ ≤ B*H := Nat.mul_le_mul_right H hz'

theorem head_grid_decode (H z h : Nat) (hh : h < H) :
    (z*H+h)/H = z ∧ (z*H+h)%H = h := by
  have hH : 0 < H := by omega
  constructor
  · rw [Nat.add_comm, Nat.add_mul_div_right _ _ hH]
    simp [Nat.div_eq_of_lt hh]
  · rw [Nat.add_comm, Nat.add_mul_mod_self_right]
    exact Nat.mod_eq_of_lt hh

theorem scale_tail_bound (N start : Nat) (hs : start < N) :
    start / 64 < (N+63)/64 := by omega

theorem complete_head_mask (D BD d : Nat) (hD : D ≤ BD) (hd : d < D) :
    d < BD ∧ decide (d < D) = Bool.true := by simp [hd]; omega

/-- Every tensor has its own logical-to-storage map; there is no equality
premise relating Q, K, V or output strides. Storage offsets are carried by the
pointer passed to Triton and therefore omitted from this relative index. -/
def address (strideZ strideH strideN strideD z h n d : Nat) : Nat :=
  z*strideZ+h*strideH+n*strideN+d*strideD

theorem value_tile_address (sz sh sn sd z h start lane d : Nat) :
    z*sz+h*sh+lane*sn+d*sd+start*sn = address sz sh sn sd z h (start+lane) d := by
  simp [address]; ring

/-- Source syntax is shared with FPEquiv. The legacy activeHead field is not
used: repaired masks use the full logical headDim. -/
def source (BM BN BD : Nat) (p : Parameters) : ComputeKernel :=
  TypedSource.repaired p.Q p.K p.V p.QS p.KS p.Out
    p.queryBatch p.queryHead p.queryRow p.queryDim
    p.keyBatch p.keyHead p.keyRow p.keyDim
    p.valueBatch p.valueHead p.valueRow p.valueDim
    p.outputBatch p.outputHead p.outputRow p.outputDim
    p.batches p.heads p.context p.headDim BM BN BD p.stage p.validStage

theorem source_projection (BM BN BD : Nat) (p : Parameters) :
    (source BM BN BD p).toAlgorithm? = .ok (source BM BN BD p).toAlgKernel := by
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.RealProofs
