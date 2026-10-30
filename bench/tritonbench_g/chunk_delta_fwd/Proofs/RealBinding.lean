import bench.tritonbench_g.chunk_delta_fwd.Proofs.ReferenceTargets
import bench.tritonbench_g.chunk_delta_fwd.Proofs.IOBinding

namespace VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot.RealBinding
open VeriTile Triton FP.Structural ChunkDeltaFwd Interface Reference IOBinding
open VeriTile.Bench.TritonBenchG.ChunkDeltaFwd
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 8000000

theorem correction (p : Parameters) (xs : Inputs ℝ p) (m : Memory ℝ)
    (hb : Reads p id xs m) (head t v : Nat) (hh : head < p.B*p.H)
    (ht : t < p.T) (hv : v < p.V) (a b : Nat → Nat → ℝ)
    (ha : ∀ k, k < p.K → a k v = b k v) :
    RealMath.corrected (Real.arrays p head m) p.K a t v =
      RealMath.corrected (realInputs p xs head) p.K b t v := by
  unfold RealMath.corrected
  congr 1
  · exact hb.value head t v hh ht hv
  · apply Finset.sum_congr rfl
    intro k _
    have hw := hb.weight head t k.val hh ht k.isLt (by omega)
    change (m "weight" _).read .real * a k.val v =
      extend (weight xs) head t k.val * b k.val v
    rw [hw,ha k.val k.isLt]
    rfl

theorem state (p : Parameters) (xs : Inputs ℝ p) (m : Memory ℝ)
    (hb : Reads p id xs m) (head : Nat) (hh : head < p.B*p.H) (c : Nat) :
    ∀ k v, k < p.K → v < p.V → Real.target p head m c k v =
      Real.state (realInputs p xs head) p.T p.K p.BT (innerCount p*p.BC) p.useSeed c k v := by
  induction c with
  | zero =>
    intro k v hk hv
    cases hon : p.useSeed
    · simp [Real.target,Real.state,hon]
    · simpa [Real.target,Real.state,Real.arrays,realInputs,hon] using hb.seed hon head k v hh hk hv
  | succ c ih =>
    intro k v hk hv
    simp only [Real.target,Real.state]
    congr 1
    · exact ih k v hk hv
    · apply Finset.sum_congr rfl
      intro t _
      by_cases ht : c*p.BT+t.val < p.T
      · simp only [if_pos ht]
        have hkey := hb.key head (c*p.BT+t.val) k hh ht hk (by omega)
        have hcor := correction p xs m hb head (c*p.BT+t.val) v hh ht hv
          (Real.target p head m c)
          (Real.state (realInputs p xs head) p.T p.K p.BT (innerCount p*p.BC) p.useSeed c)
          (fun e he => ih e v he hv)
        simp only [Real.target] at hcor
        change (m "key" _).read .real * _ = extend (key xs) head (c*p.BT+t.val) k * _
        rw [hkey,hcor]
        rfl
      · simp only [if_neg ht]

theorem corrected (p : Parameters) (xs : Inputs ℝ p) (m : Memory ℝ)
    (hb : Reads p id xs m) (head t v : Nat) (hh : head < p.B*p.H)
    (ht : t < p.T) (hv : v < p.V) :
    RealMath.corrected (Real.arrays p head m) p.K (Real.target p head m (t/p.BT)) t v =
      RealMath.corrected (realInputs p xs head) p.K
        (Real.state (realInputs p xs head) p.T p.K p.BT (innerCount p*p.BC) p.useSeed (t/p.BT)) t v :=
  correction p xs m hb head t v hh ht hv _ _
    (fun k hk => state p xs m hb head hh (t/p.BT) k v hk hv)

#axiomsClean state
#axiomsClean corrected
end
end VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot.RealBinding
