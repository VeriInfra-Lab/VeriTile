import bench.tritonbench_g.chunk_delta_fwd.Sources.RealModel
namespace VeriTile.Bench.Repairs.ChunkDeltaFwd.Real
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
open VeriTile.Bench.TritonBenchG.ChunkDeltaFwd
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 6000000

def contribution (p : Parameters) (pid : PID) (m : Memory ℝ) (c b : Nat)
    (frozen : HValues p ℝ) (i : HIndex p) :=
  ∑ lane : Fin p.BC, keyLoad algebra p pid m c b ⟨i.1,lane,()⟩ *
    corrected algebra dot p pid m c b frozen ⟨lane,i.2.1,()⟩

theorem accumulated_prefix (p : Parameters) (pid : PID) (m : Memory ℝ)
    (c n : Nat) (h : HValues p ℝ) (i : HIndex p) :
    foldRange (fun b acc => accumulatedStep algebra dot p pid m c b h acc)
      0 n 1 (fun _ => zero algebra) i = ∑ b ∈ Finset.range n, contribution p pid m c b h i := by
  induction n with
  | zero => simp [foldRange_zero,zero,algebra]
  | succ n ih =>
    rw [foldRange_succ]
    change (foldRange (fun b acc => accumulatedStep algebra dot p pid m c b h acc)
      0 n 1 (fun _ => zero algebra) i)+contribution p pid m c n h i=_
    rw [ih,Finset.sum_range_succ]

theorem accumulated_eq (p : Parameters) (pid : PID) (m : Memory ℝ)
    (c : Nat) (h : HValues p ℝ) (s : Nat → Nat → ℝ)
    (hK : p.K ≤ p.BK) (hh : Represents p pid h s) (i : HIndex p) (hi : hActive p pid i) :
    accumulated algebra dot p pid m c h i =
      ∑ t : Fin (innerCount p*p.BC), if c*p.BT+t.val < p.T then
        (arrays p pid.2 m).key (c*p.BT+t.val) i.1.val *
          RealMath.corrected (arrays p pid.2 m) p.K s (c*p.BT+t.val) (column p pid i.2.1.val) else 0 := by
  unfold accumulated
  rw [accumulated_prefix,← Fin.sum_univ_eq_sum_range]
  calc
    _ = ∑ b : Fin (innerCount p), ∑ lane : Fin p.BC,
        if c*p.BT+(b.val*p.BC+lane.val) < p.T then
          (arrays p pid.2 m).key (c*p.BT+(b.val*p.BC+lane.val)) i.1.val *
            RealMath.corrected (arrays p pid.2 m) p.K s
              (c*p.BT+(b.val*p.BC+lane.val)) (column p pid i.2.1.val) else 0 := by
      apply Finset.sum_congr rfl
      intro b _
      unfold contribution
      apply Finset.sum_congr rfl
      intro lane _
      by_cases ht : time p c b.val lane.val < p.T
      · have hc := corrected_eq p pid m c b.val h s hK hh ⟨lane,i.2.1,()⟩ ⟨ht,hi.2⟩
        have ht' : c*p.BT+(b.val*p.BC+lane.val) < p.T := by simpa [time,Nat.add_assoc] using ht
        rw [hc]
        simp [keyLoad,hi.1,ht,ht',algebra,arrays,time,Nat.add_assoc]
      · have ht' : ¬c*p.BT+(b.val*p.BC+lane.val) < p.T := by simpa [time,Nat.add_assoc] using ht
        simp [keyLoad,ht,ht',zero,algebra]
    _ = _ := RealMath.block_sum (fun t => if c*p.BT+t < p.T then
      (arrays p pid.2 m).key (c*p.BT+t) i.1.val *
        RealMath.corrected (arrays p pid.2 m) p.K s (c*p.BT+t) (column p pid i.2.1.val) else 0)
      (innerCount p) p.BC

theorem initial_represents (p : Parameters) (pid : PID) (m : Memory ℝ) :
    Represents p pid (initial algebra p pid m) (target p pid.2 m 0) := by
  intro i hi
  simp [initial,target,state,hi.1,hi.2,cast,zero,algebra,arrays]

theorem states_represents (p : Parameters) (pid : PID) (m : Memory ℝ) (hK : p.K ≤ p.BK) (c : Nat) :
    Represents p pid (states algebra dot p pid m c) (target p pid.2 m c) := by
  induction c with
  | zero => exact initial_represents p pid m
  | succ c ih =>
    intro i hi
    change states algebra dot p pid m c i + accumulated algebra dot p pid m c
      (states algebra dot p pid m c) i = _
    rw [ih i hi,accumulated_eq p pid m c _ _ hK ih i hi]
    rfl

theorem corrected_state (p : Parameters) (pid : PID) (m : Memory ℝ) (hK : p.K ≤ p.BK)
    (c b : Nat) (i : CIndex p) (hi : cActive p pid c b i) :
    corrected algebra dot p pid m c b (states algebra dot p pid m c) i =
      RealMath.corrected (arrays p pid.2 m) p.K (target p pid.2 m c)
        (time p c b i.1.val) (column p pid i.2.1.val) :=
  corrected_eq p pid m c b _ _ hK (states_represents p pid m hK c) i hi

#axiomsClean accumulated_eq
#axiomsClean states_represents
#axiomsClean corrected_state
end
end VeriTile.Bench.Repairs.ChunkDeltaFwd.Real
