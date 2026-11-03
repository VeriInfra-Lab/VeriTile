import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealClosedFormula

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Real
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.AttentionFwdTriton3.Host
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 2000000

theorem advance_closed (c : Call) (positive : 0<c.BN) (a : Reference.Inputs R)
    (xs : RealInputs.Inputs c.BM c.D c.NK) (pins : Pins c a xs) (n : Nat)
    (running : (position c a.tile n : Int)<Reference.upper c a)
    (old : Reference.Carry c R) (encoded : Encodes c old (closed c xs a.tile n))
    (pointers : SourceSafety.PointerAt c a old (position c a.tile n)) :
    Encodes c (Reference.advance c ops a (position c a.tile n) old) (closed c xs a.tile (n+1)) := by
  have hk:=key_load c a xs pins _ (position_aligned c positive a n) running (position_started c a n)
  have hv:=value_load c a xs pins _ (position_aligned c positive a n) running (position_started c a n)
  rw [←pointers.1] at hk
  rw [←pointers.2.1] at hv
  have h:=advance_real c a (position c a.tile n) old (closed c xs a.tile n) encoded
    (queryValues c a.tile xs) (keyValues c xs (SourceSafety.physicalKey c a (position c a.tile n)))
    (valueValues c xs (SourceSafety.physicalKey c a (position c a.tile n))) (query_load c a xs pins) hk hv
  simpa only [closed,RealLoopFormula.closed_succ,blockScore,blockActive_eq,blockValue,physical_eq] using h

theorem iterate_closed (c : Call) (positive : 0<c.BN) (a : Reference.Inputs R)
    (xs : RealInputs.Inputs c.BM c.D c.NK) (pins : Pins c a xs) (n : Nat) (within : n≤count c a.tile)
    (old : Reference.Carry c R) (encoded : Encodes c old (closed c xs a.tile n))
    (pointers : SourceSafety.PointerAt c a old (position c a.tile n)) :
    Encodes c (Reference.iterate c ops a (position c a.tile n) (Reference.upper c a) old)
      (finalCarry c xs a.tile) := by
  rw [Reference.iterate,dif_neg (Nat.ne_of_gt positive)]
  by_cases running : (position c a.tile n : Int)<Reference.upper c a
  · rw [dif_pos running]
    have lt : n<count c a.tile := (position_running c positive a.tile n).mp running
    have next:=advance_closed c positive a xs pins n running old encoded pointers
    have ptr:=SourceSafety.advance_pointer c ops a old _ pointers
    rw [←position_succ] at ptr
    have h:=iterate_closed c positive a xs pins (n+1) (by omega)
      (Reference.advance c ops a (position c a.tile n) old) next ptr
    simpa only [position_succ,Nat.cast_add] using h
  · rw [dif_neg running]
    have equal : n=count c a.tile := by
      have h := position_running c positive a.tile n
      rw [upperKey_eq] at h
      omega
    simpa only [equal,finalCarry] using encoded
termination_by count c a.tile-n

theorem result_closed (c : Call) (positive : 0<c.BN) (a : Reference.Inputs R)
    (xs : RealInputs.Inputs c.BM c.D c.NK) (pins : Pins c a xs) :
    Encodes c (Reference.result c ops a) (finalCarry c xs a.tile) := by
  have encoded : Encodes c (Reference.initial c ops a) (closed c xs a.tile 0) := by
    simpa only [closed,RealLoopFormula.closed_zero] using initial_encoded c a xs pins
  have ptr : SourceSafety.PointerAt c a (Reference.initial c ops a) (position c a.tile 0) := by
    simpa only [position,Nat.zero_mul,Nat.add_zero,lowerKey_eq] using SourceSafety.initial_pointer c ops a
  have h:=iterate_closed c positive a xs pins 0 (Nat.zero_le _) _ encoded ptr
  simpa only [Reference.result,position,Nat.zero_mul,Nat.add_zero,lowerKey_eq,
    Int.toNat_of_nonneg (SourceSafety.lower_nonnegative c a)] using h

#axiomsClean result_closed
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Real
