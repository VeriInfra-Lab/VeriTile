import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.RuntimeLoops
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.RuntimeTransport
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
variable {α : Type} [Inhabited α]

mutual
 def avoids : ComputeStmt → Bool
  | .assign _ _ n _ => n != "scale"
  | .store .. => Bool.true
  | .forRange n _ _ _ body | .forRangeDyn n _ _ _ body => (n != "scale") && avoidsList body
  | .ifThen _ body => avoidsList body
  | .effectMarker n => n == "tl.debug_barrier"
  | _ => Bool.false
 termination_by st => (sizeOf st,0)
 decreasing_by all_goals simp_all; omega
 def avoidsList : List ComputeStmt → Bool
  | [] => Bool.true
  | a::as => avoids a && avoidsList as
 termination_by code => (sizeOf code,0)
 decreasing_by all_goals simp_all; omega
end

mutual
 theorem sameAtom (M : Algebra α) (value : α) (st : ComputeStmt) (h : avoids st=Bool.true) : Atom M value st st := by
  cases hst : st with
  | assign d sh n e => exact sameAssignment M value n (by simpa [hst,avoids] using h) e
  | store d sh mem e mask => exact sameStore M value mem e mask
  | forRange n start stop stride body =>
    have hn : n ≠ "scale" ∧ avoidsList body=Bool.true := by simpa [hst,avoids] using h
    exact rangeAtom M value n hn.1 start stop stride body body (samePair M value body hn.2)
  | forRangeDyn n start stop stride body =>
    have hn : n ≠ "scale" ∧ avoidsList body=Bool.true := by simpa [hst,avoids] using h
    exact dynamicAtom M value n hn.1 start stop stride body body (samePair M value body hn.2)
  | ifThen cond body => exact ifAtom M value cond body body (samePair M value body (by simpa [hst,avoids] using h))
  | effectMarker n =>
    have hn : n="tl.debug_barrier" := by simpa [hst,avoids] using h
    subst n
    exact barrier M value
  | _ => simp [hst,avoids] at h
 termination_by (sizeOf st,0)
 decreasing_by all_goals simp_all; omega
 theorem samePair (M : Algebra α) (value : α) (code : List ComputeStmt) (h : avoidsList code=Bool.true) :
    Pair M value code code := by
  cases hc : code with
  | nil => exact RuntimeTransport.nil M value
  | cons a as =>
    have hn : avoids a=Bool.true ∧ avoidsList as=Bool.true := by simpa [hc,avoidsList] using h
    exact cons (sameAtom M value a hn.1) (samePair M value as hn.2)
 termination_by (sizeOf code,0)
 decreasing_by all_goals simp_all; omega
end
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.RuntimeTransport
