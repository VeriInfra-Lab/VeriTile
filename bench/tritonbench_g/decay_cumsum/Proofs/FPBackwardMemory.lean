import bench.tritonbench_g.decay_cumsum.Proofs.FPBackwardBase

namespace VeriTile.Bench.Repairs.DecayCumsum.FPBackwardMemory
open VeriTile.Triton FP.Structural FPOps FPPrefix Geometry FPBackwardBase
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 4000000
variable {α : Type} [Inhabited α]

def stores (M : Algebra α) (h : Call) (p : Phase) (prog : Program h p) (t : Nat)
    (dq dk dg : Values α .real [h.BK]) (u : State α) : State α :=
  scatter M h.BK "dg" (off h p prog t) dg
    (scatter M h.BK "dko" (off h p prog t) dk (scatter M h.BK "dqo" (off h p prog t) dq u Bool.false) Bool.false)

theorem other_time (M : Algebra α) (h : Call) (p : Phase) (prog : Program h p)
    (t t' : Nat) (different : t≠t') (r r' : RegionName) (value : Values α .real [h.BK]) (u : State α) (j : Fin h.BK) (explicitCast : Bool := Bool.true) :
    (scatter M h.BK r (off h p prog t) value u explicitCast).mem r' (off h p prog t' j)=u.mem r' (off h p prog t' j) := by
  apply State.scatter_frame
  right
  rintro ⟨k,⟨⟩⟩ _ he
  exact different (FPMemory.off_unique h p prog t t' k j he.symm).1

theorem stores_other_time (M : Algebra α) (h : Call) (p : Phase) (prog : Program h p) (t t' : Nat)
    (different : t≠t') (dq dk dg : Values α .real [h.BK]) (u : State α) (r : RegionName) (j : Fin h.BK) :
    (stores M h p prog t dq dk dg u).mem r (off h p prog t' j)=u.mem r (off h p prog t' j) := by
  unfold stores
  rw [other_time M h p prog t t' different,other_time M h p prog t t' different _ _ _ _ _ Bool.false,
    other_time M h p prog t t' different _ _ _ _ _ Bool.false]

theorem stores_cell (M : Algebra α) (h : Call) (p : Phase) (prog : Program h p) (t : Nat)
    (dq dk dg : Values α .real [h.BK]) (u : State α) (r : RegionName) (hr : r∈outputRegions) (j : Fin h.BK) :
    (stores M h p prog t dq dk dg u).mem r (off h p prog t j)=
      FPReference.stored M ((if r="dqo" then dq else if r="dko" then dk else dg) (j,PUnit.unit)) (r == "dg") := by
  simp only [outputRegions,List.mem_cons,List.not_mem_nil,or_false] at hr
  rcases hr with rfl|rfl|rfl
  · simp only [stores]
    rw [scatter_other M h.BK "dg" _ dg _ "dqo" _ (by decide),
      scatter_other M h.BK "dko" _ dk _ "dqo" _ (by decide) Bool.false,FPMemory.cell (explicitCast := Bool.false)]
    rfl
  · simp only [stores]
    rw [scatter_other M h.BK "dg" _ dg _ "dko" _ (by decide),FPMemory.cell (explicitCast := Bool.false)]
    rfl
  · exact FPMemory.cell M h p prog t "dg" dg _ j

theorem stores_frame (M : Algebra α) (h : Call) (p : Phase) (prog : Program h p) (t : Fin p.span)
    (dq dk dg : Values α .real [h.BK]) (u : State α) :
    FPMemory.Frame h p prog outputRegions u (stores M h p prog t.val dq dk dg u) := by
  intro r o outside
  let uq := scatter M h.BK "dqo" (off h p prog t.val) dq u Bool.false
  let uk := scatter M h.BK "dko" (off h p prog t.val) dk uq Bool.false
  exact (FPMemory.frame M h p prog t "dg" outputRegions (by simp [outputRegions]) dg uk Bool.true r o outside).trans
    ((FPMemory.frame M h p prog t "dko" outputRegions (by simp [outputRegions]) dk uq Bool.false r o outside).trans
      (FPMemory.frame M h p prog t "dqo" outputRegions (by simp [outputRegions]) dq u Bool.false r o outside))

#axiomsClean stores_cell
#axiomsClean stores_frame
end
end VeriTile.Bench.Repairs.DecayCumsum.FPBackwardMemory
