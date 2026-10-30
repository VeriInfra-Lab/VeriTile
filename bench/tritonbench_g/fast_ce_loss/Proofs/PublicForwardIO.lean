import bench.tritonbench_g.fast_ce_loss.Proofs.IOBinding
import bench.tritonbench_g.fast_ce_loss.Proofs.SourceCells
import bench.tritonbench_g.fast_ce_loss.Proofs.SourceAccessForward

namespace VeriTile.Bench.Repairs.FastCeLoss.PublicIO
open VeriTile.Triton FP.Structural KernelSource
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 30000
variable {α : Type} [Inhabited α]

theorem forward_implements (M : Algebra α) (p : Parameters) (row : Nat)
    (wholeRow : Bool) (positive : 0<p.block) (distinct : p.loss≠p.lse) :
    (forwardIO M id id p row 0 Bool.false wholeRow).ImplementsOn
      (nativeDomain p wholeRow) (forwardFormula M p wholeRow) := by
  intro xs domain bounds bounded initial binding
  let start := (forwardInvocation M p row 0 Bool.false).initial initial
  have ht := forward_label M p row 0 Bool.false wholeRow xs initial binding
  have values := forward_tile M p row 0 Bool.false wholeRow xs initial binding
  have gather := forward_gather M p row 0 Bool.false wholeRow xs initial binding (Or.inl rfl)
  have hs : ScheduleSafe bounds [forwardInvocation M p row 0 Bool.false] initial := by
    rw [scheduleSafe_singleton]
    change TraceSafe (engine M) bounds (KernelSource.forward p) start
    apply SourceAccess.forward_safe M bounds p start positive
    · exact bounded.2.1 (0 : Fin 3) PUnit.unit trivial
    · intro lane active
      simpa only [start,forward_row,Nat.zero_mul,Nat.zero_add] using
        forward_tile_bounds M p row 0 Bool.false wholeRow xs bounds bounded lane (by simpa using active)
    · intro active
      have h := bounded.2.1 (2 : Fin 3) PUnit.unit
        ⟨Or.inl rfl,by simpa only [start,ht] using active⟩
      simpa only [start,ht,forward_row] using h
    · exact bounded.2.2.1 (1 : Fin 2) PUnit.unit (Or.inl rfl)
    · exact bounded.2.2.1 (0 : Fin 2) PUnit.unit trivial
  have valid : label p start = -100 ∨ 0≤label p start := by
    simpa only [start,ht,nativeDomain] using domain
  obtain ⟨final,ran,lse,loss,frame⟩ := SourceCells.forward M p start positive valid distinct
  have lseFormula := LogicalModel.lse_binding M p start 0
    (tile p wholeRow 0 (xs (1 : Fin 3))) (by
      intro lane active
      simpa only [start,forward_row] using values lane active)
  have lossFormula := LogicalModel.forward_loss_binding M p start
    (tile p wholeRow 0 (xs (1 : Fin 3))) (xs (2 : Fin 3) PUnit.unit)
    (by intro lane active; simpa only [start,forward_row,Nat.zero_mul,Nat.zero_add] using
      values lane (by simpa using active))
    (by intro active; simpa only [start,ht,forward_row,Invocation.initial_mem] using
      gather (by simpa only [start,ht] using active))
  refine ⟨hs,final,?_,?_,?_⟩
  · simpa only [forwardIO,runSchedule_singleton,Invocation.run,forwardInvocation,if_false] using ran
  · intro o j active
    fin_cases o <;> cases j
    · change mapCell id (final.mem p.lse row)=_
      simpa only [start,forward_row] using congrArg (mapCell id)
        (lse.trans (congrArg (Cell.mk .real ∘ KernelModel.cast M) lseFormula))
    · change mapCell id (final.mem p.loss row)=_
      simpa only [start,forward_row,ht] using congrArg (mapCell id)
        (loss.trans (congrArg (Cell.mk .real ∘ KernelModel.cast M) lossFormula))
  · intro r o outside
    exact frame r o (outside.1 (1 : Fin 2) PUnit.unit (Or.inl rfl))
      (outside.1 (0 : Fin 2) PUnit.unit trivial)

#axiomsClean forward_implements
end
end VeriTile.Bench.Repairs.FastCeLoss.PublicIO
