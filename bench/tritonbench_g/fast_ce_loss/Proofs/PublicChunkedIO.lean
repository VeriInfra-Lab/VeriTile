import bench.tritonbench_g.fast_ce_loss.Proofs.IOBinding
import bench.tritonbench_g.fast_ce_loss.Proofs.SourceCells
import bench.tritonbench_g.fast_ce_loss.Proofs.SourceAccessChunked

namespace VeriTile.Bench.Repairs.FastCeLoss.PublicIO
open VeriTile.Triton FP.Structural KernelSource
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 30000
variable {α : Type} [Inhabited α]

theorem chunked_implements (M : Algebra α) (p : Parameters) (row chunk : Nat)
    (wholeRow : Bool) (positive : 0<p.block) (distinct : p.loss≠p.lse) :
    (forwardIO M id id p row chunk Bool.true wholeRow).ImplementsOn
      (nativeDomain p wholeRow) (chunkedFormula M p chunk wholeRow) := by
  intro xs domain bounds bounded initial binding
  let start := (forwardInvocation M p row chunk Bool.true).initial initial
  have ht := forward_label M p row chunk Bool.true wholeRow xs initial binding
  have values := forward_tile M p row chunk Bool.true wholeRow xs initial binding
  have hs : ScheduleSafe bounds [forwardInvocation M p row chunk Bool.true] initial := by
    rw [scheduleSafe_singleton]
    change TraceSafe (engine M) bounds (KernelSource.chunked p) start
    apply SourceAccess.chunked_safe M bounds p start positive
    · exact bounded.2.1 (0 : Fin 3) PUnit.unit trivial
    · intro lane active
      simpa only [start,forward_row,chunked_chunk] using
        forward_tile_bounds M p row chunk Bool.true wholeRow xs bounds bounded lane
          (by simpa only [start,chunked_chunk] using active)
    · intro first active
      have h := bounded.2.1 (2 : Fin 3) PUnit.unit
        ⟨Or.inr first,by simpa only [start,ht] using active⟩
      simpa only [start,ht,forward_row] using h
    · intro first
      exact bounded.2.2.1 (1 : Fin 2) PUnit.unit (Or.inr first)
    · exact bounded.2.2.1 (0 : Fin 2) PUnit.unit trivial
  have valid : label p start = -100 ∨ 0≤label p start := by
    simpa only [start,ht,nativeDomain] using domain
  obtain ⟨final,ran,lse,loss,frame⟩ := SourceCells.chunked M p start positive valid distinct
  have lseFormula := LogicalModel.lse_binding M p start chunk
    (tile p wholeRow chunk (xs (1 : Fin 3))) (by
      intro lane active
      simpa only [start,forward_row] using values lane active)
  refine ⟨hs,final,?_,?_,?_⟩
  · simpa only [forwardIO,runSchedule_singleton,Invocation.run,forwardInvocation,if_true] using ran
  · intro o j active
    fin_cases o <;> cases j
    · change mapCell id (final.mem p.lse (row*p.chunks+chunk))=_
      simp only [start,chunked_chunk] at lse
      simpa only [start,forward_row] using congrArg (mapCell id)
        (lse.trans (congrArg (Cell.mk .real ∘ KernelModel.cast M) lseFormula))
    · have first : chunk=0 := active.resolve_left (by decide)
      have gather := forward_gather M p row chunk Bool.true wholeRow xs initial binding (Or.inr first)
      have lossFormula := LogicalModel.chunk_loss_binding M p start (xs (2 : Fin 3) PUnit.unit)
        (by intro active; simpa only [start,ht,forward_row,Invocation.initial_mem] using
          gather (by simpa only [start,ht] using active))
      change mapCell id (final.mem p.loss row)=_
      simpa only [start,forward_row,ht] using congrArg (mapCell id)
        ((loss first).trans (congrArg (Cell.mk .real ∘ KernelModel.cast M) lossFormula))
  · intro r o outside
    exact frame r o (fun first => outside.1 (1 : Fin 2) PUnit.unit (Or.inr first))
      (outside.1 (0 : Fin 2) PUnit.unit trivial)

#axiomsClean chunked_implements
end
end VeriTile.Bench.Repairs.FastCeLoss.PublicIO
