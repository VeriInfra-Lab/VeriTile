import bench.repairs.tritonbench_g.layer_norm_triton.IO
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceBackwardSafety
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceFinalResult

namespace VeriTile.Bench.Repairs.LayerNormTriton.FinalIO
open VeriTile.Triton FP.Structural SourceModel Interface
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
variable {α : Type} [Inhabited α]

/-- Ordered local tile accumulation over logical partial-gradient arrays.
No execution state, memory update or source evaluator occurs in this reference. -/
def fold (m : Model α) (M N BM BN pid cur : Nat)
    (dw db : Nat → Fin BN → α) (a : SourceFinalGraph.Carries α BM BN) :
    SourceFinalGraph.Carries α BM BN :=
  if BM=0 then a else if cur<M then
    fold m M N BM BN pid (cur+BM) dw db
      (fun i => SourceForwardGraph.bin m .add (a.1 i)
        (if cur+i.1.val<M ∧ pid*BN+i.2.1.val<N then dw (cur+i.1.val) i.2.1 else SourceForwardGraph.lit m 0),
       fun i => SourceForwardGraph.bin m .add (a.2 i)
        (if cur+i.1.val<M ∧ pid*BN+i.2.1.val<N then db (cur+i.1.val) i.2.1 else SourceForwardGraph.lit m 0))
  else a
termination_by M-cur
decreasing_by omega

def expected (m : Model α) (M N BM BN pid : Nat)
    (observe : RegionName → Nat → α → α) (vals : Final.Inputs α M BN) : Final.Outputs α BN :=
  let sums := fold m M N BM BN pid 0 (Final.value vals 0) (Final.value vals 1)
    (SourceFinalGraph.zero m BM BN)
  Final.result
    (fun j => observe "FINAL_DW" (pid*BN+j.val)
      (SourceForwardGraph.cast m (SourceFinalGraph.reduced m BM BN sums.1 (j,()))))
    (fun j => observe "FINAL_DB" (pid*BN+j.val)
      (SourceForwardGraph.cast m (SourceFinalGraph.reduced m BM BN sums.2 (j,()))))

theorem fold_eq (m : Model α) (s : State α) (M N BM BN cur : Nat)
    (dw db : Nat → Fin BN → α) (a : SourceFinalGraph.Carries α BM BN)
    (hw : ∀ r,r<M → ∀ j : Fin BN,s.pids 0*BN+j.val<N →
      (s.mem "DW" (r*N+(s.pids 0*BN+j.val))).read .real=dw r j)
    (hb : ∀ r,r<M → ∀ j : Fin BN,s.pids 0*BN+j.val<N →
      (s.mem "DB" (r*N+(s.pids 0*BN+j.val))).read .real=db r j) :
    SourceFinalGraph.fold m s M N BM BN cur a=fold m M N BM BN (s.pids 0) cur dw db a := by
  rw [SourceFinalGraph.fold,fold]
  split
  · rfl
  · split
    · rw [fold_eq m s M N BM BN (cur+BM) dw db _ hw hb]
      congr 1
      apply Prod.ext <;> funext i
      · simp only [SourceFinalGraph.next,SourceFinalGraph.loadTile]
        split
        · rename_i h
          rw [hw _ h.1 _ h.2]
        · rfl
      · simp only [SourceFinalGraph.next,SourceFinalGraph.loadTile]
        split
        · rename_i h
          rw [hb _ h.1 _ h.2]
        · rfl
    · rfl
termination_by M-cur
decreasing_by omega

theorem implements (m : Model α) (M N BM BN pid : Nat)
    (observe : RegionName → Nat → α → α) :
    (Final.io m M N BM BN pid observe).Implements (expected m M N BM BN pid observe) := by
  apply ProgramIO.Implements.intro
  · intro vals bounds boundsOK initial _
    change ScheduleSafe bounds [job m (TypedSource.finalBackward "DW" "DB" "FINAL_DW" "FINAL_DB" M N BM BN) pid] initial
    rw [scheduleSafe_singleton]
    apply SourceSafety.Final.source_safe m
    · intro row hr j hj
      exact boundsOK.2.1 (0 : Fin 2) (⟨row,hr⟩,j,()) hj
    · intro row hr j hj
      exact boundsOK.2.1 (1 : Fin 2) (⟨row,hr⟩,j,()) hj
    · intro j hj
      exact boundsOK.2.2.1 (0 : Fin 2) (j,()) hj
    · intro j hj
      exact boundsOK.2.2.1 (1 : Fin 2) (j,()) hj
  · intro vals initial binding
    let start := (job m (TypedSource.finalBackward "DW" "DB" "FINAL_DW" "FINAL_DB" M N BM BN) pid).initial initial
    have hw : ∀ row,row<M → ∀ j : Fin BN,pid*BN+j.val<N →
        (start.mem "DW" (row*N+(pid*BN+j.val))).read .real=Final.value vals 0 row j := by
      intro row hr j hj
      simpa [start,job,Invocation.initial,bindRegisters,Final.io,Final.inputs,Final.value,hr,mapValue]
        using binding (0 : Fin 2) (⟨row,hr⟩,j,()) hj
    have hb : ∀ row,row<M → ∀ j : Fin BN,pid*BN+j.val<N →
        (start.mem "DB" (row*N+(pid*BN+j.val))).read .real=Final.value vals 1 row j := by
      intro row hr j hj
      simpa [start,job,Invocation.initial,bindRegisters,Final.io,Final.inputs,Final.value,hr,mapValue]
        using binding (1 : Fin 2) (⟨row,hr⟩,j,()) hj
    obtain ⟨t,ran,_,values,frame⟩ := SourceFinalResult.executes m start M N BM BN
    have sums := fold_eq m start M N BM BN 0 _ _ (SourceFinalGraph.zero m BM BN) hw hb
    refine ⟨t,(runSchedule_singleton _ _).trans ran,?_,?_⟩
    · intro o j hj
      fin_cases o
      · have hv := (values j hj).1
        change t.mem "FINAL_DW" (pid*BN+j.1.val)=_ at hv
        change mapCell (observe "FINAL_DW" (pid*BN+j.1.val)) (t.mem "FINAL_DW" (pid*BN+j.1.val))=_
        rw [hv,sums]
        rfl
      · have hv := (values j hj).2
        change t.mem "FINAL_DB" (pid*BN+j.1.val)=_ at hv
        change mapCell (observe "FINAL_DB" (pid*BN+j.1.val)) (t.mem "FINAL_DB" (pid*BN+j.1.val))=_
        rw [hv,sums]
        rfl
    · intro r off outside
      apply frame r off
      by_cases hw : r="FINAL_DW"
      · subst r
        right
        intro j hj
        exact (outside.1 (0 : Fin 2) j hj).resolve_left (by simp [Final.io,Final.outputs])
      · by_cases hb : r="FINAL_DB"
        · subst r
          right
          intro j hj
          exact (outside.1 (1 : Fin 2) j hj).resolve_left (by simp [Final.io,Final.outputs])
        · exact .inl ⟨hw,hb⟩

#axiomsClean implements
end
end VeriTile.Bench.Repairs.LayerNormTriton.FinalIO
