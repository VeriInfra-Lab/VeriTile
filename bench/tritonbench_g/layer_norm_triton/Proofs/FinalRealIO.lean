import bench.tritonbench_g.layer_norm_triton.Proofs.FinalIO
import bench.tritonbench_g.layer_norm_triton.Proofs.SourceFinalReal

namespace VeriTile.Bench.Repairs.LayerNormTriton.FinalRealIO
open VeriTile.Triton FP.Structural SourceModel Interface
noncomputable section

def formula (M BN : Nat) (vals : Final.Inputs ℝ M BN) : Final.Outputs ℝ BN :=
  Final.result (fun j => ∑ row ∈ Finset.range M,Final.value vals 0 row j)
    (fun j => ∑ row ∈ Finset.range M,Final.value vals 1 row j)

theorem implements (m : Model ℝ) (laws : SourceFinalReal.Laws m) (M N BN pid : Nat) :
    (Final.io m M N 32 BN pid (fun _ _ => id)).Implements (formula M BN) := by
  intro vals _ bounds boundsOK initial binding
  obtain ⟨safe,t,ran,_,frame⟩ := FinalIO.implements m M N 32 BN pid (fun _ _ => id)
    vals trivial bounds boundsOK initial binding
  let start := (job m (TypedSource.finalBackward "DW" "DB" "FINAL_DW" "FINAL_DB" M N 32 BN) pid).initial initial
  have executed : FP.Structural.exec (engine m)
      (TypedSource.finalBackward "DW" "DB" "FINAL_DW" "FINAL_DB" M N 32 BN) start=some t := by
    simpa only [Final.io,runSchedule_singleton] using ran
  have memory := SourceFinalGraph.source_refines m start M N 32 BN
  obtain ⟨u,hu,hm,values,_⟩ := SourceFinalResult.executes m start M N 32 BN
  have same : u=t := Option.some.inj (hu.symm.trans executed)
  subst u
  refine ⟨safe,t,ran,?_,frame⟩
  intro port lane active
  fin_cases port
  · have sums := SourceFinalReal.reduced_values m laws start M N BN lane active
    have output := (values lane active).1.trans (congrArg (Cell.mk .real) sums.1)
    change t.mem "FINAL_DW" (pid*BN+lane.1.val)=_ at output
    change mapCell id (t.mem "FINAL_DW" (pid*BN+lane.1.val))=_
    rw [output]
    change Cell.mk .real (∑ row ∈ Finset.range M,(start.mem "DW" (row*N+(pid*BN+lane.1.val))).read .real)=
      Cell.mk .real (∑ row ∈ Finset.range M,Final.value vals 0 row lane.1)
    apply congrArg (Cell.mk .real)
    apply Finset.sum_congr rfl
    intro row hr
    have hrow := Finset.mem_range.mp hr
    simpa [start,Invocation.initial,bindRegisters,Final.io,Final.inputs,Final.value,hrow,mapValue]
      using binding (0 : Fin 2) (⟨row,hrow⟩,lane.1,()) active
  · have sums := SourceFinalReal.reduced_values m laws start M N BN lane active
    have output := (values lane active).2.trans (congrArg (Cell.mk .real) sums.2)
    change t.mem "FINAL_DB" (pid*BN+lane.1.val)=_ at output
    change mapCell id (t.mem "FINAL_DB" (pid*BN+lane.1.val))=_
    rw [output]
    change Cell.mk .real (∑ row ∈ Finset.range M,(start.mem "DB" (row*N+(pid*BN+lane.1.val))).read .real)=
      Cell.mk .real (∑ row ∈ Finset.range M,Final.value vals 1 row lane.1)
    apply congrArg (Cell.mk .real)
    apply Finset.sum_congr rfl
    intro row hr
    have hrow := Finset.mem_range.mp hr
    simpa [start,Invocation.initial,bindRegisters,Final.io,Final.inputs,Final.value,hrow,mapValue]
      using binding (1 : Fin 2) (⟨row,hrow⟩,lane.1,()) active
#axiomsClean implements
end
end VeriTile.Bench.Repairs.LayerNormTriton.FinalRealIO
