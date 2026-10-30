import bench.repairs.tritonbench_g.nested_loops_processing.Proofs.Contract
import bench.repairs.tritonbench_g.nested_loops_processing.Misc.Legacy.CorrectPublic

namespace VeriTile.Bench.Repairs.NestedLoopsProcessing.HistoricalCorrect
open Host Shared.Storage
open VeriTile.Bench.Repairs.LayernormFwdTriton.Memory
noncomputable section

theorem checked_printed (api : API Word Store Byte) (lib : Library api backings)
    (request : Request) (rowIsInt colIsInt : Bool) (s : State Store)
    (ok : ((Admission.wrapper api lib rowIsInt colIsInt request).run s).result=.ok ()) :
    Admission.dimensions rowIsInt colIsInt request.rows request.cols = Bool.true ∧
    ∃ (x : Shaped request) (y : Output backings x),
      ((allocate api lib x).run ((input api lib request).run s).state).result=.ok y ∧
      ((Admission.wrapper api lib rowIsInt colIsInt request).run s).state.printed=some ⟨[x.rows,x.cols],y.tensor⟩ ∧
      ∀ i,i<x.rows*x.cols →
        (signedValue (y.tensor.read api ((Admission.wrapper api lib rowIsInt colIsInt request).run s).state.memory i):ℝ)=
          realTarget x.rows x.cols i := by
  have admitted := Admission.success_checked api lib rowIsInt colIsInt request s ok
  refine ⟨admitted,?_⟩
  rw [Admission.wrapper_eq api lib rowIsInt colIsInt request admitted] at ok ⊢
  exact Correct.source_success api lib request s ok

theorem checked_all_outcomes_frame (api : API Word Store Byte) (lib : Library api backings)
    (request : Request) (rowIsInt colIsInt : Bool) (s : State Store) :
    Frame api backings s.memory ((Admission.wrapper api lib rowIsInt colIsInt request).run s).state.memory :=
  (Admission.wrapper api lib rowIsInt colIsInt request).frame s

abbrev source_addresses := @Admission.checked_addresses
abbrev rejected_unchanged := @Admission.rejected_unchanged

#axiomsClean checked_printed
#axiomsClean checked_all_outcomes_frame
#specNonCircular realTarget avoiding [Admission.wrapper,Host.wrapper,sourceStores,execute]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.NestedLoopsProcessing.HistoricalCorrect
