import bench.repairs.tritonbench_g.nested_loops_processing.Proofs.Wrapper
import bench.repairs.tritonbench_g.nested_loops_processing.Proofs.Real

/-! Public independent Real theorem for the separately named repaired source.
The output is the actual tensor passed to print; the wrapper returns None. -/
namespace VeriTile.Bench.Repairs.NestedLoopsProcessing.Correct
open Host
open VeriTile.Bench.Repairs.NestedLoopsProcessing.Shared.Storage
open VeriTile.Bench.Repairs.LayernormFwdTriton.Memory
noncomputable section

theorem source_success (api : API Word Store Byte) (lib : Library api backings)
    (request : Request) (s : State Store)
    (ok : ((wrapper api lib request).run s).result=.ok ()) :
    ∃ (x : Shaped request) (y : Output backings x),
      ((allocate api lib x).run ((input api lib request).run s).state).result=.ok y ∧
      ((wrapper api lib request).run s).state.printed=some ⟨[x.rows,x.cols],y.tensor⟩ ∧
      ∀ i,i<x.rows*x.cols →
        (signedValue (y.tensor.read api ((wrapper api lib request).run s).state.memory i):ℝ)=
          realTarget x.rows x.cols i := by
  obtain ⟨x,y,allocated,printed,values⟩:=wrapper_success api lib request s ok
  refine ⟨x,y,allocated,printed,?_⟩
  intro i bound
  rw [values.2 i (by simpa using bound)]
  exact int32_real_exact x.rows x.cols i

theorem all_outcomes_frame (api : API Word Store Byte) (lib : Library api backings)
    (request : Request) (s : State Store) :
    Frame api backings s.memory ((wrapper api lib request).run s).state.memory :=
  wrapper_frame api lib request s

#axiomsClean source_success
#axiomsClean all_outcomes_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.NestedLoopsProcessing.Correct
