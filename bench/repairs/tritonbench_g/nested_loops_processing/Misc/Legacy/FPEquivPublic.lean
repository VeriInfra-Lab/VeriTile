import bench.repairs.tritonbench_g.nested_loops_processing.Proofs.Wrapper
import bench.repairs.tritonbench_g.nested_loops_processing.Proofs.Real

/-! Public production int32 equivalence. The file name follows the benchmark
entry convention; this task has no floating payload arithmetic and no deferred
numerical ports. Byte representation is an ordinary local storage contract. -/
namespace VeriTile.Bench.Repairs.NestedLoopsProcessing.FPEquiv
open Host
open VeriTile.Bench.Repairs.NestedLoopsProcessing.Shared.Storage
open VeriTile.Bench.Repairs.LayernormFwdTriton.Memory
noncomputable section

def numericalObligations : List String := []

theorem source_success (api : API Word Store Byte) (codec : Codec api)
    (lib : Library api backings) (request : Request) (s : State Store)
    (ok : ((wrapper api lib request).run s).result=.ok ()) :
    ∃ (x : Shaped request) (y : Output backings x),
      ((allocate api lib x).run ((input api lib request).run s).state).result=.ok y ∧
      ((wrapper api lib request).run s).state.printed=some ⟨[x.rows,x.cols],y.tensor⟩ ∧
      (∀ i,i<x.rows*x.cols → y.tensor.read api ((wrapper api lib request).run s).state.memory i=
        target 0 (x.rows*x.cols) x.cols arangeWord (fun _=>0) i) ∧
      (∀ i,i<x.rows*x.cols → ∀ offset,offset<4 →
        api.byte (((wrapper api lib request).run s).state.memory y.tensor.owner) (y.tensor.dense.address i+offset)=
          (target 0 (x.rows*x.cols) x.cols arangeWord (fun _=>0) i).extractLsb' (8*offset) 8) ∧
      (∀ i,i<x.rows*x.cols →
        (signedValue (y.tensor.read api ((wrapper api lib request).run s).state.memory i):ℝ)=
          realTarget x.rows x.cols i) := by
  obtain ⟨x,y,allocated,printed,values⟩:=wrapper_success api lib request s ok
  refine ⟨x,y,allocated,printed,?_,?_,?_⟩
  · intro i bound;exact values.2 i (by simpa using bound)
  · intro i bound offset small
    rw [codec.byte _ _ _ small]
    change (y.tensor.read api ((wrapper api lib request).run s).state.memory i).extractLsb' (8*offset) 8=_
    rw [values.2 i (by simpa using bound)]
  · intro i bound
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
end VeriTile.Bench.Repairs.NestedLoopsProcessing.FPEquiv
