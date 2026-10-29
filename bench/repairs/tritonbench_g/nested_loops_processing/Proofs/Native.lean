import bench.repairs.tritonbench_g.nested_loops_processing.Proofs.Library

namespace VeriTile.Bench.Repairs.NestedLoopsProcessing.Host
open VeriTile.Bench.Repairs.NestedLoopsProcessing.Shared.Storage
open VeriTile.Bench.Repairs.NestedLoopsProcessing.Shared.Address (Dense)
open VeriTile.Bench.Repairs.LayernormFwdTriton.Memory
open VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
noncomputable section
open scoped Classical

/-- Source-order phase linearization is justified by the uniform bar.sync
contract, including different-thread conflicts. Within each phase the existing
active-lane permutation law remains the only ordering choice. -/
theorem trial_cross_store_order (trial : Trial count cols) (t u : Nat)
    (order : t<u) (bound : u<24) (a b : Lane) :
    trial.synchronization.completed t a < trial.synchronization.completed u b :=
  trial.synchronization.cross_store_order t u order bound a b

def trialState (api : API Word Store Byte) (x : Shaped request) (y : Output backings x)
    (trial : Trial (x.rows*x.cols) x.cols) (s : State Store) : State Store :=
  {s with memory := (writeOwner s.memory y.tensor.owner
    (execute api 0 y.placement trial.writes (s.memory y.tensor.owner)))}

theorem trial_frame (api : API Word Store Byte) (x : Shaped request) (y : Output backings x)
    (trial : Trial (x.rows*x.cols) x.cols) (s : State Store) :
    Frame api backings s.memory (trialState api x y trial s).memory := by
  apply writeOwner_frame
  intro key hk owner
  apply execute_frame
  intro inside
  apply y.callerApart key hk
  refine ⟨owner,?_,?_⟩
  · exact inside.1
  · simpa [Output.placement,Packed.backing] using inside.2

/-- The actual completed bounded writes, followed by the native completion
callback. Success uses all source stores in statement order and any lane
linearization. Failure permits any actual partially completed tile writes. -/
def trialAction (api : API Word Store Byte) (x : Shaped request) (y : Output backings x)
    (trial : Trial (x.rows*x.cols) x.cols) (finish : Effect api backings Unit) :
    Action (world api backings) Unit where
  run s:=
    let last:=(call .execute finish.call).run (trialState api x y trial s)
    ⟨last.state,match last.result with
      | .error e=>.error e
      | .ok _=>match trial.failure with
        | some code=>.error (.api .execute code)
        | none=>.ok (),[.execute]++last.trace⟩
  frame s key hk:=(finish.call.frame (trialState api x y trial s) key hk).trans
    (trial_frame api x y trial s key hk)

theorem trial_success (api : API Word Store Byte) (x : Shaped request) (y : Output backings x)
    (trial : Trial (x.rows*x.cols) x.cols) (finish : Effect api backings Unit) (s : State Store)
    (ok : ((trialAction api x y trial finish).run s).result=.ok ()) : trial.failure=none := by
  cases hf:(finish.call.run (trialState api x y trial s)).2 with
  | error e=>simp [trialAction,call,hf,Except.mapError] at ok
  | ok u=>
    cases ht:trial.failure with
    | none=>rfl
    | some code=>simp [trialAction,call,hf,ht,Except.mapError] at ok

theorem trial_after_state (api : API Word Store Byte) (x : Shaped request) (y : Output backings x)
    (trial : Trial (x.rows*x.cols) x.cols) (finish : Effect api backings Unit) (s : State Store) :
    ((trialAction api x y trial finish).run s).state=
      (finish.call.run (trialState api x y trial s)).1 := rfl

theorem trial_output (api : API Word Store Byte) (x : Shaped request) (y : Output backings x)
    (trial : Trial (x.rows*x.cols) x.cols) (finish : Effect api backings Unit) (s : State Store)
    (positive : 0<x.rows*x.cols) (input : Values api s x.tensor arangeWord)
    (initial : Values api s y.tensor (fun _=>0))
    (ok : ((trialAction api x y trial finish).run s).result=.ok ()) :
    Values api ((trialAction api x y trial finish).run s).state y.tensor
      (target 0 (x.rows*x.cols) x.cols arangeWord (fun _=>0)) := by
  have success:=trial_success api x y trial finish s ok
  have complete:=trial.complete success
  have columns:0<x.cols:=Nat.pos_of_mul_pos_left positive
  refine ⟨finish.retained (trialState api x y trial s) _ initial.1,?_⟩
  intro i hi
  change y.tensor.read api (finish.call.run (trialState api x y trial s)).1.memory i=_
  rw [effect_packed api finish (trialState api x y trial s) y.tensor initial.1 i hi]
  simp only [Packed.read,trialState,writeOwner,Function.update_self,Trial.writes,success,↓reduceIte,List.take_of_length_le complete]
  have bound:i<x.rows*x.cols:=by simpa using hi
  change api.read (execute api 0 y.placement trial.schedule.order (s.memory y.tensor.owner))
    (y.placement.output.address i)=_
  rw [execute_schedule_readback api 0 y.placement trial.schedule columns _ i bound]
  apply target_congr_inside 0 (x.rows*x.cols) x.cols columns _ arangeWord _ (fun _=>0) ?_ ?_ i bound
  · intro j hj
    simpa [readInput,Output.placement,Packed.read,Dense.address,y.owner] using input.2 j (by simpa using hj)
  · intro j hj
    simpa [Output.placement,Packed.read,Dense.address] using initial.2 j (by simpa using hj)

#axiomsClean trial_cross_store_order
#axiomsClean trial_frame
#axiomsClean trial_after_state
#axiomsClean trial_output
end
end VeriTile.Bench.Repairs.NestedLoopsProcessing.Host
