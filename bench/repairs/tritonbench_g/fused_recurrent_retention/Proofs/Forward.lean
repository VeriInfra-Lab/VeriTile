import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.Allocations
namespace VeriTile.Bench.Repairs.FusedRecurrentRetention
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

def forwardPartials (a : Arithmetic α) (s : Shape) (p : Plans) (b : InputBuffers)
    (alloc : Allocation 3 caller) (seed storeFinal : Bool) (m : Memory α) : Memory α :=
  let intermediate := partialPhase a s p b seed Bool.false .output (alloc.storage 0) (s.NK*rowCount s s.DV) m
  if storeFinal then partialPhase a s p b seed Bool.false .finalState (alloc.storage 1) (s.BH*s.DK*s.DV) intermediate
  else intermediate

def forwardRun (a : Arithmetic α) (s : Shape) (p : Plans) (b : InputBuffers)
    (alloc : Allocation 3 caller) (seed storeFinal : Bool) (m : Memory α) : Memory α :=
  reduction a p.sumO (alloc.storage 0) (alloc.storage 2) (rowCount s s.DV)
    (forwardPartials a s p b alloc seed storeFinal m)

def forwardCode (s : Shape) (p : Plans) (b : InputBuffers) (alloc : Allocation 3 caller)
    (seed storeFinal : Bool) : Code α Unit := do
  let _ ← partialPhaseCode s p b seed Bool.false .output (alloc.storage 0) (s.NK*rowCount s s.DV)
  let _ ← if storeFinal then
    partialPhaseCode s p b seed Bool.false .finalState (alloc.storage 1) (s.BH*s.DK*s.DV)
    else pure ()
  reductionCode p.sumO (alloc.storage 0) (alloc.storage 2) (rowCount s s.DV)

theorem eval_forwardCode (a : Arithmetic α) (s : Shape) (p : Plans) (b : InputBuffers)
    (alloc : Allocation 3 caller) (inputs : ∀ id,id∈b.storageIds → id∈caller)
    (seed storeFinal : Bool) (m : Memory α) :
    (forwardCode s p b alloc seed storeFinal).eval a m=((),forwardRun a s p b alloc seed storeFinal m) := by
  cases storeFinal <;>
    simp [forwardCode,forwardRun,forwardPartials,bind,eval_bind,Event.eval,pure,
      eval_partialPhaseCode a s p b _ _ _ _ _ _ (alloc.input_fresh b inputs 0),
      eval_partialPhaseCode a s p b _ _ _ _ _ _ (alloc.input_fresh b inputs 1),
      eval_reductionCode a _ _ _ _ _ (alloc.ne 2 0 (by decide))]

theorem forward_partials_value (a : Arithmetic α) (s : Shape) (p : Plans) (b : InputBuffers)
    (alloc : Allocation 3 caller) (seed storeFinal : Bool) (m : Memory α)
    (split h t v : Nat) (hs : split<s.NK) (hh : h<s.BH) (ht : t<s.T) (hv : v<s.DV) :
    forwardPartials a s p b alloc seed storeFinal m
      (alloc.storage 0,(partialOffset s s.DV split h t v : Int))=
        partialO a s p (b.observe s m) seed h t split v := by
  have hb := partialOffset_bound s s.NK s.DV split h t v hs hh ht hv
  cases storeFinal <;> unfold forwardPartials
  · simp only [Bool.false_eq_true,↓reduceIte,partialPhase]
    rw [fillRange_get _ _ _ hb]
    exact partialO_at a s p _ seed Bool.false split h t v hh ht hv
  · simp only [↓reduceIte]
    rw [partialPhase_frame _ _ _ _ _ _ _ _ _ _ _ (alloc.ne 0 1 (by decide))]
    unfold partialPhase
    rw [fillRange_get _ _ _ hb]
    exact partialO_at a s p _ seed Bool.false split h t v hh ht hv

theorem forward_output (a : Arithmetic α) (s : Shape) (p : Plans) (hp : p.Valid s) (b : InputBuffers)
    (alloc : Allocation 3 caller) (seed storeFinal : Bool) (m : Memory α)
    (h t v : Nat) (hh : h<s.BH) (ht : t<s.T) (hv : v<s.DV) :
    forwardRun a s p b alloc seed storeFinal m (alloc.storage 2,(rowAddress s s.DV h t v : Int))=
      resultO a s p (b.observe s m) seed h t v := by
  unfold forwardRun
  rw [reduction_value _ _ _ _ _ _ _ (rowAddress_bound s s.DV h t v hh ht hv)]
  apply (p.sumO (rowAddress s s.DV h t v)).eval_congr_bound a s.NK (hp.sumO _)
  intro split hs
  rw [row_to_partial]
  exact forward_partials_value a s p b alloc seed storeFinal m split h t v hs hh ht hv

theorem forward_state (a : Arithmetic α) (s : Shape) (p : Plans) (b : InputBuffers)
    (alloc : Allocation 3 caller) (inputs : ∀ id,id∈b.storageIds → id∈caller)
    (seed : Bool) (m : Memory α) (h k v : Nat) (hh : h<s.BH) (hk : k<s.DK) (hv : v<s.DV) :
    forwardRun a s p b alloc seed Bool.true m (alloc.storage 1,(stateOffset s h k v : Int))=
      state a s (b.observe s m) seed h k v s.T := by
  unfold forwardRun
  rw [reduction_frame _ _ _ _ _ _ _ (alloc.ne 1 2 (by decide))]
  simp only [forwardPartials,↓reduceIte,partialPhase]
  rw [fillRange_get _ _ _ (stateOffset_bound s h k v hh hk hv),
    fillRange_observations b s (alloc.storage 0) _ _ m (alloc.input_fresh b inputs 0)]
  exact partialState_at a s p _ seed Bool.false h k v hk hv

theorem forward_frame (a : Arithmetic α) (s : Shape) (p : Plans) (b : InputBuffers)
    (alloc : Allocation 3 caller) (seed storeFinal : Bool) (m : Memory α) (q : Address) (hq : q.1∈caller) :
    forwardRun a s p b alloc seed storeFinal m q=m q := by
  have hn (i : Fin 3) : q.1≠alloc.storage i := by
    intro he
    exact alloc.fresh i (he ▸ hq)
  unfold forwardRun
  rw [reduction_frame _ _ _ _ _ _ _ (hn 2)]
  cases storeFinal <;>
    simp [forwardPartials,partialPhase,fillRange_frame _ _ _ _ _ (hn 0),
      fillRange_frame _ _ _ _ _ (hn 1)]

#axiomsClean eval_forwardCode
#axiomsClean forward_output
#axiomsClean forward_state
#axiomsClean forward_frame
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention
