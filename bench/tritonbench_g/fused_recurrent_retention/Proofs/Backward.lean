import bench.tritonbench_g.fused_recurrent_retention.Proofs.Forward
namespace VeriTile.Bench.Repairs.FusedRecurrentRetention
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

def backwardPartials (a : Arithmetic α) (s : Shape) (p : Plans) (b : InputBuffers)
    (alloc : Allocation 6 caller) (seed final : Bool) (m : Memory α) : Memory α :=
  let dq := partialPhase a s p b seed final .dq (alloc.storage 0) (s.NV*rowCount s s.DK) m
  let dk := partialPhase a s p b seed final .dk (alloc.storage 1) (s.NV*rowCount s s.DK) dq
  partialPhase a s p b seed final .dv (alloc.storage 2) (s.NK*rowCount s s.DV) dk

def backwardRun (a : Arithmetic α) (s : Shape) (p : Plans) (b : InputBuffers)
    (alloc : Allocation 6 caller) (seed final : Bool) (m : Memory α) : Memory α :=
  let intermediate := backwardPartials a s p b alloc seed final m
  let dq := reduction a p.sumQ (alloc.storage 0) (alloc.storage 3) (rowCount s s.DK) intermediate
  let dk := reduction a p.sumK (alloc.storage 1) (alloc.storage 4) (rowCount s s.DK) dq
  reduction a p.sumV (alloc.storage 2) (alloc.storage 5) (rowCount s s.DV) dk

def barriersCode (n : Nat) : Code α Unit := foldCode (fun _ _ => emit .barrier) n ()

@[simp] theorem eval_barriersCode (a : Arithmetic α) (n : Nat) (m : Memory α) :
    (barriersCode n).eval a m=((),m) := by
  induction n with
  | zero => rfl
  | succ n ih =>
    change (foldCode (fun _ _ => emit .barrier) n () >>= fun _ => emit .barrier).eval a m=_
    simp only [bind,eval_bind,show (foldCode (fun _ _ => emit .barrier) n ()).eval a m=((),m) from ih,
      Prod.fst,Prod.snd,eval_emit,Event.eval]

def backwardCode (s : Shape) (p : Plans) (b : InputBuffers) (alloc : Allocation 6 caller)
    (seed final : Bool) : Code α Unit := do
  let _ ← partialPhaseCode s p b seed final .dq (alloc.storage 0) (s.NV*rowCount s s.DK)
  let _ ← barriersCode (s.NV*s.NK*s.BH)
  let _ ← partialPhaseCode s p b seed final .dk (alloc.storage 1) (s.NV*rowCount s s.DK)
  let _ ← partialPhaseCode s p b seed final .dv (alloc.storage 2) (s.NK*rowCount s s.DV)
  let _ ← reductionCode p.sumQ (alloc.storage 0) (alloc.storage 3) (rowCount s s.DK)
  let _ ← reductionCode p.sumK (alloc.storage 1) (alloc.storage 4) (rowCount s s.DK)
  reductionCode p.sumV (alloc.storage 2) (alloc.storage 5) (rowCount s s.DV)

theorem eval_backwardCode (a : Arithmetic α) (s : Shape) (p : Plans) (b : InputBuffers)
    (alloc : Allocation 6 caller) (inputs : ∀ id,id∈b.storageIds → id∈caller)
    (seed final : Bool) (m : Memory α) :
    (backwardCode s p b alloc seed final).eval a m=((),backwardRun a s p b alloc seed final m) := by
  simp [backwardCode,backwardRun,backwardPartials,bind,eval_bind,
    eval_partialPhaseCode a s p b _ _ _ _ _ _ (alloc.input_fresh b inputs 0),
    eval_partialPhaseCode a s p b _ _ _ _ _ _ (alloc.input_fresh b inputs 1),
    eval_partialPhaseCode a s p b _ _ _ _ _ _ (alloc.input_fresh b inputs 2),
    eval_reductionCode a _ _ _ _ _ (alloc.ne 3 0 (by decide)),
    eval_reductionCode a _ _ _ _ _ (alloc.ne 4 1 (by decide)),
    eval_reductionCode a _ _ _ _ _ (alloc.ne 5 2 (by decide))]

theorem backward_partialQ (a : Arithmetic α) (s : Shape) (p : Plans) (b : InputBuffers)
    (alloc : Allocation 6 caller) (seed final : Bool) (m : Memory α)
    (split h t k : Nat) (hs : split<s.NV) (hh : h<s.BH) (ht : t<s.T) (hk : k<s.DK) :
    backwardPartials a s p b alloc seed final m (alloc.storage 0,(partialOffset s s.DK split h t k : Int))=
      partialQ a s p (b.observe s m) seed h t split k := by
  unfold backwardPartials
  rw [partialPhase_frame _ _ _ _ _ _ _ _ _ _ _ (alloc.ne 0 2 (by decide)),
    partialPhase_frame _ _ _ _ _ _ _ _ _ _ _ (alloc.ne 0 1 (by decide))]
  unfold partialPhase
  rw [fillRange_get _ _ _ (partialOffset_bound s s.NV s.DK split h t k hs hh ht hk)]
  exact partialQ_at a s p _ seed final split h t k hh ht hk

theorem backward_partialK (a : Arithmetic α) (s : Shape) (p : Plans) (b : InputBuffers)
    (alloc : Allocation 6 caller) (inputs : ∀ id,id∈b.storageIds → id∈caller)
    (seed final : Bool) (m : Memory α)
    (split h t k : Nat) (hs : split<s.NV) (hh : h<s.BH) (ht : t<s.T) (hk : k<s.DK) :
    backwardPartials a s p b alloc seed final m (alloc.storage 1,(partialOffset s s.DK split h t k : Int))=
      partialK a s p (b.observe s m) final h t split k := by
  unfold backwardPartials
  rw [partialPhase_frame _ _ _ _ _ _ _ _ _ _ _ (alloc.ne 1 2 (by decide))]
  unfold partialPhase
  rw [fillRange_get _ _ _ (partialOffset_bound s s.NV s.DK split h t k hs hh ht hk),
    fillRange_observations b s (alloc.storage 0) _ _ m (alloc.input_fresh b inputs 0)]
  exact partialK_at a s p _ seed final split h t k hh ht hk

theorem backward_partialV (a : Arithmetic α) (s : Shape) (p : Plans) (b : InputBuffers)
    (alloc : Allocation 6 caller) (inputs : ∀ id,id∈b.storageIds → id∈caller)
    (seed final : Bool) (m : Memory α)
    (split h t v : Nat) (hs : split<s.NK) (hh : h<s.BH) (ht : t<s.T) (hv : v<s.DV) :
    backwardPartials a s p b alloc seed final m (alloc.storage 2,(partialOffset s s.DV split h t v : Int))=
      partialV a s p (b.observe s m) final h t split v := by
  unfold backwardPartials partialPhase
  rw [fillRange_get _ _ _ (partialOffset_bound s s.NK s.DV split h t v hs hh ht hv),
    fillRange_observations b s (alloc.storage 1) _ _ _ (alloc.input_fresh b inputs 1),
    fillRange_observations b s (alloc.storage 0) _ _ m (alloc.input_fresh b inputs 0)]
  exact partialV_at a s p _ seed final split h t v hh ht hv

theorem backward_dq (a : Arithmetic α) (s : Shape) (p : Plans) (hp : p.Valid s) (b : InputBuffers)
    (alloc : Allocation 6 caller) (seed final : Bool) (m : Memory α)
    (h t k : Nat) (hh : h<s.BH) (ht : t<s.T) (hk : k<s.DK) :
    backwardRun a s p b alloc seed final m (alloc.storage 3,(rowAddress s s.DK h t k : Int))=
      resultQ a s p (b.observe s m) seed h t k := by
  unfold backwardRun
  rw [reduction_frame _ _ _ _ _ _ _ (alloc.ne 3 5 (by decide)),
    reduction_frame _ _ _ _ _ _ _ (alloc.ne 3 4 (by decide)),
    reduction_value _ _ _ _ _ _ _ (rowAddress_bound s s.DK h t k hh ht hk)]
  apply (p.sumQ (rowAddress s s.DK h t k)).eval_congr_bound a s.NV (hp.sumQ _)
  intro split hs
  rw [row_to_partial]
  exact backward_partialQ a s p b alloc seed final m split h t k hs hh ht hk

theorem backward_dk (a : Arithmetic α) (s : Shape) (p : Plans) (hp : p.Valid s) (b : InputBuffers)
    (alloc : Allocation 6 caller) (inputs : ∀ id,id∈b.storageIds → id∈caller)
    (seed final : Bool) (m : Memory α)
    (h t k : Nat) (hh : h<s.BH) (ht : t<s.T) (hk : k<s.DK) :
    backwardRun a s p b alloc seed final m (alloc.storage 4,(rowAddress s s.DK h t k : Int))=
      resultK a s p (b.observe s m) final h t k := by
  unfold backwardRun
  rw [reduction_frame _ _ _ _ _ _ _ (alloc.ne 4 5 (by decide)),
    reduction_value _ _ _ _ _ _ _ (rowAddress_bound s s.DK h t k hh ht hk)]
  apply (p.sumK (rowAddress s s.DK h t k)).eval_congr_bound a s.NV (hp.sumK _)
  intro split hs
  rw [reduction_frame _ _ _ _ _ _ _ (alloc.ne 1 3 (by decide)),row_to_partial]
  exact backward_partialK a s p b alloc inputs seed final m split h t k hs hh ht hk

theorem backward_dv (a : Arithmetic α) (s : Shape) (p : Plans) (hp : p.Valid s) (b : InputBuffers)
    (alloc : Allocation 6 caller) (inputs : ∀ id,id∈b.storageIds → id∈caller)
    (seed final : Bool) (m : Memory α)
    (h t v : Nat) (hh : h<s.BH) (ht : t<s.T) (hv : v<s.DV) :
    backwardRun a s p b alloc seed final m (alloc.storage 5,(rowAddress s s.DV h t v : Int))=
      resultV a s p (b.observe s m) final h t v := by
  unfold backwardRun
  rw [reduction_value _ _ _ _ _ _ _ (rowAddress_bound s s.DV h t v hh ht hv)]
  apply (p.sumV (rowAddress s s.DV h t v)).eval_congr_bound a s.NK (hp.sumV _)
  intro split hs
  rw [reduction_frame _ _ _ _ _ _ _ (alloc.ne 2 4 (by decide)),
    reduction_frame _ _ _ _ _ _ _ (alloc.ne 2 3 (by decide)),row_to_partial]
  exact backward_partialV a s p b alloc inputs seed final m split h t v hs hh ht hv

theorem backward_frame (a : Arithmetic α) (s : Shape) (p : Plans) (b : InputBuffers)
    (alloc : Allocation 6 caller) (seed final : Bool) (m : Memory α) (q : Address) (hq : q.1∈caller) :
    backwardRun a s p b alloc seed final m q=m q := by
  have hn (i : Fin 6) : q.1≠alloc.storage i := by
    intro he
    exact alloc.fresh i (he ▸ hq)
  unfold backwardRun backwardPartials
  rw [reduction_frame _ _ _ _ _ _ _ (hn 5),reduction_frame _ _ _ _ _ _ _ (hn 4),
    reduction_frame _ _ _ _ _ _ _ (hn 3),partialPhase_frame _ _ _ _ _ _ _ _ _ _ _ (hn 2),
    partialPhase_frame _ _ _ _ _ _ _ _ _ _ _ (hn 1),partialPhase_frame _ _ _ _ _ _ _ _ _ _ _ (hn 0)]

#axiomsClean eval_backwardCode
#axiomsClean backward_dq
#axiomsClean backward_dk
#axiomsClean backward_dv
#axiomsClean backward_frame
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention
