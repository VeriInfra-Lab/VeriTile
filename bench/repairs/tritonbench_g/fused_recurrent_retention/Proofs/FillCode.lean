import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.PartialCode
namespace VeriTile.Bench.Repairs.FusedRecurrentRetention
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

def Agrees (ids : Finset Nat) (m m' : Memory α) : Prop :=
  ∀ p,p.1∈ids → m p=m' p

theorem agrees_refl (ids : Finset Nat) (m : Memory α) : Agrees ids m m := fun _ _ => rfl

theorem agrees_write (ids : Finset Nat) (m before : Memory α) (storage index : Nat) (value : α)
    (fresh : storage∉ids) (same : Agrees ids m before) :
    Agrees ids (writeWord m (storage,(index : Int)) value) before := by
  intro p hp
  rw [writeWord_frame]
  · exact same p hp
  · intro he
    apply fresh
    simpa only [he,Prod.fst] using hp

theorem agrees_fill (ids : Finset Nat) (m before : Memory α) (storage count : Nat) (value : Nat → α)
    (fresh : storage∉ids) (same : Agrees ids m before) :
    Agrees ids (fillRange storage count value m) before := by
  intro p hp
  rw [fillRange_frame]
  · exact same p hp
  · intro he
    exact fresh (he ▸ hp)

def fillCode (storage : Nat) (value : Nat → Code α α) : List Nat → Code α Unit
  | [] => pure ()
  | i::rest => do
    let x ← value i
    let _ ← emit (.store (storage,(i : Int)) x)
    fillCode storage value rest

theorem eval_fillCode (a : Arithmetic α) (ids : Finset Nat) (storage : Nat)
    (value : Nat → Code α α) (expected : Nat → α) (before : Memory α)
    (fresh : storage∉ids)
    (law : ∀ i current,Agrees ids current before → (value i).eval a current=(expected i,current))
    (items : List Nat) (m : Memory α) (same : Agrees ids m before) :
    (fillCode storage value items).eval a m=((),fill storage expected items m) := by
  induction items generalizing m with
  | nil => rfl
  | cons i rest ih =>
    simp only [fillCode,bind,eval_bind,law i m same,Prod.fst,Prod.snd,eval_emit,Event.eval]
    rw [ih _ (agrees_write ids m before storage i (expected i) fresh same)]
    rfl

def partialPhaseCode (s : Shape) (p : Plans) (b : InputBuffers) (seed final : Bool)
    (kind : PartialKind) (storage count : Nat) : Code α Unit :=
  fillCode storage (partialCellCode s p b seed final kind) (List.range count)

def partialPhase (a : Arithmetic α) (s : Shape) (p : Plans) (b : InputBuffers) (seed final : Bool)
    (kind : PartialKind) (storage count : Nat) (m : Memory α) : Memory α :=
  fillRange storage count (partialValue a s p (b.observe s m) seed final kind) m

@[simp] theorem eval_partialPhaseCode (a : Arithmetic α) (s : Shape) (p : Plans) (b : InputBuffers)
    (seed final : Bool) (kind : PartialKind) (storage count : Nat) (m : Memory α)
    (fresh : storage∉b.storageIds) :
    (partialPhaseCode s p b seed final kind storage count).eval a m=
      ((),partialPhase a s p b seed final kind storage count m) := by
  apply eval_fillCode a b.storageIds storage _ _ m fresh _ _ m (agrees_refl _ _)
  intro i current same
  rw [eval_partialCellCode,observations_congr b s current m same]

def reduceCellCode (plan : SumPlan) (input count offset : Nat) : Code α α :=
  plan.code (fun split => emit (.read (input,((split*count+offset : Nat) : Int))))

@[simp] theorem eval_reduceCellCode (a : Arithmetic α) (plan : SumPlan)
    (input count offset : Nat) (m : Memory α) :
    (reduceCellCode plan input count offset).eval a m=
      (plan.eval a (fun split => m (input,((split*count+offset : Nat) : Int))),m) := by
  apply plan.eval_code a _ _ m
  intro i
  rfl

def reductionCode (plan : Nat → SumPlan) (input output count : Nat) : Code α Unit :=
  fillCode output (fun offset => reduceCellCode (plan offset) input count offset) (List.range count)

@[simp] theorem eval_reductionCode (a : Arithmetic α) (plan : Nat → SumPlan)
    (input output count : Nat) (m : Memory α) (fresh : output≠input) :
    (reductionCode plan input output count).eval a m=((),reduction a plan input output count m) := by
  apply eval_fillCode a {input} output _ _ m (by simpa) _ _ m (agrees_refl _ _)
  intro i current same
  rw [eval_reduceCellCode]
  congr 1
  congr 1
  funext split
  exact same _ (by simp)

theorem reduction_value (a : Arithmetic α) (plan : Nat → SumPlan) (input output count offset : Nat)
    (m : Memory α) (ho : offset<count) :
    reduction a plan input output count m (output,(offset : Int))=
      (plan offset).eval a (fun split => m (input,((split*count+offset : Nat) : Int))) :=
  fillRange_get output count offset ho _ m

theorem reduction_frame (a : Arithmetic α) (plan : Nat → SumPlan) (input output count : Nat)
    (m : Memory α) (p : Address) (hp : p.1≠output) :
    reduction a plan input output count m p=m p := fillRange_frame _ _ _ _ _ hp

#axiomsClean eval_fillCode
#axiomsClean eval_partialPhaseCode
#axiomsClean eval_reductionCode
#axiomsClean reduction_value
#axiomsClean reduction_frame
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention
