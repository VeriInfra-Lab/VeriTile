import bench.tritonbench_g.fused_recurrent_retention.Proofs.FlatLayout
namespace VeriTile.Bench.Repairs.FusedRecurrentRetention
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

def scaledKeyCode (s : Shape) (b : InputBuffers) (h t k : Nat) : Code α α := do
  let q ← loadCode (decide (k<s.DK)) (b.qAddress s h t k)
  let scale ← emit (.scale s.DK)
  emit (.mul q scale)

@[simp] theorem eval_scaledKeyCode (a : Arithmetic α) (s : Shape) (b : InputBuffers)
    (h t k : Nat) (m : Memory α) :
    (scaledKeyCode s b h t k).eval a m=
      (a.mul (keyLoad a s (b.observe s m).q h t k) (a.scale s.DK),m) := by
  simp [scaledKeyCode,bind,eval_bind,Event.eval,keyLoad,InputBuffers.observe]

def partialOCode (s : Shape) (p : Plans) (b : InputBuffers) (seed : Bool) (h t split v : Nat) : Code α α :=
  p.output.code (fun j => stateCode s b seed h (split*s.BK+j) v (t+1))
    (fun j => scaledKeyCode s b h t (split*s.BK+j))

def partialQCode (s : Shape) (p : Plans) (b : InputBuffers) (seed : Bool) (h t split k : Nat) : Code α α := do
  let x ← p.dq.code (fun j => stateCode s b seed h k (split*s.BV+j) (t+1))
    (fun j => loadCode (decide (split*s.BV+j<s.DV)) (b.doAddress s h t (split*s.BV+j)))
  let scale ← emit (.scale s.DK)
  emit (.mul x scale)

def partialKCode (s : Shape) (p : Plans) (b : InputBuffers) (final : Bool) (h t split k : Nat) : Code α α :=
  p.dk.code (fun j => adjointCode s b final h t k (split*s.BV+j))
    (fun j => loadCode (decide (split*s.BV+j<s.DV)) (b.vAddress s h t (split*s.BV+j)))

def partialVCode (s : Shape) (p : Plans) (b : InputBuffers) (final : Bool) (h t split v : Nat) : Code α α :=
  p.dv.code (fun j => adjointCode s b final h t (split*s.BK+j) v)
    (fun j => loadCode (decide (split*s.BK+j<s.DK)) (b.kAddress s h t (split*s.BK+j)))

@[simp] theorem eval_partialOCode (a : Arithmetic α) (s : Shape) (p : Plans) (b : InputBuffers)
    (seed : Bool) (h t split v : Nat) (m : Memory α) :
    (partialOCode s p b seed h t split v).eval a m=(partialO a s p (b.observe s m) seed h t split v,m) := by
  exact p.output.eval_code a _ _ _ _ m (fun _ => eval_stateCode a _ _ _ _ _ _ _ _)
    (fun _ => eval_scaledKeyCode a _ _ _ _ _ _)

@[simp] theorem eval_partialQCode (a : Arithmetic α) (s : Shape) (p : Plans) (b : InputBuffers)
    (seed : Bool) (h t split k : Nat) (m : Memory α) :
    (partialQCode s p b seed h t split k).eval a m=(partialQ a s p (b.observe s m) seed h t split k,m) := by
  have he := p.dq.eval_code a
    (fun j => stateCode s b seed h k (split*s.BV+j) (t+1))
    (fun j => loadCode (decide (split*s.BV+j<s.DV)) (b.doAddress s h t (split*s.BV+j)))
    (fun j => state a s (b.observe s m) seed h k (split*s.BV+j) (t+1))
    (fun j => valueLoad a s (b.observe s m).do_ h t (split*s.BV+j)) m
    (fun _ => eval_stateCode a _ _ _ _ _ _ _ _)
    (fun _ => by simp [valueLoad,InputBuffers.observe])
  simp only [partialQCode,bind,eval_bind,he,Prod.fst,Prod.snd,eval_emit,Event.eval,partialQ]

@[simp] theorem eval_partialKCode (a : Arithmetic α) (s : Shape) (p : Plans) (b : InputBuffers)
    (final : Bool) (h t split k : Nat) (m : Memory α) :
    (partialKCode s p b final h t split k).eval a m=(partialK a s p (b.observe s m) final h t split k,m) := by
  apply p.dk.eval_code a _ _ _ _ m
  · intro j; exact eval_adjointCode a _ _ _ _ _ _ _ _
  · intro j; simp [valueLoad,InputBuffers.observe]

@[simp] theorem eval_partialVCode (a : Arithmetic α) (s : Shape) (p : Plans) (b : InputBuffers)
    (final : Bool) (h t split v : Nat) (m : Memory α) :
    (partialVCode s p b final h t split v).eval a m=(partialV a s p (b.observe s m) final h t split v,m) := by
  apply p.dv.eval_code a _ _ _ _ m
  · intro j; exact eval_adjointCode a _ _ _ _ _ _ _ _
  · intro j; simp [keyLoad,InputBuffers.observe]

def partialValue (a : Arithmetic α) (s : Shape) (p : Plans) (x : Inputs α)
    (seed final : Bool) (kind : PartialKind) (i : Nat) : α :=
  let w := kind.width s
  let h := flatHead s w i
  let t := flatTime s w i
  let split := flatSplit s w i
  let c := flatChannel w i
  match kind with
  | .output => partialO a s p x seed h t split c
  | .dq => partialQ a s p x seed h t split c
  | .dk => partialK a s p x final h t split c
  | .dv => partialV a s p x final h t split c
  | .finalState => state a s x seed (i/s.DV/s.DK) (i/s.DV%s.DK) (i%s.DV) s.T

def partialCellCode (s : Shape) (p : Plans) (b : InputBuffers) (seed final : Bool)
    (kind : PartialKind) (i : Nat) : Code α α :=
  let w := kind.width s
  let h := flatHead s w i
  let t := flatTime s w i
  let split := flatSplit s w i
  let c := flatChannel w i
  match kind with
  | .output => partialOCode s p b seed h t split c
  | .dq => partialQCode s p b seed h t split c
  | .dk => partialKCode s p b final h t split c
  | .dv => partialVCode s p b final h t split c
  | .finalState => stateCode s b seed (i/s.DV/s.DK) (i/s.DV%s.DK) (i%s.DV) s.T

@[simp] theorem eval_partialCellCode (a : Arithmetic α) (s : Shape) (p : Plans) (b : InputBuffers)
    (seed final : Bool) (kind : PartialKind) (i : Nat) (m : Memory α) :
    (partialCellCode s p b seed final kind i).eval a m=
      (partialValue a s p (b.observe s m) seed final kind i,m) := by
  cases kind <;> simp [partialCellCode,partialValue]

#axiomsClean eval_partialOCode
#axiomsClean eval_partialQCode
#axiomsClean eval_partialKCode
#axiomsClean eval_partialVCode
#axiomsClean eval_partialCellCode
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention
