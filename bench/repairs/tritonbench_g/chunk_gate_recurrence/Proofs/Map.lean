import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.Whole
namespace VeriTile.Bench.Repairs.ChunkGateRecurrence
noncomputable section

structure OpsHom (f : α → β) (a : Ops α) (b : Ops β) : Prop where
  zero : f a.zero=b.zero
  add : ∀ x y, f (a.add x y)=b.add (f x) (f y)
  mul : ∀ x y, f (a.mul x y)=b.mul (f x) (f y)
  fma : ∀ x y z, f (a.fma x y z)=b.fma (f x) (f y) (f z)

def Inputs.map (f : α → β) (a : Inputs α) : Inputs β :=
  ⟨fun p t j => f (a.x p t j),fun h t => f (a.gate h t),
    fun p j => f (a.seed p j),fun p t j => f (a.incoming p t j)⟩

theorem inputsOf_map (f : α → β) (m : Memory α) : inputsOf (Memory.map f m)=(inputsOf m).map f := rfl

namespace Reduction
theorem eval_map (f : α → β) (a : Ops α) (b : Ops β) (hom : OpsHom f a b)
    (p : Plan n) (v : Fin n → α) : f (eval a p v)=eval b p (fun i => f (v i)) := by
  unfold eval
  induction p.tree with
  | zero => exact hom.zero
  | leaf i => rfl
  | add l r hl hr =>
    change f (a.add _ _)=b.add _ _
    rw [hom.add,hl,hr]

theorem product_map (f : α → β) (a : Ops α) (b : Ops β) (hom : OpsHom f a b)
    (p : ProductPlan n) (v : Fin n → α × α) :
    f (product a p v)=product b p (fun i => (f (v i).1,f (v i).2)) := by
  unfold product
  induction p.tree with
  | zero => exact hom.zero
  | mul i => exact hom.mul _ _
  | add l r hl hr =>
    change f (a.add _ _)=b.add _ _
    rw [hom.add,hl,hr]
  | fma i r hr =>
    change f (a.fma _ _ _)=b.fma _ _ _
    rw [hom.fma,hr]
end Reduction

namespace Reference
theorem forward_map (f : α → β) (aops : Ops α) (bops : Ops β) (hom : OpsHom f aops bops)
    (c : Shape) (a : Inputs α) (seed : Bool) (p t j : Nat) :
    f (forward aops c a seed p t j)=forward bops c (a.map f) seed p t j := by
  induction t with
  | zero => cases seed <;> simp [forward,Inputs.map,hom.zero]
  | succ t ih => simp only [forward,hom.fma,ih,Inputs.map]

theorem reverse_map (f : α → β) (aops : Ops α) (bops : Ops β) (hom : OpsHom f aops bops)
    (c : Shape) (a : Inputs α) (p t j : Nat) :
    f (reverse aops c a p t j)=reverse bops c (a.map f) p t j := by
  induction t with
  | zero => exact hom.zero
  | succ t ih => simp only [reverse,hom.fma,ih,Inputs.map]

theorem dx_map (f : α → β) (aops : Ops α) (bops : Ops β) (hom : OpsHom f aops bops)
    (c : Shape) (a : Inputs α) (p t j : Nat) :
    f (dx aops c a p t j)=dx bops c (a.map f) p t j := by
  unfold dx
  split_ifs
  · exact reverse_map f aops bops hom c a p _ j
  · exact hom.zero

theorem tileGate_map (f : α → β) (aops : Ops α) (bops : Ops β) (hom : OpsHom f aops bops)
    (c : Shape) (trees : Trees c) (a : Inputs α) (seed : Bool) (p t : Nat) :
    f (tileGate aops c trees a seed p t)=tileGate bops c trees (a.map f) seed p t := by
  unfold tileGate
  split_ifs
  · rw [Reduction.product_map f aops bops hom]
    simp only [forward_map f aops bops hom,reverse_map f aops bops hom]
  · exact hom.zero

theorem dgate_map (f : α → β) (aops : Ops α) (bops : Ops β) (hom : OpsHom f aops bops)
    (c : Shape) (trees : Trees c) (a : Inputs α) (seed : Bool) (p t : Nat) :
    f (dgate aops c trees a seed p t)=dgate bops c trees (a.map f) seed p t := by
  unfold dgate
  rw [Reduction.eval_map f aops bops hom]
  simp only [tileGate_map f aops bops hom]
end Reference

/-- Scalar homomorphisms transport the entire current-memory execution,
including both grids, saved output, scratch, tail zeroing and host sums. -/
theorem Source.run_map (f : α → β) (aops : Ops α) (bops : Ops β) (hom : OpsHom f aops bops)
    (c : Shape) (trees : Trees c) (seed : Bool) (m : Memory α) :
    Memory.map f (Source.run aops c trees seed m)=Source.run bops c trees seed (Memory.map f m) := by
  rw [Source.run_complete,Source.run_complete]
  funext r p t j
  simp only [Memory.map,Source.completeImage,inputsOf_map]
  split_ifs <;> simp only [Reference.forward_map f aops bops hom,
    Reference.reverse_map f aops bops hom,Reference.dx_map f aops bops hom,
    Reference.tileGate_map f aops bops hom,Reference.dgate_map f aops bops hom]

#axiomsClean Source.run_map
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence
