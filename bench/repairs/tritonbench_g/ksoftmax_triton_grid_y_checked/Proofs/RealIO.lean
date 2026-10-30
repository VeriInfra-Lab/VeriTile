import bench.repairs.tritonbench_g.ksoftmax_triton_grid_y_checked.Proofs.Interface

namespace VeriTile.Bench.Repairs.KsoftmaxGridYChecked.Interface
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.Ksoftmax
open VeriTile.Bench.TritonBenchG.KsoftmaxTriton
noncomputable section

theorem refine_program {A : Type} [Inhabited A] (M : Algebra A) (op : Operation)
    (B Q K D : Nat) (positive : 0<D) (covered : K≤D)
    (domain : Inputs A B Q D → Prop) (target : Inputs A B Q D → Outputs A B Q D)
    (formula : ∀ v, domain v → ∀ (b : Fin B) (r : Fin Q) (i : Fin D), i.val<K →
      orderedRow M op K r (left v b r) (right v b r) i =
        target v ⟨0,by simp [outputs]⟩ (b,r,i,PUnit.unit)) :
    (programIO M op B Q K D (selectedRows B Q K)).ImplementsOn domain target := by
  intro v hd bounds hb initial hin
  obtain ⟨safe, final, execution, values, frame⟩ :=
    program_implements M op B Q K D positive covered (selectedRows B Q K)
      (selectedRows_complete B Q K) v trivial bounds hb initial hin
  refine ⟨safe,final,execution,?_,frame⟩
  intro p j active
  have hp : p.val<1 := p.isLt
  have pz : p=⟨0,by simp [programIO,memory,outputs]⟩ := Fin.ext (show p.val=0 by omega)
  subst p
  have got := values ⟨0,by simp [programIO,memory,outputs]⟩ j active
  exact got.trans (congrArg (Cell.mk .real) (formula v hd j.1 j.2.1 j.2.2.1 active))

theorem forward_implements (B Q K D : Nat) (positive : 0<D) (covered : K≤D)
    (log causal : Bool) (mode : Forward.MaskMode) :
    (programIO Forward.TypedExecution.model (.forward log causal mode) B Q K D (selectedRows B Q K)).ImplementsOn
      (ForwardDomain K causal mode) (forwardFormula K log causal mode) := by
  apply refine_program _ _ B Q K D positive covered
  intro v hd b r i hi
  have h := KernelExtended.forward_value K r positive log causal mode
    (left v b r) (right v b r) (hd b r (by omega)) i
  exact congrArg (fun c : Cell (WithBot ℝ) => c.read .real) h

theorem some_getD (x : WithBot ℝ) (h : x.isSome) : x=some (x.getD 0) := by
  cases x with
  | bot => cases h
  | coe x => rfl

theorem backward_implements (B Q K D : Nat) (positive : 0<D) (covered : K≤D)
    (log causal : Bool) :
    (programIO Forward.TypedExecution.model (.backward log causal) B Q K D (selectedRows B Q K)).ImplementsOn
      (BackwardDomain K causal Bool.false) (backwardFormula K log causal) := by
  apply refine_program _ _ B Q K D positive covered
  intro v hd b r i _
  have h := backwardValue_real K r log causal (left v b r) (right v b r)
    (fun j => (left v b r j).getD 0) (fun j => (right v b r j).getD 0)
    (fun j hj => some_getD _ (hd b r j hj).1)
    (fun j hj => some_getD _ (by simpa using (hd b r j hj).2)) i
  exact congrArg (fun c : Cell (WithBot ℝ) => c.read .real) h

theorem log_backward_implements (B Q K D : Nat) (positive : 0<D) (covered : K≤D)
    (causal : Bool) :
    (programIO Forward.TypedExecution.model (.backward Bool.true causal) B Q K D (selectedRows B Q K)).ImplementsOn
      (BackwardDomain K causal Bool.true) (logBackwardFormula K causal) := by
  apply refine_program _ _ B Q K D positive covered
  intro v hd b r i _
  have h := KernelLogBackward.source_value K r causal (left v b r) (right v b r)
    (fun j => (left v b r j).getD 0) (fun j hj => some_getD _ (hd b r j hj).1) i
  exact congrArg (fun c : Cell (WithBot ℝ) => c.read .real) h

#axiomsClean forward_implements
#axiomsClean backward_implements
#axiomsClean log_backward_implements
end
end VeriTile.Bench.Repairs.KsoftmaxGridYChecked.Interface
