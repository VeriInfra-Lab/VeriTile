import bench.repairs.tritonbench_g.fused_layernorm_triton.Proofs.PublicIO

namespace VeriTile.Bench.Repairs.FusedLayernormTriton.IOResults
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.FusedLayernormTriton FPStages Whole
noncomputable section
set_option maxHeartbeats 1000000

def realFormula (D : Nat) (xs : Nat → Nat → ℝ) (ws bs : Nat → ℝ) (row : Nat) : ℝ×ℝ×(Nat → ℝ) :=
  (KernelValues.mean xs D row,KernelValues.inverse xs D row,KernelValues.output xs ws bs D row)

theorem real (rest : Algebra ℝ) (X W Bias Y Mean Rstd : RegionName) (rows D : Nat)
    (positive : 0<D) (sep : Separation X W Bias Y Mean Rstd)
    (c : Contract.Configuration) (complete : c.tree.Complete) :
    (PublicIO.io (ideal rest) realEqual id X W Bias Y Mean Rstd rows D c).Implements
      (PublicIO.expected X W Bias Y Mean Rstd rows D (realFormula D)) := by
  apply PublicIO.implements
  intro initial xs ws bs hx hw hb
  obtain ⟨final,ran,means,inverses,outputs,frame⟩ := KernelValues.real rest initial X W Bias Y Mean Rstd
    rows D positive xs ws bs hx hw hb sep c complete
  refine ⟨final,ran,?_,frame⟩
  intro row hr
  refine ⟨?_,?_,?_⟩
  · rw [means row hr];rfl
  · rw [inverses row hr];rfl
  · intro col hc
    rw [outputs row hr col hc];rfl

def nativeFormula (I : Native.Instructions) (P : Native.Profile) (D : Nat) (c : Contract.Configuration)
    (xs : Nat → Nat → Native.Term) (ws bs : Nat → Native.Term) (row : Nat) :
    Native.Term×Native.Term×(Nat → Native.Term) :=
  let ref := Native.reference I P (xs row) ws bs D c.block c.tree
  (.bits ref.mean,.bits ref.rstd,fun col => .bits (ref.output col))

private theorem mapCell_comp {α β γ : Type} (f : α → β) (g : β → γ) (c : Cell α) :
    mapCell (g ∘ f) c=mapCell g (mapCell f c) := by
  cases c with
  | mk d v => cases d <;> rfl

theorem native (I : Native.Instructions) (P : Native.Profile)
    (X W Bias Y Mean Rstd : RegionName) (rows D : Nat)
    (sep : Separation X W Bias Y Mean Rstd) (c : Contract.Configuration) :
    (PublicIO.io Native.algebra (Native.equal I P) ((fun word => (.bits word : Native.Term)) ∘ Native.observe I P)
      X W Bias Y Mean Rstd rows D c).Implements
      (PublicIO.expected X W Bias Y Mean Rstd rows D (nativeFormula I P D c)) := by
  apply PublicIO.implements
  intro initial xs ws bs hx hw hb
  obtain ⟨final,ran,values,frame⟩ := KernelValues.native I P initial X W Bias Y Mean Rstd rows D
    xs ws bs hx hw hb sep c
  refine ⟨final,ran,?_,frame⟩
  intro row hr
  obtain ⟨hm,hrstd,hy⟩ := values row hr
  refine ⟨?_,?_,?_⟩
  · rw [mapCell_comp,show mapCell (Native.observe I P) (final.mem Mean row)=_ from hm];rfl
  · rw [mapCell_comp,show mapCell (Native.observe I P) (final.mem Rstd row)=_ from hrstd];rfl
  · intro col hc
    rw [mapCell_comp,show mapCell (Native.observe I P) (final.mem Y (col+D*row))=_ from hy col hc];rfl

#axiomsClean real
#axiomsClean native
end
end VeriTile.Bench.Repairs.FusedLayernormTriton.IOResults
