import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.IOImplements
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.KernelRealResult
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.KernelNativeResult

namespace VeriTile.Bench.Repairs.LayerNormWelfold.IOResults
open VeriTile.Triton FP.Structural Reuse Samples
noncomputable section

def realFormula (D : Nat) (x w bias : Nat → ℝ) (row : Nat) : ℝ × ℝ × (Nat → ℝ) :=
  (Math.mean x D row,Math.inverse x D row,Math.output x w bias D row)

theorem real (rest : Algebra ℝ) (Y Mean Rstd : RegionName) (S D : Nat)
    (c : Configuration) (complete : c.tree.Complete)
    (sep : Separation "X" "W" "Bias" Y Mean Rstd) :
    (IO.layerNormIO (ideal rest) id Y Mean Rstd S D c.block c.tree).Implements
      (IO.result Y Mean Rstd S D (realFormula D)) := by
  apply IOProof.implements _ _ _ _ _ _ _ _ _ (sep.inputs "X" (Or.inl rfl)).1
  intro initial x w bias hx hw hb
  obtain ⟨final,ran,hm,hi,hy,frame⟩ := KernelRealResult.kernel_real rest initial x w bias
    S D Y Mean Rstd c complete sep hx hw hb
  refine ⟨final,ran,?_,?_,?_,frame⟩
  · intro row hr
    rw [hm row hr]
    rfl
  · intro row hr
    rw [hi row hr]
    rfl
  · intro row hr col hc
    rw [hy row hr col hc]
    rfl

def graphFormula (D : Nat) (c : Configuration) (x w bias : Nat → NativeKernel.Term)
    (row : Nat) : NativeKernel.Term × NativeKernel.Term × (Nat → NativeKernel.Term) :=
  let inputs := KernelGraph.logicalInputs NativeKernel.algebra x w bias D row
  (Native.eval NativeKernel.constructors inputs (Native.meanExpr D c.block c.tree),
   Native.eval NativeKernel.constructors inputs (Native.inverseExpr D c.block c.tree),
   fun col => Native.eval NativeKernel.constructors inputs (Native.outputExpr D c.block col c.tree))

theorem graph (Y Mean Rstd : RegionName) (S D : Nat) (c : Configuration)
    (sep : Separation "X" "W" "Bias" Y Mean Rstd) :
    (IO.layerNormIO NativeKernel.algebra id Y Mean Rstd S D c.block c.tree).Implements
      (IO.result Y Mean Rstd S D (graphFormula D c)) := by
  apply IOProof.implements _ _ _ _ _ _ _ _ _ (sep.inputs "X" (Or.inl rfl)).1
  intro initial x w bias hx hw hb
  obtain ⟨final,ran,hm,hi,hy,frame⟩ := KernelNativeResult.kernel_graph initial x w bias
    S D Y Mean Rstd c sep hx hw hb
  refine ⟨final,ran,?_,?_,?_,frame⟩
  · intro row hr
    rw [hm row hr]
    rfl
  · intro row hr
    rw [hi row hr]
    rfl
  · intro row hr col hc
    rw [hy row hr col hc]
    rfl

/-- Injective Word embedding retains arbitrary input graph terms while the
native output observation is exactly the old Word-valued result. -/
def observed (I : NativeKernel.Instructions) (P : NativeKernel.Profile)
    (term : NativeKernel.Term) : NativeKernel.Term := .bits (NativeKernel.observe I P term)

def nativeFormula (I : NativeKernel.Instructions) (P : NativeKernel.Profile)
    (D : Nat) (c : Configuration) (x w bias : Nat → NativeKernel.Term) (row : Nat) :
    NativeKernel.Term × NativeKernel.Term × (Nat → NativeKernel.Term) :=
  let ref := NativeKernel.reference I P x w bias D c.block row c.tree
  (.bits ref.mean,.bits ref.inverse,fun col => .bits (ref.output col))

private theorem observed_cell (I : NativeKernel.Instructions) (P : NativeKernel.Profile)
    (cell : Cell NativeKernel.Term) :
    mapCell (observed I P) cell =
      mapCell (fun word => (.bits word : NativeKernel.Term)) (NativeKernel.observeCell I P cell) := by
  cases cell with
  | mk d value => cases d <;> rfl

theorem native (I : NativeKernel.Instructions) (P : NativeKernel.Profile)
    (Y Mean Rstd : RegionName) (S D : Nat) (c : Configuration)
    (sep : Separation "X" "W" "Bias" Y Mean Rstd) :
    (IO.layerNormIO NativeKernel.algebra (observed I P) Y Mean Rstd S D c.block c.tree).Implements
      (IO.result Y Mean Rstd S D (nativeFormula I P D c)) := by
  apply IOProof.implements _ _ _ _ _ _ _ _ _ (sep.inputs "X" (Or.inl rfl)).1
  intro initial x w bias hx hw hb
  obtain ⟨final,ran,hm,hi,hy,frame,_⟩ := KernelNativeResult.kernel_native I P initial x w bias
    S D Y Mean Rstd c sep hx hw hb
  refine ⟨final,ran,?_,?_,?_,frame⟩
  · intro row hr
    rw [observed_cell,hm row hr]
    rfl
  · intro row hr
    rw [observed_cell,hi row hr]
    rfl
  · intro row hr col hc
    rw [observed_cell,hy row hr col hc]
    rfl

#axiomsClean real
#axiomsClean graph
#axiomsClean native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormWelfold.IOResults
