import bench.repairs.tritonbench_g.fused_rwkv6_kernel_scalar_abi_checked.Proofs.Observations

namespace VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.RawCorrect
open VeriTile Triton FusedRwkv6Kernel
noncomputable section
attribute [local instance] Classical.propDecidable
set_option maxHeartbeats 8000000

/-- Independent exact mathematical partial, before the wrapper's Torch sum. -/
theorem real_partial (c : Shape) (m : Memory ℝ) (scale : ℝ) (z ik t f : Nat)
    (hz : z<c.Z) (ht : t<c.T) (hf : f<c.V) :
    target realOps c (sequentialPlans c) (inputs c m) scale .o (partialAddress c z ik t f)=
      Real.partialFormula c (inputs c m) scale z ik (visit c t) f := by
  have d:=partial_decode c z ik t f hz ht hf
  simp only [target,d.1,d.2.1,d.2.2.1,d.2.2.2,scalarPartial,sequentialPlans]
  change (Tree.chain c.BK).eval 0 (·+·) _=_
  rw [tree_real _ _ (Tree.chain_legal c.BK),←Fin.sum_univ_eq_sum_range]
  exact Real.partial_scalar c (inputs c m) scale z ik (visit c t) f hf

/-- Complete mathematical kernel execution on raw region memory, with the
independent partial/final-state recurrences and the whole output frame. -/
theorem rwkv_real (c : Shape) (hc : Sizes c) (scale : ℝ) (m : Memory ℝ) :
    ∃ out, Source.launch Real.algebra c scale (Source.cells m) = some out ∧
      (∀ z<c.Z, ∀ ik<c.NK, ∀ t<c.T, ∀ f<c.V,
        Source.words out .o (partialAddress c z ik t f) =
          Real.partialFormula c (inputs c m) scale z ik (visit c t) f) ∧
      (c.final=Bool.true → ∀ z<c.Z, ∀ e<c.K, ∀ f<c.V,
        Source.words out .ht (stateAddress c z e f) = realState c (inputs c m) z e f c.T) ∧
      (∀ r b, ¬Output c r b → Source.words out r b = m r b) := by
  obtain ⟨out,ho,hw⟩ := Source.launch_memory Real.algebra c hc (sequentialPlans c)
    (Source.real_rows c) scale (Source.cells m)
  simp only [Source.words_cells,Real.source_ops] at hw
  refine ⟨out,ho,?_,?_,?_⟩
  · intro z hz ik hik t ht f hf
    rw [hw,if_pos (partial_output c z ik t f hz hik ht hf)]
    exact real_partial c m scale z ik t f hz ht hf
  · intro hs z hz e he f hf
    rw [hw,if_pos (terminal_output c z e f hs hz he hf)]
    have d := state_decode c z e f he hf
    simp only [target,d.1,d.2.1,d.2.2]
    exact scalar_real_state c (inputs c m) z e f c.T he hf
  · intro r b outside
    rw [hw,if_neg outside]

#axiomsClean rwkv_real
#specNonCircular Real.partialFormula avoiding [Source.launch,Source.runGrid,FP.Structural.exec,runLoop,runProgram]
#specNonCircular realState avoiding [Source.launch,Source.runGrid,FP.Structural.exec,runLoop,runProgram]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.RawCorrect
