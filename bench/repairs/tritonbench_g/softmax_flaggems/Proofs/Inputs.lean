import bench.repairs.tritonbench_g.softmax_flaggems.Proofs.NativeHost
namespace VeriTile.Bench.Repairs.SoftmaxFlaggems.Inputs
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems HostGrid
noncomputable section

/-- Binding a direct input leaf is an identity update. There is no input copy. -/
theorem input_binding_identity (memory : RegionName → Nat → Native.Word) (X : RegionName) (count : Nat) :
    FPHost.writeSlice (Native.entry memory) X count (Native.input X (memory X))=Native.entry memory := by
  unfold FPHost.writeSlice Native.entry Native.input
  congr 1
  funext r o
  split_ifs with h
  · rcases h with ⟨rfl,_⟩;rfl
  · rfl

/-- If input and upstream-gradient leaves are bound to their entry bytes, the
whole call preserves every logical cell outside its two fresh output regions.
The wider historical host frame merely allowed initial binding writes. -/
theorem reference_frame (R : Native.Reductions) (shape : List Nat) (ax : Fin shape.length)
    (device sms : Nat) (waves : WavesAboveOne) (layouts : LayoutPrimitives) (strides : List Nat)
    (trials : List (Fin 6)) (choice : Fin 6) (memory : RegionName → Nat → Native.Word)
    (emptyP emptyY : Nat → Native.Word) (region : RegionName) (o : Nat)
    (outside : ¬((region="out" ∨ region="in_grad") ∧ o<shape.prod)) :
    (FPHost.reference (Native.algebra R) shape ax device sms waves layouts strides trials choice
      "x" "out" "grad" "in_grad" (Native.entry memory)
      (Native.input "x" (memory "x")) (Native.input "out" emptyP)
      (Native.input "grad" (memory "grad")) (Native.input "in_grad" emptyY)).finalState.mem region o=
      (Native.entry memory).mem region o := by
  by_cases hz : shape.prod=0
  · simp [FPHost.reference,hz,FPHost.pack]
  · obtain ⟨hm,hn,hk,count⟩ := FPHost.dimensions_positive shape ax hz
    have hp : ¬(region="out" ∧ o<shape.prod) := by tauto
    have hy : ¬(region="in_grad" ∧ o<shape.prod) := by tauto
    simp only [FPHost.reference,hz,if_false,FPHost.pack]
    rw [FPTuning.frame (Native.algebra R) "out" "grad" "in_grad" (dimensions shape ax) hn hk _ _ region o
      (by simpa [←count] using hy)]
    unfold FPHost.prepareBackward
    simp only [show ("grad":RegionName)≠"out" by decide,if_false,FPHost.writeSlice,if_neg hy]
    split_ifs with isGrad
    · rcases isGrad with ⟨rfl,_⟩
      rfl
    · rw [FPTrace.forward_frame (Native.algebra R) "x" "out" (dimensions shape ax)
        (forwardConfig waves (dimensions shape ax) sms) hn hk
        (forwardConfig_positive waves _ sms).1 (forwardConfig_positive waves _ sms).2
        (forwardConfig_one waves _ sms) _ region o (by simpa [←count] using hp)]
      unfold FPHost.prepareForward
      rw [input_binding_identity]
      simp [FPHost.writeSlice,hp]

#axiomsClean input_binding_identity
#axiomsClean reference_frame
end
end VeriTile.Bench.Repairs.SoftmaxFlaggems.Inputs
