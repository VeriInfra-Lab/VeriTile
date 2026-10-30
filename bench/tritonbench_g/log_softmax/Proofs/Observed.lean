import bench.tritonbench_g.log_softmax.Proofs.KernelGrid

/-! Complete symbolic source executions remain equal under any fixed lowering
and instruction interpretation. Unlike a Word multiply/subtract interpreter,
this interface preserves expression structure until native observation. -/
namespace VeriTile.Bench.Repairs.LogSoftmax.Observed
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section

def run [Inhabited A] (direction : Direction) (algebra : Algebra A) (candidate : Bool)
    (M N K BM C : Nat) : List (Nat × Nat) → State A → Option (State A)
  | [],s => some s
  | p::rest,s => do
    let next ← FP.Structural.exec (engine algebra)
      (KernelGrid.program direction candidate M N K BM C) (KernelGrid.positioned s p.1 p.2)
    run direction algebra candidate M N K BM C rest next

/-- The same complete symbolic state witnesses both actual source variants.
No law about rounded arithmetic, reductions or contraction is needed. -/
theorem success_frame [Inhabited A] (direction : Direction) (algebra : Algebra A)
    (M N K BM C : Nat) (hc : 0<C) (order : List (Nat × Nat)) (s : State A) :
    ∃ final,run direction algebra Bool.false M N K BM C order s=some final ∧
      run direction algebra Bool.true M N K BM C order s=some final ∧
      (∀ (r : RegionName) i,r≠"output" ∨ M*N*K≤ i → final.mem r i=s.mem r i) := by
  induction order generalizing s with
  | nil => exact ⟨s,rfl,rfl,fun _ _ _ => rfl⟩
  | cons p rest ih =>
    have first : ∃ mid,
        FP.Structural.exec (engine algebra) (KernelGrid.program direction Bool.false M N K BM C)
          (KernelGrid.positioned s p.1 p.2)=some mid ∧
        FP.Structural.exec (engine algebra) (KernelGrid.program direction Bool.true M N K BM C)
          (KernelGrid.positioned s p.1 p.2)=some mid ∧
        (∀ (r : RegionName) i,r≠"output" ∨ M*N*K≤ i → mid.mem r i=s.mem r i) := by
      cases direction
      · obtain ⟨mid,ran⟩ := Kernel.forward_total (engine algebra) "output" "input" M N K BM C
          (KernelGrid.positioned s p.1 p.2) hc
        refine ⟨mid,ran,(Rewrite.forward_profiled algebra "output" "input" M N K BM C _).symm.trans ran,?_⟩
        intro r i outside
        exact Kernel.forward_frame (engine algebra) "output" "input" M N K BM C _ mid hc ran r i outside
      · obtain ⟨mid,ran⟩ := Kernel.backward_total (engine algebra) "input" "gradient" "output" M N K BM C
          (KernelGrid.positioned s p.1 p.2)
        refine ⟨mid,ran,(Rewrite.backward_profiled algebra "input" "gradient" "output" M N K BM C _).symm.trans ran,?_⟩
        intro r i outside
        exact Kernel.backward_frame (engine algebra) "input" "gradient" "output" M N K BM C _ mid ran r i outside
    obtain ⟨mid,ra,rb,frame⟩ := first
    obtain ⟨final,fa,fb,exterior⟩ := ih mid
    refine ⟨final,?_,?_,?_⟩
    · simp only [run,ra]; exact fa
    · simp only [run,rb]; exact fb
    · intro r i outside
      exact (exterior r i outside).trans (frame r i outside)

#axiomsClean success_frame
end
end VeriTile.Bench.Repairs.LogSoftmax.Observed
