import bench.repairs.tritonbench_g.parallel_attention.KernelOnly.ForwardScalar

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.ParallelAttention
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000

theorem foldN_projection {β γ : Type} (f : Nat → β → β) (g : Nat → γ → γ)
    (observe : β → γ) (n : Nat) (initial : β)
    (step : ∀ j a, observe (f j a)=g j (observe a)) :
    observe (foldN f n initial)=foldN g n (observe initial) := by
  induction n with
  | zero => rfl
  | succ n ih => simp only [foldN,step,ih]

theorem forward_scalar_values {α : Type} [Inhabited α] (a : Scalars α)
    (shape : Shape) (scale : ℝ) (s : State α) (r : Fin 128) (v : Fin shape.BV) :
    let values := forwardValues (scalarAlgebra a) (scalarDotPrimitive a) (forwardConfig shape scale) s
    (a.cast (values.output ⟨r,v,()⟩),a.cast (values.normalizer ⟨r,()⟩)) =
      forwardLane (a.ops scale) sourcePlans shape (inputReaders shape s) (launchTile shape s) r.val v.val := by
  let c := forwardConfig shape scale
  let tile := launchTile shape s
  let x := inputReaders shape s
  let observe := fun (z : ForwardAccumulator α 128 shape.BV) => (z.output ⟨r,v,()⟩,z.normalizer ⟨r,()⟩)
  let update := forwardUpdate (scalarAlgebra a) (scalarDotPrimitive a) c s
  let step := forwardStep (a.ops scale) sourcePlans shape x tile r.val v.val
  have hprefix (initial : ForwardAccumulator α 128 shape.BV) :
      observe (foldRange (update Bool.false) 0 (tile.chunk*128) 32 initial) =
        foldN (step Bool.false) (tile.chunk*4) (observe initial) := by
    have bound : tile.chunk*128 = 0+(tile.chunk*4)*32 := by omega
    rw [bound,foldRange_count _ _ _ _ (by omega)]
    apply foldN_projection
    intro j z
    simpa [observe,update,step,c,tile,x] using forward_scalar_step a shape scale s Bool.false j r v z
  have diagonal (initial : ForwardAccumulator α 128 shape.BV) :
      observe (foldRange (update Bool.true) (tile.chunk*128) ((tile.chunk+1)*128) 32 initial) =
        foldN (step Bool.true) 4 (observe initial) := by
    have bound : (tile.chunk+1)*128 = tile.chunk*128+4*32 := by omega
    rw [bound,foldRange_count _ _ _ _ (by omega)]
    apply foldN_projection
    intro j z
    exact forward_scalar_step a shape scale s Bool.true j r v z
  let initial : ForwardAccumulator α 128 shape.BV := ⟨fun _ => a.literal 0,fun _ => a.literal 0⟩
  have whole := diagonal (foldRange (update Bool.false) 0 (tile.chunk*128) 32 initial)
  rw [hprefix] at whole
  exact congrArg (fun z : α×α => (a.cast z.1,a.cast z.2)) whole

#axiomsClean foldN_projection
#axiomsClean forward_scalar_values
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel
