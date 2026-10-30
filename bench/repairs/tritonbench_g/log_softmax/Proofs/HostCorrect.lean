import bench.repairs.tritonbench_g.log_softmax.Proofs.Host
import bench.repairs.tritonbench_g.log_softmax.Proofs.Real

namespace VeriTile.Bench.Repairs.LogSoftmax.HostCorrect
open VeriTile.Torch Address Execution Host
open VeriTile.Bench.Repairs.LayernormFwdTriton.Memory
noncomputable section
open Classical

theorem row_binding (api : API (WithBot ℝ) Store Byte) (a : Arguments) (h : ShapeContract a)
    (buffer : Dense (a.M*a.N*a.K)) (s : Store) (input : Nat → ℝ)
    (binding : ∀ i,i<a.input.numel → api.read s (buffer.address i)=some (input i))
    (q : Coord a.M a.N a.K) (j : Nat) (hj : j<a.N) :
    rowRead api buffer s q j=some (input (offset a.N a.K q.1.val j q.2.2.val)) := by
  rw [rowRead,safe_active hj]
  apply binding
  rw [←input_numel a h]
  exact VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealAddress.non_inner_bound q.1.isLt hj q.2.2.isLt

/-- All forward values, exact return metadata and saved-output identity,
including empty outputs and every original autotuning probe. The original
contiguous call's logical input binding is an ordinary Torch boundary. -/
theorem log_softmax_real (api : API (WithBot ℝ) Store Byte) (a : Arguments) (h : InputContract a)
    (b : Buffers a.M a.N a.K) (probes : List (Grid.Selected a.M a.N a.K))
    (selected : Grid.Selected a.M a.N a.K) (s : Store) (input : Nat → ℝ)
    (binding : ∀ i,i<a.input.numel → api.read s (b.first.address i)=some (input i)) :
    ∃ result,forwardCall api RealProof.ops a b probes selected s=.ok result ∧
      result.shape=a.input.shape ∧ result.dtype=.floating .fp32 ∧ result.device=a.input.device ∧
      result.output=b.output ∧ result.context.saved=result.output ∧ result.context.dim=a.axis ∧
      (∀ q : Coord a.M a.N a.K,api.read result.memory (result.output.address (linear q))=
        some (RealProof.targetForward
          (fun j : Fin a.N => input (offset a.N a.K q.1.val j.val q.2.2.val)) q.2.1)) ∧
      (∀ i,¬b.output.contains i → api.byte result.memory i=api.byte s i) := by
  refine ⟨⟨dispatch api RealProof.ops .forward b probes selected s,a.input.shape,.floating .fp32,
    a.input.device,b.output,⟨b.output,a.axis⟩⟩,forward_success api RealProof.ops a h b probes selected s,
    rfl,rfl,rfl,rfl,rfl,rfl,?_,?_⟩
  · intro q
    rw [dispatch_values]
    change forward RealProof.ops selected.plan _ q.2.1.val=_
    rw [forward_congr_active RealProof.ops selected.plan _
      (fun j => some (input (offset a.N a.K q.1.val j q.2.2.val)))
      (row_binding api a h.2.2 b.first s input binding q) _ q.2.1.isLt]
    exact RealProof.forward_target selected.plan _ q.2.1
  · exact dispatch_frame api RealProof.ops .forward b probes selected s

/-- All backward values read the saved forward output and original contiguous
upstream gradient; the returned tuple is (FP32 gradient, None, None). Both
inputs and every other byte outside the new gradient are framed. -/
theorem log_softmax_backward_real (api : API (WithBot ℝ) Store Byte)
    (a : BackwardArguments) (h : BackwardContract a) (ctx : Context a.saved)
    (b : Buffers a.saved.M a.saved.N a.saved.K) (saved : ctx.saved=b.first) (dimension : ctx.dim=a.saved.axis)
    (probes : List (Grid.Selected a.saved.M a.saved.N a.saved.K))
    (selected : Grid.Selected a.saved.M a.saved.N a.saved.K) (s : Store) (y g : Nat → ℝ)
    (bindingY : ∀ i,i<a.saved.input.numel → api.read s (ctx.saved.address i)=some (y i))
    (bindingG : ∀ i,i<a.saved.input.numel → api.read s (b.second.address i)=some (g i)) :
    ∃ result,backwardCall api RealProof.ops a ctx b saved dimension probes selected s=.ok result ∧
      result.shape=a.saved.input.shape ∧ result.dtype=.floating .fp32 ∧ result.device=a.saved.input.device ∧
      result.output=b.output ∧ result.dimGradient=none ∧ result.dtypeGradient=none ∧
      (∀ q : Coord a.saved.M a.saved.N a.saved.K,api.read result.memory (result.output.address (linear q))=
        some (RealProof.targetBackward a.saved.N
          (fun j => y (offset a.saved.N a.saved.K q.1.val j q.2.2.val))
          (fun j => g (offset a.saved.N a.saved.K q.1.val j q.2.2.val)) q.2.1.val)) ∧
      (∀ i,¬b.output.contains i → api.byte result.memory i=api.byte s i) := by
  refine ⟨⟨dispatch api RealProof.ops .backward b probes selected s,a.saved.input.shape,.floating .fp32,
    a.saved.input.device,b.output,none,none⟩,
    backward_success api RealProof.ops a h ctx b saved dimension probes selected s,
    rfl,rfl,rfl,rfl,rfl,rfl,?_,?_⟩
  · intro q
    rw [dispatch_values]
    change backward RealProof.ops selected.plan _ _ q.2.1.val=_
    have pinsY : ∀ i,i<a.saved.input.numel → api.read s (b.first.address i)=some (y i) := by
      simpa only [saved] using bindingY
    rw [backward_congr_active RealProof.ops selected.plan _ _
      (fun j => some (y (offset a.saved.N a.saved.K q.1.val j q.2.2.val)))
      (fun j => some (g (offset a.saved.N a.saved.K q.1.val j q.2.2.val)))
      (row_binding api a.saved h.2.2.2.2 b.first s y pinsY q)
      (row_binding api a.saved h.2.2.2.2 b.second s g bindingG q) _ q.2.1.isLt]
    exact RealProof.backward_target selected.plan _ _ _
  · exact dispatch_frame api RealProof.ops .backward b probes selected s

#axiomsClean log_softmax_real
#axiomsClean log_softmax_backward_real
end
end VeriTile.Bench.Repairs.LogSoftmax.HostCorrect
