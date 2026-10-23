import bench.repairs.tritonbench_g.parallel_attention.Proofs.IOInputBinding

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel.IOData
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.ParallelAttention
noncomputable section
variable {α : Type} [Inhabited α]

theorem forward_bound (M : Algebra α) (D : DotPrimitive α) (direction : Direction)
    (shape : Shape) (scale : ℝ) (xs : ChannelValues α (inputPorts shape)) (s : State α)
    (p : Program shape 128)
    (h : InputsAgree direction shape (fun r off=>(s.mem r off).read .real) (inputValues shape xs)) :
    forwardValues M D (forwardConfig shape scale) (withProgram p s)=
      IOLogical.forwardValues M D (forwardConfig shape scale) (pid p) (inputValues shape xs) := by
  rw [IOLogical.forward_projection]
  exact IOLogical.forward_congr M D _ _ _ _ (forward_inputs direction shape scale p _ _ h)

theorem query_bound (M : Algebra α) (D : DotPrimitive α)
    (shape : Shape) (scale : ℝ) (xs : ChannelValues α (inputPorts shape)) (s : State α)
    (p : Program shape 64)
    (h : InputsAgree .backward shape (fun r off=>(s.mem r off).read .real) (inputValues shape xs)) :
    queryValues M D (dqConfig shape scale p.tile) s=
      IOLogical.queryValues M D (dqConfig shape scale p.tile) (inputValues shape xs) := by
  rw [IOLogical.query_projection]
  exact IOLogical.query_congr M D _ _ _ (query_inputs shape scale p _ _ h)

theorem keyValue_bound (M : Algebra α) (D : DotPrimitive α)
    (shape : Shape) (scale : ℝ) (xs : ChannelValues α (inputPorts shape)) (s : State α)
    (p : Program shape 64)
    (h : InputsAgree .backward shape (fun r off=>(s.mem r off).read .real) (inputValues shape xs)) :
    backwardValues M D (backwardConfig shape scale p.tile) s=
      IOLogical.backwardValues M D (backwardConfig shape scale p.tile) (inputValues shape xs) := by
  rw [IOLogical.backward_projection]
  exact IOLogical.backward_congr M D _ _ _ (backward_inputs shape scale p _ _ h)

#axiomsClean forward_bound
#axiomsClean query_bound
#axiomsClean keyValue_bound
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel.IOData
