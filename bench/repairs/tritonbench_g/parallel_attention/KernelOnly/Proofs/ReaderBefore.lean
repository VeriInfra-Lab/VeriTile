import bench.repairs.tritonbench_g.parallel_attention.Proofs.IOReal
import bench.repairs.tritonbench_g.parallel_attention.KernelOnly.Coverage

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel.ReaderBefore
open VeriTile.Triton FP.Structural
open scoped VeriTile.Triton.FP.Structural.ProgramIO
open VeriTile.Bench.Repairs.ParallelAttention
open IOData
open scoped BigOperators
noncomputable section

/-- Actual FP32 forward source or combined DQ/barrier/DKV source over the full
grid. Real is the mathematical interpretation of the source's floating cells.
DQ/DK are value-shard partial outputs. The Z mask retains the shard-zero guard;
no output allocation or host shard reduction is hidden inside this kernel IO. -/
abbrev attentionIO (direction : Direction) (shape : Shape) (scale : ℝ) : ProgramIO ℝ ℝ :=
  dataIO ℝ direction shape (realJobs direction shape scale)

/-- Logical score, causal square weight, and value-shard cotangent. DZ is
attached only to shard zero; DQ/DK remain partial gradients for that shard. -/
def squaredScore (shape : Shape) (scale : ℝ) (x : TensorInputs ℝ) (bh t j : Nat) : ℝ :=
  scale * ∑ e∈Finset.range shape.K, x.q bh t e*x.k bh j e

def causalWeight (shape : Shape) (scale : ℝ) (x : TensorInputs ℝ) (bh t j : Nat) : ℝ :=
  if j≤t then (squaredScore shape scale x bh t j)^2 else 0

def cotangent (shape : Shape) (x : TensorInputs ℝ) (bh shard t j : Nat) : ℝ :=
  (∑ p∈Finset.range shape.BV,
    if shard*shape.BV+p<shape.V then x.do_ bh t (shard*shape.BV+p)*x.v bh j (shard*shape.BV+p) else 0)+
    (if shard=0 then x.dz bh t else 0)

/-- Independent squared causal attention and all five actual output channels. -/
def mathematical (shape : Shape) (scale : ℝ) (xs : ChannelValues ℝ (inputPorts shape)) :
    ChannelValues ℝ (outputPorts shape) :=
  let x := tensorValues shape xs
  fun out=>match out with
  | ⟨0,_⟩ => fun j=> (show ℝ from ∑ n∈Finset.range shape.T,
      causalWeight shape scale x j.1.val (row j) n*x.v j.1.val n (col j))
  | ⟨1,_⟩ => fun j=> (show ℝ from ∑ n∈Finset.range shape.T,
      causalWeight shape scale x j.1.val (j.2.1.val*128+j.2.2.2.1.val) n)
  | ⟨2,_⟩ => fun j=> (show ℝ from scale*∑ n∈Finset.range shape.T,
      if n≤row j then 2*cotangent shape x j.1.val j.2.2.1.val (row j) n*
        squaredScore shape scale x j.1.val (row j) n*x.k j.1.val n j.2.2.2.2.1.val else 0)
  | ⟨3,_⟩ => fun j=> (show ℝ from ∑ t∈Finset.range shape.T,
      if t<shape.T ∧ row j≤t then 2*scale*cotangent shape x j.1.val j.2.2.1.val t (row j)*
        squaredScore shape scale x j.1.val t (row j)*x.q j.1.val t j.2.2.2.2.1.val else 0)
  | ⟨4,_⟩ => fun j=> (show ℝ from ∑ t∈Finset.range shape.T,
      if t<shape.T ∧ row j≤t then (squaredScore shape scale x j.1.val t (row j))^2*x.do_ j.1.val t (col j) else 0)


theorem forward_grid_real (shape : Shape) (accepted : ShapeContract shape) (scale : ℝ) :
    Spec.Real (attentionIO .forward shape scale ⊨ mathematical shape scale) :=
  IOData.real_forward shape accepted scale

theorem backward_grid_real (shape : Shape) (accepted : ShapeContract shape) (scale : ℝ) :
    Spec.Real (attentionIO .backward shape scale ⊨ mathematical shape scale) :=
  IOData.real_backward shape accepted scale

end
end VeriTile.Bench.Repairs.ParallelAttentionKernel.ReaderBefore
