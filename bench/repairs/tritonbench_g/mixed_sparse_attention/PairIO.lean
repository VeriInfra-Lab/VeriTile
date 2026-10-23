import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.FusedMask

/-! Candidate source selection reuses the canonical sparse-array IO. -/
namespace VeriTile.Bench.Repairs.MixedSparseAttention.FusedMask
open VeriTile Triton FP.Structural
open Whole002 Source Whole Checked
noncomputable section
variable {α : Type} [Inhabited α]

def invocation (M : Algebra α) (D : Dot α) (c : Config) (p : Program) : Invocation α :=
  { KernelIO.invocation M D c p with kernel:=kernel c }

def variant (M : Algebra α) (D : Dot α) (c : Config) (work : List Program) :
    ArrayProgramIO.Variant (KernelIO.io M D c work) where
  schedule:=fun _=>work.map (invocation M D c)
  observe:=fun _ _=>id

end
end VeriTile.Bench.Repairs.MixedSparseAttention.FusedMask
