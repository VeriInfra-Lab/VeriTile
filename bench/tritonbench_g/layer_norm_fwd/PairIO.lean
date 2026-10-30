import bench.tritonbench_g.layer_norm_fwd.OrdinaryIO
import bench.tritonbench_g.layer_norm_fwd.NativeIO
import bench.tritonbench_g.layer_norm_fwd.Proofs.Pair

namespace VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.GridPair
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open VeriTile.Bench.Repairs.LayerNormFwd
open OrdinaryEquivalence
noncomputable section
variable {α : Type} [Inhabited α]

def candidateInvocation (A : Algebra α) (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (row : Nat) : Invocation α :=
  { NativeInterface.invocation A f r stride M N eps p row with
    kernel := (Pair.indexCommuted f r stride M N eps p).kernel }

def original (A : Algebra α) (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) : Implementation (NativeInterface.io A f r stride M N eps p) :=
  Implementation.original _

def indexCommuted (A : Algebra α) (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) : Implementation (NativeInterface.io A f r stride M N eps p) where
  schedule := fun _ => List.ofFn (fun row : Fin M => candidateInvocation A f r stride M N eps p row.val)
  writes := fun _ row => Contract.written f r stride M N row.val
  observe := fun _ _ => id

end
end VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.GridPair
