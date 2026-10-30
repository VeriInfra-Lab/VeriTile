import bench.repairs.tritonbench_g.fused_layernorm_triton.Kernels
import VeriTile.Triton.Source

/-! Proof-only projections of the complete readable kernels. No projected list
is an authoritative source definition. Every context equation is checked by
reduction against the full quotation. -/
namespace VeriTile.Bench.Repairs.FusedLayernormTriton.SourcePieces
open VeriTile.Triton
noncomputable section
set_option maxRecDepth 30000

def source (hoist : Bool) (X W Bias Y Mean Rstd : RegionName) (S D B : Nat)
    (tree : Kernels.ReductionTree B) : ComputeKernel :=
  if hoist then Kernels.hoisted X W Bias Y Mean Rstd S D B tree
  else Kernels.kernel X W Bias Y Mean Rstd S D B tree

def statistics (hoist : Bool) (X W Bias Y Mean Rstd : RegionName) (S D B : Nat)
    (tree : Kernels.ReductionTree B) : Source.DynamicRange :=
  (Source.dynamicRange? (source hoist X W Bias Y Mean Rstd S D B tree).surfaceBody "roffset").getD
    ⟨[],.constNat 0,.constNat 0,.constNat 1,[],[]⟩

def normalization (hoist : Bool) (X W Bias Y Mean Rstd : RegionName) (S D B : Nat)
    (tree : Kernels.ReductionTree B) : Source.DynamicRange :=
  (Source.dynamicRange? (statistics hoist X W Bias Y Mean Rstd S D B tree).after "roffset").getD
    ⟨[],.constNat 0,.constNat 0,.constNat 1,[],[]⟩

theorem complete (hoist : Bool) (X W Bias Y Mean Rstd : RegionName) (S D B : Nat)
    (tree : Kernels.ReductionTree B) :
    (source hoist X W Bias Y Mean Rstd S D B tree).surfaceBody =
      (statistics hoist X W Bias Y Mean Rstd S D B tree).before ++
      [.forRange "roffset" B D B (statistics hoist X W Bias Y Mean Rstd S D B tree).body] ++
      (normalization hoist X W Bias Y Mean Rstd S D B tree).before ++
      [.forRange "roffset" 0 D B (normalization hoist X W Bias Y Mean Rstd S D B tree).body] := by
  cases hoist <;> rfl

theorem common_prefix (X W Bias Y Mean Rstd : RegionName) (S D B : Nat)
    (tree : Kernels.ReductionTree B) :
    (statistics Bool.true X W Bias Y Mean Rstd S D B tree).before =
      (statistics Bool.false X W Bias Y Mean Rstd S D B tree).before := by rfl

theorem common_body (X W Bias Y Mean Rstd : RegionName) (S D B : Nat)
    (tree : Kernels.ReductionTree B) :
    (statistics Bool.true X W Bias Y Mean Rstd S D B tree).body =
      (statistics Bool.false X W Bias Y Mean Rstd S D B tree).body := by rfl

def projections (X W Bias Y Mean Rstd : RegionName) (S D B : Nat)
    (tree : Kernels.ReductionTree B) : List ComputeStmt :=
  (normalization Bool.false X W Bias Y Mean Rstd S D B tree).body.take 3

theorem normalization_body (X W Bias Y Mean Rstd : RegionName) (S D B : Nat)
    (tree : Kernels.ReductionTree B) :
    (normalization Bool.false X W Bias Y Mean Rstd S D B tree).body =
      projections X W Bias Y Mean Rstd S D B tree ++
      (normalization Bool.true X W Bias Y Mean Rstd S D B tree).body := by rfl

theorem hoisted_before (X W Bias Y Mean Rstd : RegionName) (S D B : Nat)
    (tree : Kernels.ReductionTree B) :
    (normalization Bool.true X W Bias Y Mean Rstd S D B tree).before =
      (normalization Bool.false X W Bias Y Mean Rstd S D B tree).before ++
      projections X W Bias Y Mean Rstd S D B tree := by rfl

end
end VeriTile.Bench.Repairs.FusedLayernormTriton.SourcePieces
