import bench.tritonbench_g.fused_layernorm_triton.Proofs.LogicalCoordinates

namespace VeriTile.Bench.Repairs.FusedLayernormTriton.Correctness
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.FusedLayernormTriton FPStages Whole
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- Complete selected row grid: FP32 X/weight/bias inputs, FP32 statistics
and BF16 Y storage. The erased .real channel retains the source's explicit
FP32 operations; this interpretation uses Real arithmetic and ideal casts.
All logical columns are active; source chunk tails are masked. Inputs may
alias each other; the public theorem requires the three outputs to be fresh. -/
def realIO (rest : Algebra ℝ) (X W Bias Y Mean Rstd : RegionName)
    (rows D : Nat) (c : Contract.Configuration) : LogicalProgramIO ℝ ℝ (PublicIO.TensorInputs ℝ rows D) (PublicIO.TensorOutputs ℝ rows D) :=
  PublicIO.logicalIO rest X W Bias Y Mean Rstd rows D c

/-- Independent mean, reciprocal standard deviation and affine normalization.
The epsilon is the selected source's fixed 1e-5. -/
def normalization (rest : Algebra ℝ) (X W Bias Y Mean Rstd : RegionName)
    (rows D : Nat) (c : Contract.Configuration) :
    PublicIO.TensorInputs ℝ rows D → PublicIO.TensorOutputs ℝ rows D :=
  PublicIO.logicalExpected rows D (fun xs ws bs row =>
    let mean := ((List.range D).map (xs row)).sum / D
    let variance := ((List.range D).map (fun col => (xs row col-mean)^2)).sum / D
    let inverse := 1 / Real.sqrt (variance + (1e-5 : ℝ))
    (mean,inverse,fun col => (xs row col-mean)*inverse*ws col+bs col))

/-- Every source access is backed; the full grid succeeds and produces all
three logical outputs, preserving every cell outside their declared windows. -/
specification kernel_real (rest : Algebra ℝ) (X W Bias Y Mean Rstd : RegionName)
    (rows D : Nat) (positive : 0<D) (sep : Separation X W Bias Y Mean Rstd)
    (c : Contract.Configuration) (complete : c.tree.Complete) :
    Spec.Real (realIO rest X W Bias Y Mean Rstd rows D c ⊨
      normalization rest X W Bias Y Mean Rstd rows D c) :=
  PublicIO.logical_real rest X W Bias Y Mean Rstd rows D positive sep c complete

#specNonCircular normalization avoiding [FP.Structural.exec,FP.Structural.runSchedule,
  Whole.rowValue,Whole.rowOutput,Contract.source,KernelProof.launch]
#axiomsClean kernel_real
#stmtSurfaceSubset kernel_real ⊆ [Spec.Real,realIO,normalization,LogicalProgramIO.Implements,
  FP.Structural.Algebra,RegionName,Separation,Contract.Configuration,Contract.Configuration.tree,Contract.Configuration.block,
  Kernels.ReductionTree.Complete,PublicIO.TensorInputs,PublicIO.TensorOutputs]
end
end VeriTile.Bench.Repairs.FusedLayernormTriton.Correctness
