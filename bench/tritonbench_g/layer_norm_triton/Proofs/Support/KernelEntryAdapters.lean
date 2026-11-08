import bench.tritonbench_g.layer_norm_triton.Kernels
namespace VeriTile.Bench.Repairs.LayerNormTriton.TypedSource
open VeriTile.Triton
noncomputable section
def forward (X Y W Bias Mean Rstd : RegionName) (stride N BLOCK_SIZE : Nat)
    (eps : ℝ) : ComputeKernel := forwardKernel X Y W Bias Mean Rstd stride N BLOCK_SIZE eps

def groupedBackwardKernel (DX DY DW DB X W Mean Rstd : RegionName)
    (stride M N GROUP_SIZE_M BLOCK_SIZE_N : Nat) : ComputeKernel := VeriTile.Bench.Repairs.LayerNormTriton.groupedBackwardKernel DX DY DW DB X W Mean Rstd stride M N GROUP_SIZE_M BLOCK_SIZE_N

def finalBackwardKernel (DW DB FINAL_DW FINAL_DB : RegionName)
    (M N BLOCK_SIZE_M BLOCK_SIZE_N : Nat) : ComputeKernel := VeriTile.Bench.Repairs.LayerNormTriton.finalBackwardKernel DW DB FINAL_DW FINAL_DB M N BLOCK_SIZE_M BLOCK_SIZE_N

def forwardCandidateKernel (X Y W Bias Mean Rstd : RegionName) (stride N BLOCK_SIZE : Nat)
    (eps : ℝ) : ComputeKernel := VeriTile.Bench.Repairs.LayerNormTriton.forwardCandidateKernel X Y W Bias Mean Rstd stride N BLOCK_SIZE eps

def groupedBackwardCandidateKernel (DX DY DW DB X W Mean Rstd : RegionName)
    (stride M N GROUP_SIZE_M BLOCK_SIZE_N : Nat) : ComputeKernel := VeriTile.Bench.Repairs.LayerNormTriton.groupedBackwardCandidateKernel DX DY DW DB X W Mean Rstd stride M N GROUP_SIZE_M BLOCK_SIZE_N

def finalBackwardCandidateKernel (DW DB FINAL_DW FINAL_DB : RegionName)
    (M N BLOCK_SIZE_M BLOCK_SIZE_N : Nat) : ComputeKernel := VeriTile.Bench.Repairs.LayerNormTriton.finalBackwardCandidateKernel DW DB FINAL_DW FINAL_DB M N BLOCK_SIZE_M BLOCK_SIZE_N

end
end VeriTile.Bench.Repairs.LayerNormTriton.TypedSource
