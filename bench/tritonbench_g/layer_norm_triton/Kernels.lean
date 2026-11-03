import bench.tritonbench_g.layer_norm_triton.Sources.Body
namespace VeriTile.Bench.Repairs.LayerNormTriton.TypedSource
open VeriTile.Triton
noncomputable section
def forward (X Y W Bias Mean Rstd : RegionName) (stride N BLOCK_SIZE : Nat)
    (eps : ℝ) : ComputeKernel := Bodies.forward X Y W Bias Mean Rstd stride N BLOCK_SIZE eps

def groupedBackward (DX DY DW DB X W Mean Rstd : RegionName)
    (stride M N GROUP_SIZE_M BLOCK_SIZE_N : Nat) : ComputeKernel := Bodies.groupedBackward DX DY DW DB X W Mean Rstd stride M N GROUP_SIZE_M BLOCK_SIZE_N

def finalBackward (DW DB FINAL_DW FINAL_DB : RegionName)
    (M N BLOCK_SIZE_M BLOCK_SIZE_N : Nat) : ComputeKernel := Bodies.finalBackward DW DB FINAL_DW FINAL_DB M N BLOCK_SIZE_M BLOCK_SIZE_N

def forwardCandidate (X Y W Bias Mean Rstd : RegionName) (stride N BLOCK_SIZE : Nat)
    (eps : ℝ) : ComputeKernel := Bodies.forwardCandidate X Y W Bias Mean Rstd stride N BLOCK_SIZE eps

def groupedBackwardCandidate (DX DY DW DB X W Mean Rstd : RegionName)
    (stride M N GROUP_SIZE_M BLOCK_SIZE_N : Nat) : ComputeKernel := Bodies.groupedBackwardCandidate DX DY DW DB X W Mean Rstd stride M N GROUP_SIZE_M BLOCK_SIZE_N

def finalBackwardCandidate (DW DB FINAL_DW FINAL_DB : RegionName)
    (M N BLOCK_SIZE_M BLOCK_SIZE_N : Nat) : ComputeKernel := Bodies.finalBackwardCandidate DW DB FINAL_DW FINAL_DB M N BLOCK_SIZE_M BLOCK_SIZE_N

end
end VeriTile.Bench.Repairs.LayerNormTriton.TypedSource
