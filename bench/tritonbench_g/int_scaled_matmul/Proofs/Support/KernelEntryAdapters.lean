import bench.tritonbench_g.int_scaled_matmul.Kernels
import bench.tritonbench_g.int_scaled_matmul.Proofs.Sources.Parameters

/-! Typed adapters to the exact copied repaired Python bodies. -/
namespace VeriTile.Bench.Repairs.IntScaledMatmul.Kernels
open VeriTile.Triton
open Revision001
noncomputable section
def unscaledSource
    (a_ptr b_ptr : Region .int) (c_ptr : RegionName)
    (M N K : Nat)
    (stride_am stride_ak stride_bk stride_bn stride_cm stride_cn : Nat)
    (BLOCK_M BLOCK_N BLOCK_K GROUP_M : Nat) :
    ComputeKernel :=
  Body.unscaledKernel (Region.cast a_ptr) (Region.cast b_ptr) c_ptr M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn BLOCK_M BLOCK_N BLOCK_K GROUP_M

def unscaledCopyFreeSource
    (a_ptr b_ptr : Region .int) (c_ptr : RegionName)
    (M N K : Nat)
    (stride_am stride_ak stride_bk stride_bn stride_cm stride_cn : Nat)
    (BLOCK_M BLOCK_N BLOCK_K GROUP_M : Nat) :
    ComputeKernel :=
  Body.unscaledCandidateKernel (Region.cast a_ptr) (Region.cast b_ptr) c_ptr M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn BLOCK_M BLOCK_N BLOCK_K GROUP_M

def scaledSource
    (a_ptr b_ptr : Region .int) (c_ptr s1_ptr : RegionName)
    (M N K : Nat)
    (stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      stride_s1m stride_s1n : Nat)
    (BLOCK_M BLOCK_N BLOCK_K GROUP_M : Nat) (EVEN_K : Bool) :
    ComputeKernel :=
  Body.scaledKernel (Region.cast a_ptr) (Region.cast b_ptr) c_ptr s1_ptr M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn stride_s1m stride_s1n BLOCK_M BLOCK_N BLOCK_K GROUP_M EVEN_K

def scaledSharedIndicesSource
    (a_ptr b_ptr : Region .int) (c_ptr s1_ptr : RegionName)
    (M N K : Nat)
    (stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      stride_s1m stride_s1n : Nat)
    (BLOCK_M BLOCK_N BLOCK_K GROUP_M : Nat) (EVEN_K : Bool) :
    ComputeKernel :=
  Body.scaledCandidateKernel (Region.cast a_ptr) (Region.cast b_ptr) c_ptr s1_ptr M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn stride_s1m stride_s1n BLOCK_M BLOCK_N BLOCK_K GROUP_M EVEN_K

def unscaled (p : BlockWord.Parameters) : ComputeKernel :=
  unscaledSource
  (Region.cast p.input) (Region.cast p.weight) p.output p.rows p.columns p.inner
  p.inputRowStride p.inputInnerStride p.weightInnerStride p.weightColumnStride
  p.outputRowStride p.outputColumnStride p.blockRows p.blockColumns p.blockInner p.groupRows

def unscaledCopyFree (p : BlockWord.Parameters) : ComputeKernel :=
  unscaledCopyFreeSource
  (Region.cast p.input) (Region.cast p.weight) p.output p.rows p.columns p.inner
  p.inputRowStride p.inputInnerStride p.weightInnerStride p.weightColumnStride
  p.outputRowStride p.outputColumnStride p.blockRows p.blockColumns p.blockInner p.groupRows

def scaled (p : ScaledWord.Parameters) : ComputeKernel :=
  scaledSource
  (Region.cast p.input) (Region.cast p.weight) p.output p.scale p.rows p.columns p.inner
  p.inputRowStride p.inputInnerStride p.weightInnerStride p.weightColumnStride
  p.outputRowStride p.outputColumnStride p.scaleRowStride p.scaleColumnStride
  p.blockRows p.blockColumns p.blockInner p.groupRows p.even

def scaledSharedIndices (p : ScaledWord.Parameters) : ComputeKernel :=
  scaledSharedIndicesSource
  (Region.cast p.input) (Region.cast p.weight) p.output p.scale p.rows p.columns p.inner
  p.inputRowStride p.inputInnerStride p.weightInnerStride p.weightColumnStride
  p.outputRowStride p.outputColumnStride p.scaleRowStride p.scaleColumnStride
  p.blockRows p.blockColumns p.blockInner p.groupRows p.even

end
end VeriTile.Bench.Repairs.IntScaledMatmul.Kernels
