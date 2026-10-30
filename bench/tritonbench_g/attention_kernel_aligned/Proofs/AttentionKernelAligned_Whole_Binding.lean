import bench.tritonbench_g.attention_kernel_aligned.Proofs.NativePlan
import bench.tritonbench_g.attention_kernel_aligned.Proofs.Physical
namespace VeriTile.Bench.Repairs.AttentionKernelAligned.Whole
open Full
inductive Profile where | baseline | independent | permuted | unaligned | alignedFull
  deriving DecidableEq
def argumentClass (n : Nat) : Nat := if n=1 then 1 else if n%16=0 then 16 else 0
def pointerClasses : Profile → List Nat
  | .baseline=>[16, 16, 16, 16, 16]
  | .independent=>[16, 16, 0, 16, 16]
  | .permuted=>[16, 0, 16, 16, 16]
  | .unaligned=>[0, 0, 0, 16, 0]
  | .alignedFull=>[16, 16, 16, 16, 16]
def integerClasses : Profile → List Nat
  | .baseline=>[16, 16, 16, 1, 16, 16, 16, 1, 16, 16, 16, 1, 16, 16, 16, 1, 16, 16, 16, 1, 1, 0, 16]
  | .independent=>[16, 16, 16, 0, 16, 16, 16, 1, 16, 16, 16, 0, 16, 16, 16, 1, 16, 16, 16, 0, 0, 0, 16]
  | .permuted=>[16, 16, 16, 1, 16, 16, 16, 0, 16, 16, 1, 0, 16, 16, 16, 1, 16, 16, 16, 0, 0, 0, 16]
  | .unaligned=>[16, 16, 16, 1, 16, 16, 16, 1, 16, 16, 16, 1, 16, 16, 16, 1, 16, 16, 16, 0, 0, 0, 16]
  | .alignedFull=>[16, 16, 16, 1, 16, 16, 16, 1, 16, 16, 16, 1, 16, 16, 16, 1, 16, 16, 16, 0, 0, 16, 16]
def dynamicValues (p : Parameters) : List Nat :=
  [p.qBatchStride,p.qHeadStride,p.qRowStride,p.qColStride,
   p.kBatchStride,p.kHeadStride,p.kSeqStride,p.kDimStride,
   p.vBatchStride,p.vHeadStride,p.vSeqStride,p.vDimStride,
   p.outBatchStride,p.outHeadStride,p.outRowStride,p.outColStride,
   p.biasBatchStride,p.biasHeadStride,p.biasRowStride,p.batches,p.heads,p.length,0]

/-- Decoded integer compiler metadata. Source staging fixes all numerical
operations; this ordinary compiler boundary only binds launch constants and
MMA/reduction routes. Saved artifact coverage is recorded separately. -/
structure CompilerBinding (p : Parameters) (_storage : TensorStorage p) (_abi : InputABI p)
    (_pl : Placement p _storage) (_profile : Profile) where
  serialAbsent : p.nativeSerial=Bool.false
  sharedBytes : Nat
  sharedFits : sharedBytes≤163840
  columns : p.nativeColumns∈[1,2,4,8]
  columns_bound : p.nativeColumns≤p.numWarps
  routes : VeriTile.Bench.Repairs.AttentionKernelAligned.NativePlan.Routes
  routing : VeriTile.Bench.Repairs.AttentionKernelAligned.NativePlan.LocalRouting p routes

theorem saved_resource_fit : (24704:Nat)≤163840 := by decide
#axiomsClean saved_resource_fit
#specNonCircular CompilerBinding avoiding [invoke,runPrograms,VeriTile.Triton.FP.Structural.exec]
end VeriTile.Bench.Repairs.AttentionKernelAligned.Whole
