import bench.repairs.tritonbench_g.matmul_dequantize.Kernels
import bench.repairs.tritonbench_g.matmul_dequantize.IO
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.MatmulDequantize.Candidate
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulDequantizeWhole001
noncomputable section
namespace First
abbrev Dot := VeriTile.Bench.TritonBenchG.MatmulDequantize.FPTypedFirst.Dot
abbrev Parameters := VeriTile.Bench.TritonBenchG.MatmulDequantize.FPTypedFirst.Parameters
def source (p : Parameters) : ComputeKernel := KernelSource.groupedGPTQ .fp16
  p.input p.output p.scale p.weight p.zero p.rows p.columns p.inner
  p.inputRowStride p.inputInnerStride p.weightInnerStride p.weightColumnStride
  p.outputRowStride p.outputColumnStride p.scaleGroupStride p.scaleColumnStride
  p.zeroGroupStride p.zeroColumnStride p.groupSize p.noGroups
  p.blockRows p.blockColumns p.blockInner p.groupRows

def invocation {A : Type} (M : Algebra A) (D : Dot A) (p : Parameters) (pid : Nat) : Invocation A :=
  { FirstIO.invocation M D p pid with kernel := source p }
def variant {A : Type} (M : Algebra A) (D : Dot A) (p : Parameters) (jobs : List Nat) :
    ProgramIO.Variant (FirstIO.program M D p jobs) where
  schedule := fun _ => jobs.map (invocation M D p)
  observe := fun _ _ => id
end First
namespace Second
abbrev Dot := FPTypedSecond.Dot
abbrev Parameters := FPTypedSecond.Parameters
def source (p : Parameters) : ComputeKernel := KernelSource.groupedS2 .fp32
  p.input p.output p.scale p.weight p.zero p.rows p.columns p.inner
  p.inputRowStride p.inputInnerStride p.weightInnerStride p.weightColumnStride
  p.outputRowStride p.outputColumnStride p.scaleGroupStride p.scaleColumnStride
  p.zeroGroupStride p.zeroColumnStride p.groupSize
  p.blockRows p.blockColumns p.blockInner p.groupRows p.steps

def invocation {A : Type} (M : Algebra A) (D : Dot A) (p : Parameters) (pid : Nat) : Invocation A :=
  { SecondIO.invocation M D p pid with kernel := source p }
def variant {A : Type} (M : Algebra A) (D : Dot A) (p : Parameters) (jobs : List Nat) :
    ProgramIO.Variant (SecondIO.program M D p jobs) where
  schedule := fun _ => jobs.map (invocation M D p)
  observe := fun _ _ => id
end Second
namespace D32
abbrev Parameters := Dequant32.Parameters
def source (p : Parameters) : ComputeKernel := KernelSource.affineDequant .fp32
  p.weight p.scale p.zero p.output p.rows p.columns p.groupSize
  p.weightRowStride p.weightColumnStride p.scaleRowStride p.scaleColumnStride
  p.zeroRowStride p.zeroColumnStride p.outputRowStride p.outputColumnStride p.blockRows p.blockColumns

def invocation {A : Type} (M : Algebra A) (p : Parameters) (pid : Nat) : Invocation A :=
  { D32IO.invocation M p pid with kernel := source p }
def variant {A : Type} (M : Algebra A) (p : Parameters) (jobs : List Nat) :
    ProgramIO.Variant (D32IO.program M p jobs) where
  schedule := fun _ => jobs.map (invocation M p)
  observe := fun _ _ => id
end D32
namespace D16
abbrev Parameters := VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.Parameters
def source (p : Parameters) : ComputeKernel := KernelSource.affineDequant .fp16
  p.weight p.scale p.zero p.output p.rows p.columns p.groupSize
  p.weightRowStride p.weightColumnStride p.scaleRowStride p.scaleColumnStride
  p.zeroRowStride p.zeroColumnStride p.outputRowStride p.outputColumnStride p.blockRows p.blockColumns

def invocation {A : Type} (M : Algebra A) (p : Parameters) (pid : Nat) : Invocation A :=
  { D16IO.invocation M p pid with kernel := source p }
def variant {A : Type} (M : Algebra A) (p : Parameters) (jobs : List Nat) :
    ProgramIO.Variant (D16IO.program M p jobs) where
  schedule := fun _ => jobs.map (invocation M p)
  observe := fun _ _ => id
end D16
end
end VeriTile.Bench.Repairs.MatmulDequantize.Candidate
