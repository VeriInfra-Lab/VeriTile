import bench.repairs.tritonbench_g.triton_argmax_nan_first.CandidateKernels
import bench.repairs.tritonbench_g.triton_argmax_nan_first.IO
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.Pair
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source
noncomputable section

inductive Precision where | fp32 | fp16 | bf16 | fp64 | integer

def firstOriginal : Precision → RegionName → RegionName → Region .int → Nat → Nat → Nat → Bool → ComputeKernel
  | .fp32 => TypedSource.stage1FP32
  | .fp16 => TypedSource.stage1FP16
  | .bf16 => TypedSource.stage1BF16
  | .fp64 => TypedSource.stage1FP64
  | .integer => TypedSource.stage1Integer

def firstCandidate : Precision → RegionName → RegionName → Region .int → Nat → Nat → Nat → Bool → ComputeKernel
  | .fp32 => CandidateSource.stage1FP32
  | .fp16 => CandidateSource.stage1FP16
  | .bf16 => CandidateSource.stage1BF16
  | .fp64 => CandidateSource.stage1FP64
  | .integer => CandidateSource.stage1Integer

def precision : TypedScratch.Kind → Precision
  | .fp32 => .fp32 | .fp16 => .fp16 | .bf16 => .bf16 | .fp64 => .fp64

def candidateFirst {casts kind} : BoundSource.Binding casts kind → GlobalConfig → Bool → ComputeKernel
  | .floating k _ _,c,index64 => firstCandidate (precision k) BoundSource.X BoundSource.V BoundSource.I c.size c.block 0 index64
  | .integer _ _,c,index64 => firstCandidate .integer BoundSource.X BoundSource.V BoundSource.I c.size c.block 0 index64

def firstInvocation {casts kind} (binding : BoundSource.Binding casts kind) (rest : Algebra Word)
    (c : GlobalConfig) (index64 : Bool) (pid : Nat) : Invocation Word :=
  IO.atProgram (binding.ops rest) (candidateFirst binding c index64) pid

def globalSchedule {casts kind} (binding : BoundSource.Binding casts kind) (rest : Algebra Word)
    (c : GlobalConfig) (index64 : Bool) :=
  (List.range c.blocks).map (firstInvocation binding rest c index64) ++
    [IO.unchanged (binding.ops rest) (binding.second c)]

def globalVariant (casts : Request.Casts) (kind : Request.Kind) (binding : BoundSource.Binding casts kind)
    (rest : Algebra Word) (c : GlobalConfig) (index64 : Bool) :
    ProgramIO.Variant (IO.nativeGlobal casts kind binding rest c index64) where
  schedule := fun _=>globalSchedule binding rest c index64
  observe := fun _ _=>id

def dimensionInvocation (rest : Algebra Word) (c : DimConfig) (index64 : Bool) (job : Nat × Nat) : Invocation Word :=
  IO.atProgram (Typed.backend rest)
    (CandidateSource.dimension BoundSource.X BoundSource.O c.outer c.extent c.inner 128 128 0 index64)
    (job.1*c.inner+job.2)

def dimensionVariant (rest : Algebra Word) (c : DimConfig) (index64 : Bool) :
    ProgramIO.Variant (IO.nativeDimension rest c index64) where
  schedule := fun _=>(BoundSource.groups c).map (dimensionInvocation rest c index64)
  observe := fun _ _=>id

end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.Pair
