import bench.repairs.tritonbench_g.triton_argmax_nan_first.CandidateKernels
import bench.repairs.tritonbench_g.triton_argmax_nan_first.IO
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.JITProgramPair
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source
noncomputable section

def candidateFirst {casts kind} (binding : BoundSource.Binding casts kind) (c : GlobalConfig) (index64 : Bool) :=
  JITCandidate.stage1 BoundSource.X BoundSource.V BoundSource.I (JITBinding.dtype binding) c.block index64

def firstInvocation {casts kind} (binding : BoundSource.Binding casts kind) (rest : Algebra Word)
    (c : GlobalConfig) (index64 : Bool) (pid : Nat) : Invocation Word :=
  JITIO.firstLaunch (JITBinding.ops binding rest) (candidateFirst binding c index64) c.size 0 pid

def globalSchedule {casts kind} (binding : BoundSource.Binding casts kind) (rest : Algebra Word)
    (c : GlobalConfig) (index64 : Bool) :=
  (List.range c.blocks).map (firstInvocation binding rest c index64) ++
    [JITIO.secondLaunch (JITBinding.ops binding rest) (JITBinding.second binding c) c.blocks]

def globalVariant (casts : Request.Casts) (kind : Request.Kind) (binding : BoundSource.Binding casts kind)
    (rest : Algebra Word) (c : GlobalConfig) (index64 : Bool) :
    ProgramIO.Variant (JITIO.nativeGlobal casts kind binding rest c index64) where
  schedule := fun _=>globalSchedule binding rest c index64
  observe := fun _ _=>id

def dimensionInvocation (rest : Algebra Word) (c : DimConfig) (index64 : Bool) (job : Nat × Nat) : Invocation Word :=
  JITIO.dimensionLaunch (Typed.backend rest)
    (JITCandidate.dimension BoundSource.X BoundSource.O 128 128 index64)
    c.outer c.extent c.inner 0 (job.1*c.inner+job.2)

def dimensionVariant (rest : Algebra Word) (c : DimConfig) (index64 : Bool) :
    ProgramIO.Variant (JITIO.nativeDimension rest c index64) where
  schedule := fun _=>(BoundSource.groups c).map (dimensionInvocation rest c index64)
  observe := fun _ _=>id

end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.JITProgramPair
