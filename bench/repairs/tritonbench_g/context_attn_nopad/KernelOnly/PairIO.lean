import bench.repairs.tritonbench_g.context_attn_nopad.KernelOnly.IO
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.PairIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnNopad
noncomputable section
variable {α : Type} [Inhabited α]

def candidateInvocation (c : Call) (r : Regions) (o : FPBackend.Ops α) (idx : GridIndex c.grid) : Invocation α :=
  { IOData.job c r o idx with kernel := blockStartReusedFP32 c.layout r c.scale }

def candidateVariant (c : Call) (r : Regions) (o : FPBackend.Ops α) :
    ProgramIO.Variant (IOData.io c r o) where
  schedule := fun _ => (FPHost.jobs c).map (candidateInvocation c r o)
  observe := fun _ _ => id

def original (c : Call) (r : Regions) (o : FPBackend.Ops α) :=
  ProgramIO.Implementation.original (IOData.io c r o) (IOData.domain c r)
def blockStartReused (c : Call) (r : Regions) (o : FPBackend.Ops α) :=
  ProgramIO.Implementation.ofVariant (candidateVariant c r o) (IOData.domain c r)
end
end VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.PairIO
