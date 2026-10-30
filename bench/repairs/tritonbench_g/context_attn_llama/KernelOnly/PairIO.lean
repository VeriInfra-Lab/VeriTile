import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.NativeIO
import VeriTile.Triton.Float.ArrayProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.PairIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnLlama
noncomputable section
variable {α : Type} [Inhabited α]

def candidateInvocation (c : Host.Call) (M : Algebra α) (idx : GridIndex c.grid) : Invocation α :=
  { NativeInterface.invocation c M idx with
    kernel := blockEndFactoredFP32 c.layout c.regions (Host.scale c.layout.D) }

def candidateVariant (c : Host.Call) (M : Algebra α) (dots : FPSuccess.DotSupport c.layout M)
    (order : List (GridIndex c.grid)) : ArrayProgramIO.Variant (NativeInterface.programIO c M dots order) where
  schedule := fun _ => order.map (candidateInvocation c M)
  observe := fun _ _ => id

def original (c : Host.Call) (M : Algebra α) (dots : FPSuccess.DotSupport c.layout M)
    (order : List (GridIndex c.grid)) :=
  ArrayProgramIO.Implementation.original (NativeInterface.programIO c M dots order) (NativeArrays.metadata c)
def blockEndFactored (c : Host.Call) (M : Algebra α) (dots : FPSuccess.DotSupport c.layout M)
    (order : List (GridIndex c.grid)) :=
  ArrayProgramIO.Implementation.ofVariant (candidateVariant c M dots order) (NativeArrays.metadata c)
end
end VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.PairIO
