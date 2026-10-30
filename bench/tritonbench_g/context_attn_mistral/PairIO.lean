import bench.tritonbench_g.context_attn_mistral.NativeIO
import VeriTile.Triton.Float.ArrayProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.ContextAttnMistralKernel.PairIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnMistral
open VeriTile.Bench.Repairs.ContextAttnMistral
noncomputable section
variable {α : Type} [Inhabited α]

def candidateInvocation (c : Host.Call) (o : FPBackend.Ops α) (idx : GridIndex c.grid) :
    Invocation α :=
  { NativeInterface.invocation c o idx with
    kernel := KernelSource.candidate c.layout c.regions (Host.scale c.layout.D) }

/-- The same array domain, finite access footprints and every final owner/lane,
with the actual block-origin-reuse source at each supplied job. -/
def candidateVariant (c : Host.Call) (o : FPBackend.Ops α) (jobs : List (GridIndex c.grid)) :
    ArrayProgramIO.Variant (NativeInterface.programIO c o jobs) where
  schedule := fun _ => jobs.map (candidateInvocation c o)
  observe := fun _ _ => id

def original (c : Host.Call) (o : FPBackend.Ops α) (jobs : List (GridIndex c.grid)) :=
  ArrayProgramIO.Implementation.original (NativeInterface.programIO c o jobs) (NativeArrays.metadata c)
def blockStartReused (c : Host.Call) (o : FPBackend.Ops α) (jobs : List (GridIndex c.grid)) :=
  ArrayProgramIO.Implementation.ofVariant (candidateVariant c o jobs) (NativeArrays.metadata c)

end
end VeriTile.Bench.Repairs.ContextAttnMistralKernel.PairIO
