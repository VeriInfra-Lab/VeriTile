import bench.repairs.tritonbench_g.chunk_gla_simple_min_dot_tile.IO
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.PairIO
open VeriTile Triton FP.Structural Host
noncomputable section

/-- The same actual launch, backend, runtime arguments and program IDs. -/
def candidateLaunch (c : Shape) (q : GridIndex (grid c)) : Invocation NativeGraph.Expr :=
  { PublicIO.nativeLaunch c q with kernel := rowReusedKernel c ScalarGraph.scaleTag }

/-- Only the actual kernel source changes. The original arbitrary symbolic
input domain, encoding, complete output and full memory footprint stay fixed. -/
def candidate (c : Shape) : ProgramIO.Variant (PublicIO.nativeIO c) where
  schedule := fun _ => (FPGrid.programs c).map (candidateLaunch c)
  observe := fun _ _ => id

def original (c : Shape) :=
  ProgramIO.Implementation.original (PublicIO.nativeIO c) (fun _ => True)

def rowReused (c : Shape) :=
  ProgramIO.Implementation.ofVariant (candidate c) (fun _ => True)

end
end VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.PairIO
