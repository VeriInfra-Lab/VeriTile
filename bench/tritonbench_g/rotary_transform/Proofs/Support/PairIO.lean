import bench.tritonbench_g.rotary_transform.Proofs.PublicIO
import bench.tritonbench_g.rotary_transform.Proofs.IndexCSE
import VeriTile.Triton.Float.ReadProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.RotaryTransformTileChecked.PairIO
open VeriTile.Bench.Repairs.RotaryTransform
open VeriTile.Triton FP.Structural
noncomputable section

def invocation (p : Ops α) (c : Config) (cu64 off64 : Bool) (pid : Program) : Invocation α :=
  { PublicIO.invocation p c cu64 off64 pid with kernel := IndexCSE.candidate c.shape cu64 off64 }

/-- Actual complete candidate grid, with exactly the original result reads,
smaller rotary write footprint, logical inputs, metadata and observer. -/
def candidate (p : Ops α) (observe : α → α) (c : Config) (cu64 off64 : Bool) :
    ReadProgramIO.Variant (PublicIO.io p observe c cu64 off64) where
  schedule := fun _ => if c.shape.total=0 ∨ c.shape.half=0 then [] else
    (programs c).map (invocation p c cu64 off64)
  observe := fun _ _ => observe

def original (c : Config) (p : Native.Instructions) (cu64 off64 : Bool) :=
  ReadProgramIO.Implementation.original
    (PublicIO.io (Native.ops p) (fun n => Native.atom n.value) c cu64 off64) (PublicIO.domain c)
def indexShared (c : Config) (p : Native.Instructions) (cu64 off64 : Bool) :=
  ReadProgramIO.Implementation.ofVariant
    (candidate (Native.ops p) (fun n => Native.atom n.value) c cu64 off64) (PublicIO.domain c)

end
end VeriTile.Bench.Repairs.RotaryTransformTileChecked.PairIO
