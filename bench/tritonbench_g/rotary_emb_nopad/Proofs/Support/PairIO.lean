import bench.tritonbench_g.rotary_emb_nopad.IO
import bench.tritonbench_g.rotary_emb_nopad.Proofs.AddressCSE
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.RotaryEmbNopad.PairIO
open VeriTile.Triton FP.Structural KernelInterface
noncomputable section

def uInvocation (rest : Algebra α) (ops : Ops α) (g : Geometry) (p q : Nat) : Invocation α :=
  { RotationIO.uInvocation rest ops g p q with kernel := Kernels.uncachedShared (Typed.uLayout g.request) }
def cInvocation (rest : Algebra α) (ops : Ops α) (g : Geometry) (p q : Nat) : Invocation α :=
  { RotationIO.cInvocation rest ops g p q with kernel := Kernels.cachedShared (Typed.cLayout g.request) }

def uncached (rest : Algebra α) (ops : Ops α) (g : Geometry) (p q : Nat) :
    ProgramIO.Variant (RotationIO.uncachedIO rest ops g p q) where
  schedule := fun _ => [uInvocation rest ops g p q]
  observe := fun _ _ => id

def cached (rest : Algebra α) (ops : Ops α) (g : Geometry) (p q : Nat) :
    ProgramIO.Variant (RotationIO.cachedIO rest ops g p q) where
  schedule := fun _ => [cInvocation rest ops g p q]
  observe := fun _ _ => id

def uncachedOriginal (rest : Algebra α) (ops : Ops α) (g : Geometry) (p q : Nat) :=
  ProgramIO.Implementation.original (RotationIO.uncachedIO rest ops g p q) (fun _ => True)
def uncachedShared (rest : Algebra α) (ops : Ops α) (g : Geometry) (p q : Nat) :=
  ProgramIO.Implementation.ofVariant (uncached rest ops g p q) (fun _ => True)
def cachedOriginal (rest : Algebra α) (ops : Ops α) (g : Geometry) (p q : Nat) :=
  ProgramIO.Implementation.original (RotationIO.cachedIO rest ops g p q) (RotationIO.MetadataDomain g)
def cachedShared (rest : Algebra α) (ops : Ops α) (g : Geometry) (p q : Nat) :=
  ProgramIO.Implementation.ofVariant (cached rest ops g p q) (RotationIO.MetadataDomain g)

end
end VeriTile.Bench.Repairs.RotaryEmbNopad.PairIO
