import VeriTile.Triton.Float.ProgramEquivalence
import bench.repairs.tritonbench_g.int_scaled_matmul.Proofs.NativePrimitive

namespace VeriTile.Bench.Repairs.IntScaledMatmul.NativeIO
open VeriTile.Triton FP.Structural Revision001 FiniteExecution
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- The source is unchanged; integer dot uses the fixed K32 instruction
partition. The running signed32 carry update remains a separate wrapping add. -/
def intInvocation (rest : Algebra Word) (x : Launch) (pid : Nat) : Invocation Word :=
  { IntIO.invocation rest (blockParameters x) pid with
    backend := NativePrimitive.intBackend rest x.config }

def intVariant (rest : Algebra Word) (x : Launch) : ProgramIO.Variant (IntIO.io rest (blockParameters x)) where
  schedule := fun _ => (List.range (Geometry.count (BlockWord.geometry (blockParameters x)))).map
    (intInvocation rest x)
  observe := fun _ _ => id

/-- The scaled instruction profile keeps the actual promoted-carry multiply
order and the same arbitrary total cast port. No FP32 words are excluded. -/
def scaledInvocation (rest : Algebra Word) (undefined : Word → BitVec 32)
    (x : Launch) (scale : RegionName) (pid : Nat) : Invocation Word :=
  { ScaledIO.invocation rest undefined (scaledParameters x scale) pid with
    backend := NativePrimitive.scaledBackend rest undefined x.config }

def scaledVariant (rest : Algebra Word) (undefined : Word → BitVec 32)
    (x : Launch) (scale : RegionName) : ProgramIO.Variant (ScaledIO.io rest undefined (scaledParameters x scale)) where
  schedule := fun _ => (List.range (Geometry.count (ScaledWord.geometry (scaledParameters x scale)))).map
    (scaledInvocation rest undefined x scale)
  observe := fun _ _ => id

theorem intInvocation_eq (rest : Algebra Word) (x : Launch)
    (legal : Contract.ConfigContract x.config) (pid : Nat) :
    intInvocation rest x pid=IntIO.invocation rest (blockParameters x) pid := by
  unfold intInvocation
  rw [NativePrimitive.intBackend_eq rest x.config legal]
  rfl

theorem scaledInvocation_eq (rest : Algebra Word) (undefined : Word → BitVec 32)
    (x : Launch) (scale : RegionName) (legal : Contract.ConfigContract x.config) (pid : Nat) :
    scaledInvocation rest undefined x scale pid=ScaledIO.invocation rest undefined (scaledParameters x scale) pid := by
  unfold scaledInvocation
  rw [NativePrimitive.scaledBackend_eq rest undefined x.config legal]
  rfl

theorem intInvocations_eq (rest : Algebra Word) (x : Launch)
    (legal : Contract.ConfigContract x.config) :
    intInvocation rest x=IntIO.invocation rest (blockParameters x) :=
  funext (intInvocation_eq rest x legal)

theorem scaledInvocations_eq (rest : Algebra Word) (undefined : Word → BitVec 32)
    (x : Launch) (scale : RegionName) (legal : Contract.ConfigContract x.config) :
    scaledInvocation rest undefined x scale=ScaledIO.invocation rest undefined (scaledParameters x scale) :=
  funext (scaledInvocation_eq rest undefined x scale legal)

theorem intIO_eq (rest : Algebra Word) (x : Launch) (legal : Contract.ConfigContract x.config) :
    (intVariant rest x).toProgramIO=IntIO.io rest (blockParameters x) := by
  unfold intVariant ProgramIO.Variant.toProgramIO
  rw [intInvocations_eq rest x legal]
  rfl

theorem scaledIO_eq (rest : Algebra Word) (undefined : Word → BitVec 32)
    (x : Launch) (scale : RegionName) (legal : Contract.ConfigContract x.config) :
    (scaledVariant rest undefined x scale).toProgramIO=ScaledIO.io rest undefined (scaledParameters x scale) := by
  unfold scaledVariant ProgramIO.Variant.toProgramIO
  rw [scaledInvocations_eq rest undefined x scale legal]
  rfl

#axiomsClean intIO_eq
#axiomsClean scaledIO_eq
end
end VeriTile.Bench.Repairs.IntScaledMatmul.NativeIO
