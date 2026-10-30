import bench.tritonbench_g.rotary_transform_ops.Kernels
import VeriTile.Triton.Float.Structural

namespace VeriTile.Bench.Repairs.RotaryTransformOps.KernelAlgebra
open VeriTile.Triton FP.Structural
noncomputable section
open Classical

/-- Interpretation of the operations actually used by the selected FP32
source: constants zero/one, same-format copies, negation and three binary
operators. Unused callback fields supply no laws about those operations.
For native words, same-format FP32 loads/casts preserve their raw bits. -/
def backend {α : Type} (p : Ops α) : Algebra α where
  literal := fun _ _ value => if value = 1 then p.one else p.zero
  negInf := p.zero
  binary := fun _ _ code => match code with
    | .add => p.add | .sub => p.sub | .mul => p.mul | _ => fun a _ => a
  unary := fun _ code => match code with
    | .neg _ => p.neg | _ => id
  cast := fun _ _ _ => id
  fromNat := fun _ value => if value = 1 then p.one else p.zero
  fromInt := fun _ value => if value = 1 then p.one else p.zero
  fp32Bits := fun _ => p.zero
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ _ => p.zero
  reduceSum := fun _ {_} _ _ _ _ => p.zero

def regionName : RotaryTransformOps.Region → RegionName
  | .input => "x" | .output => "out" | .cosine => "cos" | .sine => "sin"
  | .cumulative => "cu" | .offsets => "offsets" | .other _ => "other"

/-- The compute-facing FP32 quotation uses the erased `.real` carrier.
Its FP32 precision remains in ComputeDType tags and the chosen callbacks;
the carrier is opaque and does not assert real-number arithmetic. -/
def memory {α : Type} [Inhabited α] (s : State α) : Memory α :=
  fun role offset => (s.mem (regionName role) offset).read .real

@[simp] theorem literal_zero {α : Type} (p : Ops α) (precision : Option ComputeDType)
    (dtype : FloatDType) : (backend p).literal precision dtype 0 = p.zero := by
  simp [backend]

@[simp] theorem literal_one {α : Type} (p : Ops α) (precision : Option ComputeDType)
    (dtype : FloatDType) : (backend p).literal precision dtype 1 = p.one := by
  simp [backend]

#axiomsClean literal_zero
#axiomsClean literal_one
end
end VeriTile.Bench.Repairs.RotaryTransformOps.KernelAlgebra
