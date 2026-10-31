import bench.tritonbench_g.int8_matmul_quantization.Sources.Mathematical_Int8MatmulQuantization
import bench.tritonbench_g.quantize_kv_transform.Sources.Mathematical_MixedBits
import bench.tritonbench_g.kcache_copy_triton.Sources.Mathematical_TypedCopy

namespace VeriTile.Bench.TritonBenchG.Int8MatmulQuantization.FPQuantize
open VeriTile.Triton FP.Structural FP.IntegerConversion
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false
abbrev Word := FP.Value .fp32
local instance sourceWordDefault : Inhabited Word := ⟨FP.Value.zero .fp32⟩

structure Parameters where
  (input output scales : RegionName)
  (rows columns inputRowStride inputColumnStride outputRowStride outputColumnStride scaleStride
    blockRows blockColumns steps : Nat)

/-- Explicit software maximum-number profile for the pointwise maximum.
A single NaN is suppressed, both NaNs return the canonical fp32 quiet NaN,
and equal ordered values select the right operand. The last two choices
remain native payload/signed-zero binding obligations; they are not obtained
from a comparison/select lowering or a non-NaN premise. -/
def pairMaximum (a b : Word) : Word :=
  if a.isNaN then (if b.isNaN then FP.Value.qNaN .fp32 else b)
  else if b.isNaN then a
  else if FP.lt .ieeeValue b a then a else b

noncomputable def backend (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int) : Algebra Word :=
  let base := QuantizeKvTransform.MixedBits.backend rest cfg u
  { base with binary := fun pr d op a b =>
      if FP.Software.fp32Context pr d ∧ op = .max then pairMaximum a b
      else base.binary pr d op a b }

abbrev Index (p : Parameters) := TileIndex [p.blockRows, p.blockColumns]
def row (p : Parameters) (s : State Word) (i : Fin p.blockRows) : Nat :=
  (s.pids 0 * p.blockRows + i.val) % p.rows

def inputOffset (p : Parameters) (s : State Word) (k : Nat) (i : Index p) : Nat :=
  row p s i.1 * p.inputRowStride + i.2.1.val * p.inputColumnStride + k * (p.blockColumns * p.inputColumnStride)

def outputOffset (p : Parameters) (s : State Word) (k : Nat) (i : Index p) : Nat :=
  row p s i.1 * p.outputRowStride + i.2.1.val * p.outputColumnStride + k * (p.blockColumns * p.outputColumnStride)

def active (p : Parameters) (k : Nat) (i : Index p) : Prop :=
  i.2.1.val < p.columns - k * p.blockColumns
instance (p : Parameters) (k : Nat) : DecidablePred (active p k) := fun _ => inferInstanceAs (Decidable (_ < _))

def loaded (p : Parameters) (s : State Word) (k : Nat) (i : Index p) : Word :=
  if active p k i then (s.mem p.input (inputOffset p s k i)).read .real else FP.Value.zero .fp32

def blockMaximum (p : Parameters) (s : State Word) (k : Nat) (i : Fin p.blockRows) : Word :=
  QuantizeKvCopy.TypedQuantize.maximum (fun j => (loaded p s k (i,j,PUnit.unit)).abs)

def runningMaximum (p : Parameters) (s : State Word) : Nat → Fin p.blockRows → Word
  | 0 => fun _ => FP.Value.zero .fp32
  | k+1 => fun i => pairMaximum (runningMaximum p s k i) (blockMaximum p s k i)

def scale (cfg : FP.Config) (p : Parameters) (s : State Word) (i : Fin p.blockRows) : Word :=
  FP.div cfg (runningMaximum p s p.steps i) (FP.Value.ofBits .fp32 0x42fe0000)

def quotient (cfg : FP.Config) (p : Parameters) (s : State Word) (k : Nat) (i : Index p) : Word :=
  FP.div cfg (loaded p s k i) (scale cfg p s i.1)

def expected (cfg : FP.Config) (p : Parameters) (s : State Word) (k : Nat) (i : Index p) : Int :=
  (toSigned 8 (quotient cfg p s k i)).getD 0

def InputDefined (cfg : FP.Config) (p : Parameters) (s : State Word) : Prop :=
  ∀ k, k < p.steps → ∀ i, active p k i → (toSigned 8 (quotient cfg p s k i)).isSome

def scaleOffset (p : Parameters) (s : State Word) (i : Fin p.blockRows) : Nat :=
  s.pids 0 * p.blockRows * p.scaleStride + i.val

end
end VeriTile.Bench.TritonBenchG.Int8MatmulQuantization.FPQuantize
