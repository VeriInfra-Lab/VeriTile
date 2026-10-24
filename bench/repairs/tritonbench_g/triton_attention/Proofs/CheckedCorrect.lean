import bench.repairs.tritonbench_g.triton_attention.Proofs.ByteViews
import bench.repairs.tritonbench_g.triton_attention.Whole001.Correct

namespace VeriTile.Bench.Repairs.TritonAttention.Whole001.Checked
open Native Memory
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 2500000

/-- Erasure reads the actual caller bytes and the actual incoming gradient's
strides. It does not equate native floating arithmetic with real arithmetic. -/
def erasedInput (c : Shape) (objects : Buffer→Tensor) (incoming : Tensor)
    (before : ByteMemory) (interpret : Word→ℝ) : Memory ℝ :=
  fun r n=>interpret (if r=Buffer.g then gradientWord c incoming before n else
    if r∈[Buffer.q,.k,.v] then gradientWord c (objects r) before n else callerWords objects before r n)

/-- Structural safety is derived solely from executed metadata/pointer checks.
No numerical-to-real relation, finite-content premise, or final result appears. -/
theorem structural_safety (g : Geometry) (hs : shapeChecks g=Bool.true)
    (t : Tensor) (hp : pointerChecks g t=Bool.true) :
    (∀ offset≤pointerEnvelope g,∀ j<t.width,
      offset<2^63 ∧ t.dataPointer%t.width=0 ∧ t.dataPointer+t.width*offset+j<2^64) ∧
    (∀ offset<t.numel,∀ j<t.width,
      t.storageBase≤t.dataPointer+t.width*offset+j ∧
      t.dataPointer+t.width*offset+j<t.storageBase+t.storageBytes) ∧
    (∀ i : Coord (checkedShape g hs),
      offset (checkedShape g hs) i<2147483647 ∧
      4*offset (checkedShape g hs) i<9223372036854775807 ∧
      4*((checkedShape g hs).elements+16384)<9223372036854775807) :=
  ⟨formed_bytes g hs t hp,active_bytes g t hp,machine_offsets (checkedShape g hs)⟩

/-- Arbitrary real cell values indexed by the same physical Tensor addresses
and observed source strides. Aliases share the same real address interpretation;
values need not be representable by any floating-point word. -/
def realInput (c : Shape) (objects : Buffer→Tensor) (incoming : Tensor)
    (before : Nat→ℝ) : Memory ℝ := fun r n=>
  let t:=if r=Buffer.g then incoming else objects r
  let element:=if r∈[Buffer.q,.k,.v,.g] then gradientOffset c t n else n
  before (t.dataPointer+4*element)

/-- Main independent real specification: arbitrary real values at actual
Tensor addresses, with no finite-word decoder or representability restriction.
The real scalar is independent of the checked native scalar observation. -/
theorem whole_real (g : Geometry) (hs : shapeChecks g=Bool.true) (device : Nat)
    (objects : Objects (checkedShape g hs) device) (incoming : Tensor) (before : Nat→ℝ)
    (scale : ScalarABI) (realScale : ℝ)
    (checks : inputChecks g (objects.tensor .q) (objects.tensor .k) (objects.tensor .v) scale=Bool.true) :
    Host.InputChecks (checkedShape g hs) (arguments (objects.tensor .q) (objects.tensor .k) (objects.tensor .v) scale) ∧
    let c:=checkedShape g hs
    let s:=realInput c objects.tensor incoming before
    (∀ h : Fin c.groups,
      readTensor c (RealProgram.run c s realScale) .o h=
        Math.attention (readTensor c s .q h) (readTensor c s .k h) (readTensor c s .v h) realScale ∧
      readTensor c (RealProgram.run c s realScale) .dq h=
        Math.gradientQ (readTensor c s .q h) (readTensor c s .k h) (readTensor c s .v h)
          (readTensor c s .g h) realScale ∧
      readTensor c (RealProgram.run c s realScale) .dk h=
        Math.gradientK (readTensor c s .q h) (readTensor c s .k h) (readTensor c s .v h)
          (readTensor c s .g h) realScale ∧
      readTensor c (RealProgram.run c s realScale) .dv h=
        Math.gradientV (readTensor c s .q h) (readTensor c s .k h) (readTensor c s .g h) realScale) ∧
    (∀ h : Fin c.groups,∀ dQ dK dV : Math.Tensor c.rows c.dim,
      HasDerivAt (fun t=>Math.loss
        (Math.perturb (readTensor c s .q h) dQ t)
        (Math.perturb (readTensor c s .k h) dK t)
        (Math.perturb (readTensor c s .v h) dV t) (readTensor c s .g h) realScale)
        ((∑ i,∑ e,readTensor c (RealProgram.run c s realScale) .dq h i e*dQ i e)+
         (∑ i,∑ e,readTensor c (RealProgram.run c s realScale) .dk h i e*dK i e)+
          ∑ i,∑ e,readTensor c (RealProgram.run c s realScale) .dv h i e*dV i e) 0) ∧
    (∀ r n,¬RealProgram.Written c r n → RealProgram.run c s realScale r n=s r n) := by
  refine ⟨inputs_admitted g hs _ _ _ scale checks,?_⟩
  dsimp only
  exact ⟨RealProgram.whole_memory_values _ _ _,RealProgram.whole_memory_vjp _ _ _,RealProgram.whole_memory_frame _ _ _⟩

/-- Separate decoded-word instance of the real specification. This does not
assert that native execution equals erased-real execution. -/
theorem whole_real_decoded (g : Geometry) (hs : shapeChecks g=Bool.true) (device : Nat)
    (objects : Objects (checkedShape g hs) device) (incoming : Tensor) (before : ByteMemory)
    (scale : ScalarABI) (interpret : Word→ℝ)
    (checks : inputChecks g (objects.tensor .q) (objects.tensor .k) (objects.tensor .v) scale=Bool.true) :
    Host.InputChecks (checkedShape g hs) (arguments (objects.tensor .q) (objects.tensor .k) (objects.tensor .v) scale) ∧
    let c:=checkedShape g hs
    let s:=erasedInput c objects.tensor incoming before interpret
    (∀ h : Fin c.groups,
      readTensor c (RealProgram.run c s (scale.project scale.raw)) .o h=
        Math.attention (readTensor c s .q h) (readTensor c s .k h) (readTensor c s .v h) (scale.project scale.raw) ∧
      readTensor c (RealProgram.run c s (scale.project scale.raw)) .dq h=
        Math.gradientQ (readTensor c s .q h) (readTensor c s .k h) (readTensor c s .v h)
          (readTensor c s .g h) (scale.project scale.raw) ∧
      readTensor c (RealProgram.run c s (scale.project scale.raw)) .dk h=
        Math.gradientK (readTensor c s .q h) (readTensor c s .k h) (readTensor c s .v h)
          (readTensor c s .g h) (scale.project scale.raw) ∧
      readTensor c (RealProgram.run c s (scale.project scale.raw)) .dv h=
        Math.gradientV (readTensor c s .q h) (readTensor c s .k h) (readTensor c s .g h) (scale.project scale.raw)) ∧
    (∀ h : Fin c.groups,∀ dQ dK dV : Math.Tensor c.rows c.dim,
      HasDerivAt (fun t=>Math.loss
        (Math.perturb (readTensor c s .q h) dQ t)
        (Math.perturb (readTensor c s .k h) dK t)
        (Math.perturb (readTensor c s .v h) dV t) (readTensor c s .g h) (scale.project scale.raw))
        ((∑ i,∑ e,readTensor c (RealProgram.run c s (scale.project scale.raw)) .dq h i e*dQ i e)+
         (∑ i,∑ e,readTensor c (RealProgram.run c s (scale.project scale.raw)) .dk h i e*dK i e)+
          ∑ i,∑ e,readTensor c (RealProgram.run c s (scale.project scale.raw)) .dv h i e*dV i e) 0) ∧
    (∀ r n,¬RealProgram.Written c r n → RealProgram.run c s (scale.project scale.raw) r n=s r n) := by
  refine ⟨inputs_admitted g hs _ _ _ scale checks,?_⟩
  dsimp only
  exact ⟨RealProgram.whole_memory_values _ _ _,RealProgram.whole_memory_vjp _ _ _,RealProgram.whole_memory_frame _ _ _⟩

#axiomsClean structural_safety
#axiomsClean whole_real
#axiomsClean whole_real_decoded
end
end VeriTile.Bench.Repairs.TritonAttention.Whole001.Checked
