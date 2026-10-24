import bench.tritonbench_g.layer_norm_ops.FPCorrect
import bench.tritonbench_g.layer_norm_ops.Proofs.FPSoftware
namespace VeriTile.Bench.TritonBenchG.LayerNormOps.FPClosed
open VeriTile.Triton FP.Structural FPStages
open MatrixVectorMultip.Typed (sequentialSum)
noncomputable section
set_option maxHeartbeats 2000000

abbrev Word := FP.Value .fp32
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Fixed values for primitives unreachable in these nine accepted stages.
They are not claimed to implement transcendental or native operations. -/
def unused : Algebra Word where
  literal := fun _ _ _ => FP.Value.zero .fp32
  negInf := FP.Value.zero .fp32
  binary := fun _ _ _ _ _ => FP.Value.zero .fp32
  unary := fun _ _ _ => FP.Value.zero .fp32
  cast := fun _ _ _ x => x
  fromNat := fun _ _ => FP.Value.zero .fp32
  fromInt := fun _ _ => FP.Value.zero .fp32
  fp32Bits := fun _ => FP.Value.zero .fp32
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ _ => FP.Value.zero .fp32
  reduceSum := fun _ {_} _ _ _ _ => FP.Value.zero .fp32

/-- The source's only nonzero real literal is the exact natural dimension N.
Encode it by rational-to-binary32 round-to-nearest-even, with no arbitrary
Real→bits callback. Other real literals are unreachable, and return +0. -/
def dimensionLiteral (N : Nat) (x : ℝ) : Word :=
  if x=0 then FP.Value.zero .fp32
  else if x=(N : ℝ) then FP.round .fp32 .ieeeValue (N : Rat)
  else FP.Value.zero .fp32

theorem dimensionLiteral_zero (N : Nat) : dimensionLiteral N 0 = FP.Value.zero .fp32 := by
  simp [dimensionLiteral]

theorem dimensionLiteral_self (N : Nat) :
    dimensionLiteral N (N : ℝ) = FP.round .fp32 .ieeeValue (N : Rat) := by
  by_cases h : N=0
  · subst N
    norm_num only [Nat.cast_zero]
    rw [dimensionLiteral_zero]
    decide +kernel
  · have hr : (N : ℝ) ≠ 0 := by exact_mod_cast h
    simp [dimensionLiteral, hr]

def withRest (rest : Algebra Word) (N : Nat) : Algebra Word :=
  { MatrixVectorMultip.Typed.backend rest .ieeeValue with
    literal := fun _ _ x => dimensionLiteral N x
    reduceSum := fun _ {_} ax keep xs => sequentialSum .ieeeValue ax keep xs }

def backend (N : Nat) : Algebra Word := withRest unused N

def bitLoaded (p : Parameters) (s : State Word) (r : RegionName) (base : Nat)
    (i : TileIndex [p.block]) : Word :=
  if i.1.val<p.columns then (s.mem r (base+i.1.val)).read .fp32 else FP.Value.zero .fp32
def bitWeight (p : Parameters) (s : State Word) (i : TileIndex [p.block]) : Word :=
  if i.1.val<p.columns then (s.mem p.weight i.1.val).read .fp32 else s.undef .fp32 p.weight i.1.val
def bitScalar (s : State Word) (r : RegionName) : Word := (s.mem r (s.pids 0)).read .fp32
def bitDY (p : Parameters) (s : State Word) := bitLoaded p s p.dy (s.pids 0*p.dyStride)
def bitXhat (p : Parameters) (s : State Word) (plain : Bool) (i : TileIndex [p.block]) : Word :=
  if i.1.val<p.columns then FP.mul .ieeeValue
    (if plain then FP.sub .ieeeValue (bitLoaded p s p.x (s.pids 0*p.xStride) i) (bitScalar s p.mean)
      else bitLoaded p s p.x (s.pids 0*p.xStride) i) (bitScalar s p.rstd)
  else FP.Value.zero .fp32
def bitWDY (p : Parameters) (s : State Word) (i : TileIndex [p.block]) : Word :=
  FP.mul .ieeeValue (bitWeight p s i) (bitDY p s i)
def bitAverage (p : Parameters) (xs : TileIndex [p.block] → Word) : Word :=
  FP.div .ieeeValue
    (sequentialSum .ieeeValue ⟨0,by simp⟩ Bool.false xs PUnit.unit)
    (FP.round .fp32 .ieeeValue (p.columns : Rat))
def bitRowC1 (p : Parameters) (s : State Word) (plain : Bool) : Word :=
  bitAverage p (fun i => FP.mul .ieeeValue (bitXhat p s plain i) (bitWDY p s i))
def bitC2 (p : Parameters) (s : State Word) : Word := bitAverage p (bitWDY p s)
def bitDX (p : Parameters) (s : State Word) (plain : Bool) (i : TileIndex [p.block]) : Word :=
  FP.mul .ieeeValue (FP.sub .ieeeValue (bitWDY p s i)
    (if plain then FP.add .ieeeValue (FP.mul .ieeeValue (bitXhat p s plain i) (bitRowC1 p s plain))
      (bitC2 p s) else FP.mul .ieeeValue (bitXhat p s plain i) (bitRowC1 p s plain))) (bitScalar s p.rstd)
def bitDW (p : Parameters) (s : State Word) (plain : Bool) (i : TileIndex [p.block]) : Word :=
  FP.add .ieeeValue (FP.Value.zero .fp32) (FP.mul .ieeeValue (bitDY p s i) (bitXhat p s plain i))
def bitDB (p : Parameters) (s : State Word) (i : TileIndex [p.block]) : Word :=
  FP.add .ieeeValue (FP.Value.zero .fp32) (bitDY p s i)
def bitC1 (p : Parameters) (s : State Word) : Word :=
  bitAverage p (fun i => FP.mul .ieeeValue
    (bitLoaded p s p.xhat (s.pids 0*p.xhatStride) i) (bitWDY p s i))
def bitSuppliedDX (p : Parameters) (s : State Word) (plain : Bool) (i : TileIndex [p.block]) : Word :=
  FP.mul .ieeeValue (FP.sub .ieeeValue (bitWDY p s i)
    (if plain then FP.add .ieeeValue
      (FP.mul .ieeeValue (bitLoaded p s p.xhat (s.pids 0*p.xhatStride) i) (bitScalar s p.c1))
      (bitScalar s p.c2)
    else FP.mul .ieeeValue (bitLoaded p s p.xhat (s.pids 0*p.xhatStride) i) (bitScalar s p.c1)))
    (bitScalar s p.rstd)
def bitResidual (p : Parameters) (s : State Word) (i : TileIndex [p.block]) : Word :=
  FP.add .ieeeValue (bitLoaded p s p.dxBase (s.pids 0*p.dxStride) i)
    (bitLoaded p s p.dres (s.pids 0*p.dresStride) i)
def bitRowReference (p : Parameters) (s : State Word) (plain : Bool) : State Word :=
  let t := scatter p s p.dx p.dxStride (fun i => .mk .fp32 (bitDX p s plain i)) s
  let u := scatter p s p.dw p.columns (fun i => .mk .fp32 (bitDW p s plain i)) t
  if plain then scatter p s p.db p.columns (fun i => .mk .fp32 (bitDB p s i)) u else u

theorem wdy_closed (rest : Algebra Word) (p : Parameters) (s : State Word) :
    wdyValue (withRest rest p.columns) p s = bitWDY p s := by
  funext i
  simp [wdyValue, bitWDY, dyValue, bitDY, weight, bitWeight, loaded, bitLoaded, op, pad,
    withRest, dimensionLiteral_zero, show (0.0 : ℝ)=0 by norm_num,
    MatrixVectorMultip.Typed.backend, FP.Software.scalarFP32, FP.Software.fp32Context]

/- These bridge the *whole numerical reference* for every fallback algebra.
The final bits therefore have no residual dependence on unreachable callbacks. -/
theorem values_closed (rest : Algebra Word) (p : Parameters) (s : State Word) (plain : Bool) :
    rowDX (withRest rest p.columns) p s plain = bitDX p s plain ∧
    rowDW (withRest rest p.columns) p s plain = bitDW p s plain ∧
    rowDB (withRest rest p.columns) p s = bitDB p s ∧
    c1Value (withRest rest p.columns) p s = bitC1 p s ∧
    rowC2 (withRest rest p.columns) p s = bitC2 p s ∧
    suppliedDX (withRest rest p.columns) p s plain = bitSuppliedDX p s plain ∧
    residualValue (withRest rest p.columns) p s = bitResidual p s := by
  unfold rowDX rowDW rowDB c1Value rowC2 suppliedDX residualValue rowC1
  simp only [wdy_closed]
  simp only [funext_iff]
  simp [average, xhatValue, dyValue, suppliedXhat, loaded, scalar,
    op, zero, pad, bitDX, bitDW, bitDB, bitC1, bitC2, bitSuppliedDX, bitResidual,
    bitRowC1, bitAverage, bitWDY, bitXhat, bitDY, bitLoaded, bitWeight, bitScalar,
    withRest, dimensionLiteral_zero, dimensionLiteral_self,
    show (0.0 : ℝ) = 0 by norm_num,
    MatrixVectorMultip.Typed.backend, FP.Software.scalarFP32, FP.Software.fp32Context]

theorem stored_closed (rest : Algebra Word) (N : Nat) (v : Word) :
    stored (withRest rest N) v = .mk .fp32 v := by
  simp [stored, withRest, MatrixVectorMultip.Typed.backend]

theorem rowReference_closed (rest : Algebra Word) (p : Parameters) (s : State Word) (plain : Bool) :
    rowReference (withRest rest p.columns) p s plain = bitRowReference p s plain := by
  obtain ⟨hdx,hdw,hdb,_⟩ := values_closed rest p s plain
  simp only [rowReference, bitRowReference, hdx, hdw, hdb, stored_closed]

/-- Actual dimension conversion can round: no all-N exact-real assumption. -/
theorem dimension_rounds :
    dimensionLiteral 16777217 (16777217 : ℝ) = FP.Value.ofBits .fp32 0x4b800000 ∧
    FP.round .fp32 .ieeeValue (16777217 : Rat) ≠ FP.Value.ofBits .fp32 0x4b800001 := by
  have h := dimensionLiteral_self 16777217
  norm_num only [Nat.cast_ofNat] at h
  rw [h]
  decide +kernel

#axiomsClean dimensionLiteral_zero
#axiomsClean dimensionLiteral_self
#axiomsClean wdy_closed
#axiomsClean values_closed
#axiomsClean stored_closed
#axiomsClean rowReference_closed
#axiomsClean dimension_rounds
end
end VeriTile.Bench.TritonBenchG.LayerNormOps.FPClosed
