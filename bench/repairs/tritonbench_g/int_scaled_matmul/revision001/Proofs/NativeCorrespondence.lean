import bench.repairs.tritonbench_g.int_scaled_matmul.revision001.Proofs.Host

namespace VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.Native
open VeriTile.Triton FP.Structural FP.IntegerConversion Host
noncomputable section
set_option maxHeartbeats 4000000
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Pinned reference profile: nearest-even FP32, no input/result flush.
Actual source/native agreement with this profile remains an explicit local
numerical requirement; it is not supplied by a compiler/runtime assumption. -/
def sourceConfig : FP.Config := .ieeeValue

/-- The three source FP scalar primitives. Integer dot/update, same-format
loads and bit transport retain the existing explicit typed interpretation. -/
structure Primitives where
  promote32 : Int → Word
  multiply32 : Word → Word → Word
  signed32 : Word → Int

/-- Only scalar primitive laws. There is no execution, memory, reduction,
row or output result premise. Undefined float conversions are unrestricted. -/
structure PrimitiveCorrespondence (native : Primitives) : Prop where
  promotion : ∀ n, SignedRange 32 n → native.promote32 n = FP.round .fp32 sourceConfig (n : Rat)
  multiplication : ∀ a b, native.multiply32 a b = FP.mul sourceConfig a b
  conversion : ∀ a z, toSigned 32 a = some z → native.signed32 a = z

/-- The total interpretation outside the signed32 input type is merely a
mathematical completion. The source accumulator is wrapped to signed32 before
promotion; no hardware behavior outside that primitive's input type is claimed. -/
def promote (native : Primitives) (n : Int) : Word :=
  if SignedRange 32 n then native.promote32 n else FP.round .fp32 sourceConfig (n : Rat)

def fallback (native : Primitives) : Nat → Word → Int := fun _ x => native.signed32 x

theorem accumulator_domain (p : ScaledWord.Parameters) (s : State Word) (n : Nat) (i : ScaledWord.CIndex p) :
    SignedRange 32 (ScaledWord.accumulation p s n i) := by
  rw [ScaledWord.accumulation_value]
  exact ⟨by decide, BitVec.le_toInt _, BitVec.toInt_lt⟩

/-- Every source accumulator promotion uses the actual native port, including
masked lanes; the mathematical completion is never selected here. -/
theorem promote_accumulator (native : Primitives) (p : ScaledWord.Parameters) (s : State Word)
    (n : Nat) (i : ScaledWord.CIndex p) :
    promote native (ScaledWord.accumulation p s n i) = native.promote32 (ScaledWord.accumulation p s n i) := by
  simp only [promote,if_pos (accumulator_domain p s n i)]

/-- Source interpreter with actual scalar ports at promotion, multiplication
and signed32 conversion. All other channels are the already proved typed
integer/memory interpretation. The undefined conversion result stays native. -/
def backend (rest : Algebra Word) (native : Primitives) : Algebra Word :=
  let base := ScaledWord.backend rest sourceConfig (fallback native)
  { base with
    fromInt := fun p n => if p = some .fp32 then promote native n else base.fromInt p n
    binary := fun p d op a b =>
      if FP.Software.fp32Context p d && op == .mul then native.multiply32 a b else base.binary p d op a b
    floatToInt := fun p d w =>
      if FP.IntegerConversion.fp32Source p d && w == 32 then some (fun x => some (native.signed32 x))
      else base.floatToInt p d w }

theorem promote_eq (native : Primitives) (h : PrimitiveCorrespondence native) (n : Int) :
    promote native n = FP.round .fp32 sourceConfig (n : Rat) := by
  unfold promote
  split_ifs with hn
  · exact h.promotion n hn
  · rfl

theorem signed_eq_total (native : Primitives) (h : PrimitiveCorrespondence native) (x : Word) :
    native.signed32 x = totalSigned native.signed32 32 x := by
  cases hx : toSigned 32 x with
  | none => simp [totalSigned,hx]
  | some z => simpa [totalSigned,hx] using h.conversion x z hx

/-- Full interpreter equality is derived from the three local laws, not
assumed as a numerical leaf. The arbitrary fallback agrees with the actual
native undefined result; active values are separately required to be defined. -/
theorem backend_eq (rest : Algebra Word) (native : Primitives) (h : PrimitiveCorrespondence native) :
    backend rest native = ScaledWord.backend rest sourceConfig (fallback native) := by
  let base := ScaledWord.backend rest sourceConfig (fallback native)
  have hp : (fun p n => if p = some .fp32 then promote native n else base.fromInt p n) = base.fromInt := by
    funext p n
    by_cases he : p = some .fp32
    · simp [he,base,ScaledWord.backend,promote_eq native h]
    · simp [he]
  have hm : (fun (p : Option ComputeDType) (d : FloatDType) (op : Binary) (a b : Word) => if FP.Software.fp32Context p d && op == .mul then
      native.multiply32 a b else base.binary p d op a b) = base.binary := by
    funext p d op a b
    by_cases he : (FP.Software.fp32Context p d && op == .mul) = Bool.true
    · have hh : FP.Software.fp32Context p d = Bool.true ∧ op = .mul := by simpa using he
      have hc := hh.1
      have ho := hh.2
      subst op
      simp [hc,base,ScaledWord.backend,FP.IntegerConversion.fp32TotalBackend,FP.Software.scalarFP32,h.multiplication]
    · simp [he]
  have hc : (fun (p : Option IndexComputePrecision) (d : FloatDType) (w : Nat) => if FP.IntegerConversion.fp32Source p d && w == 32 then
      some (fun x => some (native.signed32 x)) else base.floatToInt p d w) = base.floatToInt := by
    funext p d w
    by_cases he : (FP.IntegerConversion.fp32Source p d && w == 32) = Bool.true
    · have hh : FP.IntegerConversion.fp32Source p d = Bool.true ∧ w = 32 := by simpa using he
      have hp := hh.1
      have hw := hh.2
      subst w
      simp [hp,base,ScaledWord.backend,FP.IntegerConversion.fp32TotalBackend,fallback]
      funext x
      exact congrArg some (signed_eq_total native h x)
    · simp [he]
  change { base with fromInt := _, binary := _, floatToInt := _ } = base
  rw [hp,hm,hc]

def one (rest : Algebra Word) (native : Primitives) (p : ScaledWord.Parameters) (pid : Nat) (s : State Word) :=
  FP.Structural.exec (backend rest native) (ScaledWord.source p) (GridExecution.atProgram s pid)
def grid (rest : Algebra Word) (native : Primitives) (p : ScaledWord.Parameters) (s : State Word) :=
  GridExecution.sequence (one rest native p) (List.range (Geometry.count (ScaledWord.geometry p))) s
def call (rest : Algebra Word) (native : Primitives) (x : Arguments) (scale : Tensor) (s : State Word) :=
  if shapeCompatible x then
    if emptyOutput x then some (x.c,s)
    else (grid rest native (scaledParameters x scale) s).map (fun final => (x.c,final))
  else none

theorem call_eq (rest : Algebra Word) (native : Primitives) (h : PrimitiveCorrespondence native)
    (x : Arguments) (scale : Tensor) (s : State Word) :
    call rest native x scale s = Host.int_scaled_matmul_kernel rest sourceConfig (fallback native) x scale s := by
  have ho (p : ScaledWord.Parameters) : one rest native p = ScaledWord.one rest sourceConfig (fallback native) p := by
    funext pid s
    unfold one ScaledWord.one
    rw [backend_eq rest native h]
  simp only [call,grid,Host.int_scaled_matmul_kernel,ScaledWord.grid,ho]

#axiomsClean promote_eq
#axiomsClean accumulator_domain
#axiomsClean promote_accumulator
#axiomsClean signed_eq_total
#axiomsClean backend_eq
#axiomsClean call_eq
end
end VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.Native
