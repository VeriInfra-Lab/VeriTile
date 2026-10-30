import bench.repairs.tritonbench_g.dequantize_matmul.Proofs.Logical

namespace VeriTile.Bench.Repairs.DequantizeMatmul
open VeriTile.Triton
abbrev Word := FP.Value .fp32

/-- Software cvt.rn.tf32.f32 value profile: nearest-even removal of 13
fraction bits, represented in the FP32 container. Native conformance remains
one scalar-conversion obligation, including special/subnormal behavior. -/
def tf32RN (x : Word) : Word :=
  if x.isNaN then FP.Value.qNaN .fp32
  else if x.isInf then x
  else
    let high := x.raw/8192
    let low := x.raw%8192
    let increment := low>4096 || (low==4096 && high%2==1)
    FP.Value.ofBits .fp32 ((high+(if increment then 1 else 0))*8192)

/-- Software cvt.rna.tf32.f32 profile, with halfway cases rounded away
from zero. The selected plan fixes RN versus RNA before operand evaluation. -/
def tf32RNA (x : Word) : Word :=
  if x.isNaN then FP.Value.qNaN .fp32
  else if x.isInf then x
  else FP.Value.ofBits .fp32 ((x.raw/8192+(if x.raw%8192≥4096 then 1 else 0))*8192)
def tf32Convert : TF32Mode → Word → Word
  | .nearestEven => tf32RN
  | .nearestAway => tf32RNA

/-- Only an actual ISA instruction's one result lane may occupy this port.
Both operands have already undergone explicit the selected tf32Convert conversion. No complete
matrix, arbitrary reduction length, or entire output vector enters this port. -/
structure FixedInstructionProfile where
  mmaSyncM16N8K8RowColF32TF32TF32F32 : (Fin 8 → Word) → (Fin 8 → Word) → Word → Word
  wgmmaM64N8K8F32TF32TF32 : (Fin 8 → Word) → (Fin 8 → Word) → Word → Word

def FixedInstructionProfile.lane (p : FixedInstructionProfile) :
    MMAKind → (Fin 8 → Word) → (Fin 8 → Word) → Word → Word
  | .mmaSyncM16N8K8RowColF32TF32TF32F32 => p.mmaSyncM16N8K8RowColF32TF32TF32F32
  | .wgmmaM64N8K8F32TF32TF32 => p.wgmmaM64N8K8F32TF32TF32

/-- Same-format load/store transport is bitwise identity, not a numeric oracle. -/
def fpArithmetic (instruction : FixedInstructionProfile) : Arithmetic Word where
  zero := FP.Value.zero .fp32
  signed8 := fun q => FP.round .fp32 FP.Config.ieeeValue (Rat.ofInt q.toInt)
  load := id
  store := id
  mul := FP.mul FP.Config.ieeeValue
  add := FP.add FP.Config.ieeeValue
  fma := FP.fma FP.Config.ieeeValue
  tf32 := tf32Convert
  mma8 := instruction.lane

structure NumericLeaves (native : Arithmetic Word) (instruction : FixedInstructionProfile) : Prop where
  zero : native.zero=FP.Value.zero .fp32
  signed8 : ∀ q,native.signed8 q=FP.round .fp32 FP.Config.ieeeValue (Rat.ofInt q.toInt)
  load : native.load=id
  store : native.store=id
  mul : ∀ x y,native.mul x y=FP.mul FP.Config.ieeeValue x y
  add : ∀ x y,native.add x y=FP.add FP.Config.ieeeValue x y
  fma : ∀ x y z,native.fma x y z=FP.fma FP.Config.ieeeValue x y z
  tf32 : ∀ mode x,native.tf32 mode x=tf32Convert mode x
  mma : ∀ kind xs ys z,native.mma8 kind xs ys z=instruction.lane kind xs ys z

theorem NumericLeaves.correspondence (h : NumericLeaves native instruction) :
    LocalCorrespondence native (fpArithmetic instruction) := by
  exact ⟨h.zero,h.signed8,congrFun h.load,congrFun h.store,h.mul,h.add,h.fma,h.tf32,h.mma⟩

noncomputable section
/-- Entire returned matrix follows from scalar/fixed-instruction leaves and
structural K coverage. This is bitwise reference equality, not equality to Real. -/
theorem fp_complete (native : Arithmetic Word) (instruction : FixedInstructionProfile)
    (numeric : NumericLeaves native instruction) (x : Input Word s) (b : Buffers)
    (tile : Tile) (plan : LibraryPlan s) (sep : Separate b)
    (initial : Memory (Cell Word)) (rep : Represents x b initial)
    (r : Fin s.M) (n : Fin s.N) :
    run native s b tile plan initial (b.out.addr (r.val*s.N+n.val))=
      .fp32 (wordReference (fpArithmetic instruction) x plan r n) := by
  rw [numeric.correspondence.arithmetic_eq]
  rw [output _ _ _ _ _ sep _ _ _ r.isLt n.isLt]
  rw [result_words _ x b plan initial rep r n]

#axiomsClean NumericLeaves.correspondence
#axiomsClean fp_complete
end
end VeriTile.Bench.Repairs.DequantizeMatmul

namespace VeriTile.Bench.Repairs.DequantizeMatmul
open VeriTile.Triton
/-- Native numeric leaves exclude same-format transport and literal zero:
those operations are constructed as the fixed bitwise identities/constants. -/
structure PrimitiveOps where
  signed8 : BitVec 8 → Word
  mul : Word → Word → Word
  add : Word → Word → Word
  fma : Word → Word → Word → Word
  tf32 : TF32Mode → Word → Word
  instruction : MMAKind → (Fin 8 → Word) → (Fin 8 → Word) → Word → Word

def PrimitiveOps.arithmetic (p : PrimitiveOps) : Arithmetic Word :=
  ⟨FP.Value.zero .fp32,p.signed8,id,id,p.mul,p.add,p.fma,p.tf32,p.instruction⟩
structure PrimitiveLeaves (p : PrimitiveOps) (reference : FixedInstructionProfile) : Prop where
  signed8 : ∀ q,p.signed8 q=FP.round .fp32 FP.Config.ieeeValue (Rat.ofInt q.toInt)
  mul : ∀ x y,p.mul x y=FP.mul FP.Config.ieeeValue x y
  add : ∀ x y,p.add x y=FP.add FP.Config.ieeeValue x y
  fma : ∀ x y z,p.fma x y z=FP.fma FP.Config.ieeeValue x y z
  tf32 : ∀ mode x,p.tf32 mode x=tf32Convert mode x
  instruction : ∀ kind xs ys z,p.instruction kind xs ys z=reference.lane kind xs ys z

theorem PrimitiveLeaves.numeric (h : PrimitiveLeaves p reference) :
    NumericLeaves p.arithmetic reference :=
  ⟨rfl,h.signed8,rfl,rfl,h.mul,h.add,h.fma,h.tf32,h.instruction⟩
#axiomsClean PrimitiveLeaves.numeric
end VeriTile.Bench.Repairs.DequantizeMatmul
