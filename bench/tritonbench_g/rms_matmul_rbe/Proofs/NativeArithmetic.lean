import bench.tritonbench_g.rms_matmul_rbe.Proofs.ByteMemory
import bench.tritonbench_g.rms_matmul_rbe.Proofs.StoreTrace

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.Native
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

abbrev Word := BitVec 32
abbrev Half := BitVec 16
abbrev E5 := BitVec 8

/-- Numerical hardware/library boundaries are single scalar operations or
one output lane of the fixed m16n8k8 TF32 instruction. Their arguments and
results are raw words, retaining infinities, NaNs, signed zero and payloads.
The K64 dot, reduction tree, K loop and rotary composition are not ports. -/
structure Primitives where
  addRN : Word → Word → Word
  subRN : Word → Word → Word
  mulRN : Word → Word → Word
  mulFTZ : Word → Word → Word
  divFull : Word → Word → Word
  rsqrtApproxFTZ : Word → Word
  neg : Word → Word
  fmaRN : Word → Word → Word → Word
  cvtIntRN : Int → Word
  libdevicePow : Word → Word → Word
  libdevicePowi : Word → Int → Word
  libdeviceCos : Word → Word
  libdeviceSin : Word → Word
  mmaTF32K8 : (Fin 8 → Word) → (Fin 8 → Word) → Word → Word
  literalFinite : ℝ → Word
  literalInteger : Int → Word
  toHalfRN : Word → Half
  fromHalf : Half → Word
  toE5 : Word → E5
  fromE5 : E5 → Word
  e5ToHalf : E5 → Half

def pack16 (x : Half) : Word := BitVec.ofNat 32 x.toNat
def pack8 (x : E5) : Word := BitVec.ofNat 32 x.toNat
def unpack16 (x : Word) : Half := BitVec.ofNat 16 x.toNat
def unpack8 (x : Word) : E5 := BitVec.ofNat 8 x.toNat

/-- Typed scalar conversion, preserving the two different E5 input chains.
Narrow values remain encoded words until an explicit source cast widens them. -/
def cast (p : Primitives) (src dst : FloatDType) (x : Word) : Word :=
  if src=dst then x
  else if dst=.real then
    if src=.fp16 then p.fromHalf (unpack16 x)
    else if src=.f8e5 then p.fromE5 (unpack8 x) else x
  else if dst=.fp16 then
    if src=.f8e5 then pack16 (p.e5ToHalf (unpack8 x)) else pack16 (p.toHalfRN x)
  else if dst=.f8e5 then pack8 (p.toE5 x) else x

def literal (p : Primitives) (d : FloatDType) (x : ℝ) : Word :=
  cast p .real d (p.literalFinite x)
def e5Bits (p : Primitives) (d : FloatDType) (x : Float8E5Bits) : Word :=
  cast p .f8e5 d (BitVec.ofNat 32 x.val)

def arithmetic (p : Primitives) : Arithmetic Word where
  zero := 0
  one := 0x3f800000
  add := p.addRN
  sub := p.subRN
  mul := p.mulRN
  mulFTZ := p.mulFTZ
  divide := p.divFull
  rsqrt := p.rsqrtApproxFTZ
  neg := p.neg
  fma := p.fmaRN
  ofInt := p.cvtIntRN
  pow := p.libdevicePow
  powi := p.libdevicePowi
  cos := p.libdeviceCos
  sin := p.libdeviceSin
  mma8 := p.mmaTF32K8
  literalNat := fun n => p.literalFinite (n:ℝ)
  literal := literal p
  cast := cast p
  e5Bits := e5Bits p
  fp32Bits := Float32Bits.bits
  plan := RmsMatmulRbe.orderedK8
  reduction := RmsMatmulRbe.butterfly64
  natLiteral := fun _ => rfl

/-- EPS Python bool/int/finite-float materializations retain their kinds.
The finite-float scalar port permits FP32 underflow and overflow. Runtime K
and positions use cvtIntRN; HEAD_DIM uses compile-time literalFinite. -/
def materialization (p : Primitives) : ScalarMaterialization Word where
  boolean := fun b => if b then 0x3f800000 else 0
  integer := p.literalInteger
  floating := p.literalFinite

/-- Five primitive zero facts suffice for masked-load realization. There is
no finite-input predicate, whole-dot equality or output condition here. -/
structure ZeroRealization (p : Primitives) : Prop where
  literal : p.literalFinite 0=0
  halfNarrow : p.toHalfRN 0=0
  halfWiden : p.fromHalf 0=0
  e5Half : p.e5ToHalf 0=0
  e5Single : p.fromE5 0=0

theorem zero_laws (p : Primitives) (h : ZeroRealization p) : ZeroLaws (arithmetic p) := by
  constructor
  · exact h.literal
  · change p.fromHalf (unpack16 (pack16 (p.toHalfRN (p.literalFinite 0))))=0
    rw [h.literal,h.halfNarrow]
    exact h.halfWiden
  · change p.fromHalf (unpack16 (pack16 (p.e5ToHalf 0)))=0
    rw [h.e5Half]
    exact h.halfWiden
  · change p.fromHalf (unpack16 (pack16 (p.toHalfRN (p.fromE5 0))))=0
    rw [h.e5Single,h.halfNarrow]
    exact h.halfWiden

theorem output_cast_half (p : Primitives) (x : Word) :
    PhysicalKernel.castOutput (arithmetic p) Bool.false x=pack16 (p.toHalfRN x) := rfl
theorem output_cast_e5 (p : Primitives) (x : Word) :
    PhysicalKernel.castOutput (arithmetic p) Bool.true x=pack8 (p.toE5 x) := rfl

theorem output_storable (p : Primitives) : PhysicalKernel.OutputStorable PhysicalKernel.Bytes.access (arithmetic p) := by
  intro fp8 x
  cases fp8
  · exact PhysicalKernel.Bytes.narrow16_storable (p.toHalfRN x)
  · exact PhysicalKernel.Bytes.narrowE5_storable (p.toE5 x)

theorem k8_port (p : Primitives) (x y : Fin 64 → Word) (seed : Word) :
    chain (arithmetic p) x y seed=RmsMatmulRbe.orderedK8.instructions.foldl
      (fun old lanes => p.mmaTF32K8 (fun j => x (lanes j)) (fun j => y (lanes j)) old) seed :=
  chain_plan (arithmetic p) x y seed

theorem reduction_port (p : Primitives) (x : Fin 64 → Word) :
    reduceLine (arithmetic p) x=RmsMatmulRbe.butterfly64.tree.eval p.addRN x :=
  reduction_plan (arithmetic p) x

#axiomsClean zero_laws
#axiomsClean output_storable
#axiomsClean k8_port
#axiomsClean reduction_port
#specNonCircular Primitives avoiding [arithmetic,rawTile,nativeRawTile,execution,PhysicalKernel.program]
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.Native
