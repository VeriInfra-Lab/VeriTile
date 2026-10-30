import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.Core

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization
noncomputable section
open VeriTile.Triton
abbrev FixedISAProfile := Int8DequantMatmul.FixedISAProfile
/-- Each remaining port is one scalar instruction result. Triton div.full.f32
and rzi.s16-to-byte are deliberately not replaced by unjustified IEEE divide
or an all-word saturating conversion. A separate profile is instantiated for
Torch's propagating amax/division/cast path. No reduction/output port exists. -/
structure ScalarProfile where
  widen : Word16 → Word32
  add : Word32 → Word32 → Word32
  absolute : Word32 → Word32
  maximum : Word32 → Word32 → Word32
  divide : Word32 → Word32 → Word32
  scale127 : Word32 → Word32
  integer8 : Word32 → Byte
def fpOps (isa : FixedISAProfile) (scalar : ScalarProfile) : Ops (BitVec 32) Word32 Word16 where
  toOps := Int8DequantMatmul.fpOps isa
  widen := scalar.widen
  add := scalar.add
  absolute := scalar.absolute
  maximum := scalar.maximum
  divide := scalar.divide
  scale127 := scalar.scale127
  integer8 := scalar.integer8
structure PrimitiveLeaves (a : Ops (BitVec 32) Word32 Word16)
    (isa : FixedISAProfile) (scalar : ScalarProfile) : Prop where
  base : Int8DequantMatmul.PrimitiveLeaves a.toOps isa
  widen : ∀ x,a.widen x=scalar.widen x
  add : ∀ x y,a.add x y=scalar.add x y
  absolute : ∀ x,a.absolute x=scalar.absolute x
  maximum : ∀ x y,a.maximum x y=scalar.maximum x y
  divide : ∀ x y,a.divide x y=scalar.divide x y
  scale127 : ∀ x,a.scale127 x=scalar.scale127 x
  integer8 : ∀ x,a.integer8 x=scalar.integer8 x
theorem PrimitiveLeaves.eq (h : PrimitiveLeaves a isa scalar) : a=fpOps isa scalar := by
  cases a
  simp only [fpOps,Ops.mk.injEq]
  exact ⟨h.base.eq,funext h.widen,funext (fun x => funext (h.add x)),funext h.absolute,
    funext (fun x => funext (h.maximum x)),funext (fun x => funext (h.divide x)),
    funext h.scale127,funext h.integer8⟩
#axiomsClean PrimitiveLeaves.eq
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization
