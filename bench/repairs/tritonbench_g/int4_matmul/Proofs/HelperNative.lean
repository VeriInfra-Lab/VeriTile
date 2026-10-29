import bench.repairs.tritonbench_g.int4_matmul.Proofs.HelperBytes
namespace VeriTile.Bench.Repairs.Int4Matmul.HelperNative
open Helpers HelperBytes Physical
/-- All deferred helper operations are scalar. Zero and fifteen are fixed FP32 words. -/
structure Instructions where
  fromInt : Int → Word
  neg : Word → Word
  add : Word → Word → Word
  sub : Word → Word → Word
  mul : Word → Word → Word
  div : Word → Word → Word
  max : Word → Word → Word
  min : Word → Word → Word
  clipLowerZero : Word → Word
  clipUpperZero : Word → Word
  roundClamp : Word → BitVec 4
  narrow : Word → Half
  widen : Half → Word

def arithmetic (p : Instructions) : Arithmetic Word :=
  ⟨0,0x41700000,p.fromInt,p.neg,p.add,p.sub,p.mul,p.div,p.max,p.min,p.clipLowerZero,p.clipUpperZero,p.roundClamp⟩
def halfArithmetic (p : Instructions) : HalfArithmetic Word Half :=
  ⟨p.fromInt,p.narrow,p.widen,p.sub,p.mul⟩
structure Correspondence (a r : Instructions) : Prop where
  fromInt : ∀ z,a.fromInt z=r.fromInt z
  neg : ∀ x,a.neg x=r.neg x
  add : ∀ x y,a.add x y=r.add x y
  sub : ∀ x y,a.sub x y=r.sub x y
  mul : ∀ x y,a.mul x y=r.mul x y
  div : ∀ x y,a.div x y=r.div x y
  max : ∀ x y,a.max x y=r.max x y
  min : ∀ x y,a.min x y=r.min x y
  clipLowerZero : ∀ x,a.clipLowerZero x=r.clipLowerZero x
  clipUpperZero : ∀ x,a.clipUpperZero x=r.clipUpperZero x
  roundClamp : ∀ x,a.roundClamp x=r.roundClamp x
  narrow : ∀ x,a.narrow x=r.narrow x
  widen : ∀ x,a.widen x=r.widen x

theorem correspondence_eq (a r : Instructions) (h : Correspondence a r) : a=r := by
  cases a;cases r
  have h1 := funext h.fromInt
  have h2 := funext h.neg
  have h3 := funext (fun x => funext (h.add x))
  have h4 := funext (fun x => funext (h.sub x))
  have h5 := funext (fun x => funext (h.mul x))
  have h6 := funext (fun x => funext (h.div x))
  have h7 := funext (fun x => funext (h.max x))
  have h8 := funext (fun x => funext (h.min x))
  have h9 := funext h.clipLowerZero
  have h10 := funext h.clipUpperZero
  have h11 := funext h.roundClamp
  have h12 := funext h.narrow
  have h13 := funext h.widen
  congr

theorem quantizer_congruence (s : Shape) (order : ReductionOrder s.group) (initial : Bytes)
    (scratch : Fin 3 → Nat → Byte) (a r : Instructions) (h : Correspondence a r) :
    quantizeBytes s (arithmetic a) order initial scratch=quantizeBytes s (arithmetic r) order initial scratch := by
  rw [correspondence_eq a r h]
theorem unpack_congruence (s : Shape) (initial : Bytes) (scratch : Nat → Byte)
    (a r : Instructions) (h : Correspondence a r) :
    unpack s (halfArithmetic a) initial scratch=unpack s (halfArithmetic r) initial scratch := by
  rw [correspondence_eq a r h]
theorem reference_consistent (r : Instructions) : Correspondence r r :=
  ⟨fun _ => rfl,fun _ => rfl,fun _ _ => rfl,fun _ _ => rfl,fun _ _ => rfl,
   fun _ _ => rfl,fun _ _ => rfl,fun _ _ => rfl,fun _ => rfl,fun _ => rfl,
   fun _ => rfl,fun _ => rfl,fun _ => rfl⟩
#axiomsClean correspondence_eq
#axiomsClean quantizer_congruence
#axiomsClean unpack_congruence
#axiomsClean reference_consistent
end VeriTile.Bench.Repairs.Int4Matmul.HelperNative
