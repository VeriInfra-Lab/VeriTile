import bench.tritonbench_g.matmul_dequant_int4.Sources.Physical
import bench.tritonbench_g.matmul_dequant_int4.Sources.Helpers
import bench.tritonbench_g.matmul_dequant_int4.Sources.S1
namespace VeriTile.Bench.Repairs.MatmulDequantInt4.Native
open Physical
set_option maxHeartbeats 4000000
inductive Conversion where
  | signed | unsigned
  deriving DecidableEq, Repr
/-- Source operation tag. Tensor / Python scalar 15 may use a different CUDA
implementation from Tensor/Tensor division; no bitwise identification is assumed. -/
inductive Division where
  | scalarDenom15 | tensor
  deriving DecidableEq, Repr
/-- Local scalar families retain their carrier width. Conversion mode records
standalone signed differences versus the fused compiler's unsigned nibbles.
All integer conversions interpret the finite machine argument representation. -/
structure Instructions where
  convert16 : Conversion → Int → Half
  neg16 : Half → Half
  add16 : Half → Half → Half
  sub16 : Half → Half → Half
  mul16 : Half → Half → Half
  div16 : Division → Half → Half → Half
  max16 : Half → Half → Half
  min16 : Half → Half → Half
  clipLowerZero16 : Half → Half
  clipUpperZero16 : Half → Half
  roundClamp16 : Half → BitVec 4
  convert32 : Conversion → Int → Word
  neg32 : Word → Word
  add32 : Word → Word → Word
  sub32 : Word → Word → Word
  mul32 : Word → Word → Word
  div32 : Division → Word → Word → Word
  max32 : Word → Word → Word
  min32 : Word → Word → Word
  clipLowerZero32 : Word → Word
  clipUpperZero32 : Word → Word
  roundClamp32 : Word → BitVec 4
  fma16 : Half → Half → Half → Half
  fma32 : Word → Word → Word → Word
  narrow : Word → Half
  widen : Half → Word
  mma16 : Library.MMAKind → (Fin 16 → Half) → (Fin 16 → Half) → Word → Word

def dequant16 (p : Instructions) : DequantOps Half := ⟨p.convert16 .signed,p.mul16⟩
def dequant32 (p : Instructions) : DequantOps Word := ⟨p.convert32 .signed,p.mul32⟩
def fused (p : Instructions) (kind : Library.MMAKind) : FusedOps Half Word :=
  ⟨0,0,p.convert16 .unsigned,p.mul16,p.sub16,p.neg16,p.fma16,p.mma16 kind,p.narrow⟩
def library (p : Instructions) : Library.Ops Half Word :=
  ⟨0,0,p.widen,p.narrow,p.mul32,p.add32,p.fma32,p.mma16⟩
def quant16 (p : Instructions) : Helpers.Arithmetic Half :=
  ⟨0,(fun x => p.div16 .scalarDenom15 x 0x4b80),p.convert16 .signed,p.neg16,p.add16,p.sub16,p.mul16,p.div16 .tensor,
    p.max16,p.min16,p.clipLowerZero16,p.clipUpperZero16,p.roundClamp16⟩
def quant32 (p : Instructions) : Helpers.Arithmetic Word :=
  ⟨0,(fun x => p.div32 .scalarDenom15 x 0x41700000),p.convert32 .signed,p.neg32,p.add32,p.sub32,p.mul32,p.div32 .tensor,
    p.max32,p.min32,p.clipLowerZero32,p.clipUpperZero32,p.roundClamp32⟩

/-- Twenty-seven tagged family containers: twenty-six scalar families and one fixed K16 ISA family.
Division families quantify both source-operation tags independently. No matrix dimensions,
addresses, memory state, variable-length reduction or complete-output port. -/
structure Correspondence (a r : Instructions) : Prop where
  convert16 : ∀ x y,a.convert16 x y=r.convert16 x y
  neg16 : ∀ x,a.neg16 x=r.neg16 x
  add16 : ∀ x y,a.add16 x y=r.add16 x y
  sub16 : ∀ x y,a.sub16 x y=r.sub16 x y
  mul16 : ∀ x y,a.mul16 x y=r.mul16 x y
  div16 : ∀ tag x y,a.div16 tag x y=r.div16 tag x y
  max16 : ∀ x y,a.max16 x y=r.max16 x y
  min16 : ∀ x y,a.min16 x y=r.min16 x y
  clipLowerZero16 : ∀ x,a.clipLowerZero16 x=r.clipLowerZero16 x
  clipUpperZero16 : ∀ x,a.clipUpperZero16 x=r.clipUpperZero16 x
  roundClamp16 : ∀ x,a.roundClamp16 x=r.roundClamp16 x
  convert32 : ∀ x y,a.convert32 x y=r.convert32 x y
  neg32 : ∀ x,a.neg32 x=r.neg32 x
  add32 : ∀ x y,a.add32 x y=r.add32 x y
  sub32 : ∀ x y,a.sub32 x y=r.sub32 x y
  mul32 : ∀ x y,a.mul32 x y=r.mul32 x y
  div32 : ∀ tag x y,a.div32 tag x y=r.div32 tag x y
  max32 : ∀ x y,a.max32 x y=r.max32 x y
  min32 : ∀ x y,a.min32 x y=r.min32 x y
  clipLowerZero32 : ∀ x,a.clipLowerZero32 x=r.clipLowerZero32 x
  clipUpperZero32 : ∀ x,a.clipUpperZero32 x=r.clipUpperZero32 x
  roundClamp32 : ∀ x,a.roundClamp32 x=r.roundClamp32 x
  fma16 : ∀ x y z,a.fma16 x y z=r.fma16 x y z
  fma32 : ∀ x y z,a.fma32 x y z=r.fma32 x y z
  narrow : ∀ x,a.narrow x=r.narrow x
  widen : ∀ x,a.widen x=r.widen x
  mma16 : ∀ x y z w,a.mma16 x y z w=r.mma16 x y z w

theorem correspondence_eq (a r : Instructions) (h : Correspondence a r) : a=r := by
  cases a;cases r
  simp only [Instructions.mk.injEq]
  exact ⟨funext (fun x => funext (fun y => h.convert16 x y)),
    funext (fun x => h.neg16 x),
    funext (fun x => funext (fun y => h.add16 x y)),
    funext (fun x => funext (fun y => h.sub16 x y)),
    funext (fun x => funext (fun y => h.mul16 x y)),
    funext (fun tag => funext (fun x => funext (fun y => h.div16 tag x y))),
    funext (fun x => funext (fun y => h.max16 x y)),
    funext (fun x => funext (fun y => h.min16 x y)),
    funext (fun x => h.clipLowerZero16 x),
    funext (fun x => h.clipUpperZero16 x),
    funext (fun x => h.roundClamp16 x),
    funext (fun x => funext (fun y => h.convert32 x y)),
    funext (fun x => h.neg32 x),
    funext (fun x => funext (fun y => h.add32 x y)),
    funext (fun x => funext (fun y => h.sub32 x y)),
    funext (fun x => funext (fun y => h.mul32 x y)),
    funext (fun tag => funext (fun x => funext (fun y => h.div32 tag x y))),
    funext (fun x => funext (fun y => h.max32 x y)),
    funext (fun x => funext (fun y => h.min32 x y)),
    funext (fun x => h.clipLowerZero32 x),
    funext (fun x => h.clipUpperZero32 x),
    funext (fun x => h.roundClamp32 x),
    funext (fun x => funext (fun y => funext (fun z => h.fma16 x y z))),
    funext (fun x => funext (fun y => funext (fun z => h.fma32 x y z))),
    funext (fun x => h.narrow x),
    funext (fun x => h.widen x),
    funext (fun x => funext (fun y => funext (fun z => funext (fun w => h.mma16 x y z w))))⟩

theorem reference_consistent (r : Instructions) : Correspondence r r :=
  ⟨fun _ _ => rfl,
   fun _ => rfl,
   fun _ _ => rfl,
   fun _ _ => rfl,
   fun _ _ => rfl,
   fun _ _ _ => rfl,
   fun _ _ => rfl,
   fun _ _ => rfl,
   fun _ => rfl,
   fun _ => rfl,
   fun _ => rfl,
   fun _ _ => rfl,
   fun _ => rfl,
   fun _ _ => rfl,
   fun _ _ => rfl,
   fun _ _ => rfl,
   fun _ _ _ => rfl,
   fun _ _ => rfl,
   fun _ _ => rfl,
   fun _ => rfl,
   fun _ => rfl,
   fun _ => rfl,
   fun _ _ _ => rfl,
   fun _ _ _ => rfl,
   fun _ => rfl,
   fun _ => rfl,
   fun _ _ _ _ => rfl⟩

theorem dequant16_congruence (s : Shape) (order : List Work) (initial : State Half)
    (a r : Instructions) (h : Correspondence a r) :
    dequantRun s (dequant16 a) order initial=dequantRun s (dequant16 r) order initial := by
  rw [correspondence_eq a r h]
theorem dequant32_congruence (s : Shape) (order : List Work) (initial : State Word)
    (a r : Instructions) (h : Correspondence a r) :
    dequantRun s (dequant32 a) order initial=dequantRun s (dequant32 r) order initial := by
  rw [correspondence_eq a r h]
theorem fused_congruence (s : Shape) (t : Tile) (ng : Bool) (kind : Library.MMAKind)
    (fusion : Nat → Nat → Fusion) (order : List Work) (initial : State Half)
    (a r : Instructions) (h : Correspondence a r) :
    fusedRun s t (fused a kind) ng fusion order initial=fusedRun s t (fused r kind) ng fusion order initial := by
  rw [correspondence_eq a r h]
theorem s1_congruence (s : Shape) (plan : Library.Selected s) (dequantOrder outputOrder : List Work)
    (initial : State Half) (a r : Instructions) (h : Correspondence a r) :
    S1.run s (dequant16 a) (library a) plan dequantOrder outputOrder initial=
      S1.run s (dequant16 r) (library r) plan dequantOrder outputOrder initial := by
  rw [correspondence_eq a r h]

#axiomsClean correspondence_eq
#axiomsClean reference_consistent
#axiomsClean dequant16_congruence
#axiomsClean dequant32_congruence
#axiomsClean fused_congruence
#axiomsClean s1_congruence
end VeriTile.Bench.Repairs.MatmulDequantInt4.Native
