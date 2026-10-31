import bench.tritonbench_g.bgmv_expand_slice.Tensor
import bench.tritonbench_g.bgmv_expand_slice.SignedGrid
import bench.tritonbench_g.bgmv_expand_slice.Proofs.Direct.FPScalarTree
namespace VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.Native
open VeriTile Triton FP.Structural Direct
noncomputable section
set_option maxHeartbeats 4000000

/-- Fixed-arity instructions actually needed by the source's selected dtypes.
Half words occupy the low 16 bits; the ports retain all input bit patterns. -/
structure Instructions where
  narrow : Word → BitVec 16
  widen : BitVec 16 → Word
  mul16 : BitVec 16 → BitVec 16 → BitVec 16
  add16 : BitVec 16 → BitVec 16 → BitVec 16
  add32 : Word → Word → Word

structure LocalConformance (actual selected : Instructions) : Prop where
  narrow : ∀ x,actual.narrow x=selected.narrow x
  widen : ∀ x,actual.widen x=selected.widen x
  mul16 : ∀ x y,actual.mul16 x y=selected.mul16 x y
  add16 : ∀ x y,actual.add16 x y=selected.add16 x y
  add32 : ∀ x y,actual.add32 x y=selected.add32 x y

theorem instructions_eq (actual selected : Instructions) (h : LocalConformance actual selected) :
    actual=selected := by
  cases actual;cases selected
  have h1:=funext h.narrow
  have h2:=funext h.widen
  have h3:=funext fun x => funext (h.mul16 x)
  have h4:=funext fun x => funext (h.add16 x)
  have h5:=funext fun x => funext (h.add32 x)
  congr

inductive Expr where
  | bits : Word → Expr
  | narrow : Expr → Expr
  | widen : Expr → Expr
  | mul16 : Expr → Expr → Expr
  | add16 : Expr → Expr → Expr
  | add32 : Expr → Expr → Expr
  deriving Inhabited

def Expr.eval (ops : Instructions) : Expr → Word
  | .bits w => w
  | .narrow a => (ops.narrow (a.eval ops)).zeroExtend 32
  | .widen a => ops.widen ((a.eval ops).extractLsb' 0 16)
  | .mul16 a b => (ops.mul16 ((a.eval ops).extractLsb' 0 16) ((b.eval ops).extractLsb' 0 16)).zeroExtend 32
  | .add16 a b => (ops.add16 ((a.eval ops).extractLsb' 0 16) ((b.eval ops).extractLsb' 0 16)).zeroExtend 32
  | .add32 a b => ops.add32 (a.eval ops) (b.eval ops)

/-- The shared source executes over expression constructors. Unused algebra
ports are total but cannot supply a source reduction or whole output result. -/
def algebra : Algebra Expr where
  literal := fun _ _ _ => .bits 0
  negInf := .bits 0xff800000
  binary := fun _ dtype op a b =>
    if dtype=.fp16 then if op=.mul then .mul16 a b else .add16 a b else .add32 a b
  unary := fun _ _ a => a
  cast := fun _ src dst a =>
    if src=dst then a else if dst=.fp16 then .narrow a else .widen a
  fromNat := fun _ _ => .bits 0
  fromInt := fun _ _ => .bits 0
  fp32Bits := fun w => .bits w.bits
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ _ => .bits 0
  reduceSum := fun _ {_} _ _ _ _ => .bits 0

abbrev Plans := FPScalarTree.Plans
def reduction (plans : Plans) := FPScalarTree.reduction algebra plans

def inputs (r : Request) (heap : ByteMemory) : FPValues.Data Expr where
  input := fun off => .bits (load32 heap (r.x.dataPtr.toNat+4*off))
  weights := fun off => .bits (half heap (r.w.dataPtr.toNat+2*off))
  output := fun off => .bits (load32 heap (r.y.dataPtr.toNat+4*off))
  indices := indices r heap
  other := fun _ _ => .mk .real (.bits 0)
  padding := fun dtype _ _ => defaultValue dtype

/-- Independent typed product/tree/cast/add expression on entry words. -/
def referenceTerm (plans : Plans) (r : Request) (heap : ByteMemory) (h : Admitted r heap)
    (p : Geometry.Program (call r heap h)) : Expr :=
  FPGrid.value algebra (reduction plans) (call r heap h) p (inputs r heap)

def reference (ops : Instructions) (plans : Plans) (r : Request) (heap : ByteMemory)
    (h : Admitted r heap) (p : Geometry.Program (call r heap h)) : Word :=
  (referenceTerm plans r heap h p).eval ops

def execute (plans : Plans) (r : Request) (heap : ByteMemory) (h : Admitted r heap) :
    Option (FPValues.Data Expr) :=
  Signed.Grid.run algebra (reduction plans) (call r heap h) (Geometry.programs (call r heap h)) (inputs r heap)

theorem source_execution (plans : Plans) (r : Request) (heap : ByteMemory) (h : Admitted r heap) :
    ∃ final,execute plans r heap h=some final ∧
      (∀ p : Geometry.Program (call r heap h),indices r heap p.batch≠-1 →
        final.output (FPGrid.address (call r heap h) p)=referenceTerm plans r heap h p) := by
  have proof := Signed.Grid.complete r.x r.w r.y r.i r.scalars (indices r heap) h
    algebra (reduction plans) (inputs r heap) (fun _ _ => rfl)
  exact ⟨_,proof.1,proof.2⟩

theorem source_native (plans : Plans) (r : Request) (heap : ByteMemory) (h : Admitted r heap)
    (actual selected : Instructions) (numeric : LocalConformance actual selected) :
    ∃ final,execute plans r heap h=some final ∧
      (∀ p : Geometry.Program (call r heap h),indices r heap p.batch≠-1 →
        (final.output (FPGrid.address (call r heap h) p)).eval actual=
          reference selected plans r heap h p) := by
  obtain ⟨final,run,values⟩:=source_execution plans r heap h
  refine ⟨final,run,?_⟩
  intro p active
  rw [values p active,instructions_eq actual selected numeric]
  rfl

#axiomsClean source_execution
#axiomsClean source_native
#specNonCircular reference avoiding [execute,Signed.Grid.run,FP.Structural.exec,Kernels.source]
#specNonCircular LocalConformance avoiding [reference,execute,Expr.eval]
end
end VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.Native
