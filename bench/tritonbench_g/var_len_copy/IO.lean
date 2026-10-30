import Mathlib.Tactic.FinCases
import bench.tritonbench_g.var_len_copy.Proofs.SourceProjection
import VeriTile.Triton.Memory.LogicalIO
import VeriTile.Triton.Memory.RegionalKernelIOOn

namespace VeriTile.Bench.Repairs.VarLenCopy.PublicIO
open VeriTile.Triton
noncomputable section

def inputType : Fin 4 → ChanTy
  | ⟨0,_⟩ | ⟨1,_⟩ | ⟨2,_⟩ => .int
  | ⟨3,_⟩ => .float

def inputArity (a : Segment) : Fin 4 → Nat
  | ⟨0,_⟩ | ⟨1,_⟩ | ⟨2,_⟩ => 1
  | ⟨3,_⟩ => a.len

def inputRegion : Fin 4 → RegionName
  | ⟨0,_⟩ => "length"
  | ⟨1,_⟩ => "old"
  | ⟨2,_⟩ => "new"
  | ⟨3,_⟩ => "source"

def readWindow (a : Segment) (i : Fin 4)
    (_ : ∀ k, Fin (inputArity a k) → (inputType k).carrier)
    (pid _ _ : Nat) : Fin (inputArity a i) → Nat :=
  match i with
  | ⟨0,_⟩ | ⟨1,_⟩ | ⟨2,_⟩ => fun _=>pid
  | ⟨3,_⟩ => fun j=>a.old+j.val

/-- Three signed metadata reads plus the full source segment. The source
projection is the existing Real interpretation of the exact FP32 quotation. -/
abbrev io (a : Segment) : UKernelIO where
  kernel := Source.originalKernel.eraseDType
  projection := sourceProjection
  nIn := 4
  nOut := 1
  nScr := 0
  bufs := ["length","old","new","source","output"]
  ity := inputType
  iarity := inputArity a
  ibuf := inputRegion
  oty := fun _=>.float
  oarity := fun _=>a.len
  obuf := fun _=>"output"
  obuf_mem := fun _=>by simp
  sarity := fun k=>k.elim0
  sbuf := fun k=>k.elim0
  iwin := readWindow a
  imask := fun _ _ _ _ _ _=>True
  owin := fun _ _ _ _ _ j=>a.dest+j.val
  omask := fun _ _ _ _ _ _=>True
  swin := fun k=>k.elim0
  smask := fun k=>k.elim0

/-- Only active starts are constrained. Nonpositive length metadata retains
arbitrary signed old/new starts, exactly as the original kernel theorem. -/
def metadataDomain (a : Segment) (xs : (io a).Ctx) : Prop :=
  xs (⟨0,by decide⟩ : Fin 4) (⟨0,by decide⟩ : Fin 1)=a.length.toInt ∧
  (0<a.len → xs (⟨1,by decide⟩ : Fin 4) (⟨0,by decide⟩ : Fin 1)=(a.old : Int) ∧
    xs (⟨2,by decide⟩ : Fin 4) (⟨0,by decide⟩ : Fin 1)=(a.dest : Int))

/-- Independent logical copy: every destination lane equals its input lane. -/
def formula (a : Segment) (_ _ _ : Nat) (xs : (io a).Ctx)
    (_ : Fin 1) (j : Fin a.len) : ℝ := xs (⟨3,by decide⟩ : Fin 4) j


structure LogicalInputs (a : Segment) where
  length : Int
  oldStart : Int
  newStart : Int
  source : Fin a.len → ℝ

def logicalInputs (a : Segment) : (io a).Ctx ≃ LogicalInputs a where
  toFun xs := ⟨xs (0 : Fin 4) (0 : Fin 1),xs (1 : Fin 4) (0 : Fin 1),xs (2 : Fin 4) (0 : Fin 1),xs (3 : Fin 4)⟩
  invFun xs := fun port => match port with
    | ⟨0,_⟩ => fun _ => xs.length
    | ⟨1,_⟩ => fun _ => xs.oldStart
    | ⟨2,_⟩ => fun _ => xs.newStart
    | ⟨3,_⟩ => xs.source
  left_inv := by
    intro xs; funext port j; fin_cases port
    · change Fin 1 at j; have h : j=(0 : Fin 1) := Fin.eq_zero j; subst j; rfl
    · change Fin 1 at j; have h : j=(0 : Fin 1) := Fin.eq_zero j; subst j; rfl
    · change Fin 1 at j; have h : j=(0 : Fin 1) := Fin.eq_zero j; subst j; rfl
    · rfl
  right_inv := by intro xs; cases xs; rfl

def logicalOutputs (a : Segment) :
    (Nat → Nat → Nat → Fin 1 → Fin a.len → ℝ) ≃ (Nat → Nat → Nat → Fin a.len → ℝ) where
  toFun xs p q r := xs p q r 0
  invFun xs p q r _ := xs p q r
  left_inv := by intro xs; funext p q r o j; have h : o=0 := Fin.eq_zero o; subst o; rfl
  right_inv := by intro xs; rfl

/-- The fixed regional judgment still quantifies every initial 3D PID state,
proves access safety, successful execution and the complete exterior frame. -/
def logicalIO (a : Segment) : LogicalDomainIO (io a).Ctx
    (Nat → Nat → Nat → Fin 1 → Fin a.len → ℝ)
    (LogicalInputs a) (Nat → Nat → Nat → Fin a.len → ℝ)
    (fun domain f => (io a).RegionalImplementsOn3D domain (fun p q r xs => f xs p q r)) where
  inputs := logicalInputs a
  outputs := logicalOutputs a

def logicalDomain (a : Segment) (xs : LogicalInputs a) : Prop :=
  xs.length=a.length.toInt ∧ (0<a.len → xs.oldStart=(a.old : Int) ∧ xs.newStart=(a.dest : Int))

end
end VeriTile.Bench.Repairs.VarLenCopy.PublicIO
