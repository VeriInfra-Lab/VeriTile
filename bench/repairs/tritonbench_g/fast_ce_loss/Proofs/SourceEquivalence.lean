import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.KernelSource
import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.KernelForward
import VeriTile.Meta.Specification

namespace VeriTile.Bench.Repairs.FastCeLoss.SourceEquivalence
open VeriTile Triton FP.Structural KernelSource
open scoped VeriTile.Spec
noncomputable section

/-- Actual typed source; every native primitive remains an opaque callback. -/
structure Program where
  kernel : ComputeKernel

def Equivalent (lhs rhs : Program) : Prop :=
  ∀ (α : Type) [Inhabited α] (M : Algebra α) (s : State α),
    exec (engine M) lhs.kernel s = exec (engine M) rhs.kernel s

instance : Spec.ProgramSyntax Program where
  Statement := ComputeStmt
  Signature := List RegionName × List RegionName
  signature p := (p.kernel.inputs,p.kernel.outputs)
  body p := p.kernel.surfaceBody
  structural := some Equivalent
  sameContext := fun lhs rhs => lhs=rhs

def assumptions : Spec.Assumptions ComputeStmt := []
def forwardProgram (candidate : Bool) (p : Parameters) : Program :=
  ⟨if candidate then TypedSource.forwardIndexCommuted p.hasSoftcap p.hasScaling
      p.logits p.loss p.lse p.labels p.vocab p.logitsStride p.block p.softcap p.scaling
    else forward p⟩
def chunkedProgram (candidate : Bool) (p : Parameters) : Program :=
  ⟨if candidate then TypedSource.chunkedIndexCommuted p.hasSoftcap p.hasScaling
      p.logits p.loss p.lse p.labels p.vocab p.chunks p.logitsStride p.block p.softcap p.scaling
    else chunked p⟩
def backwardProgram (candidate : Bool) (p : Parameters) : Program :=
  ⟨if candidate then TypedSource.backwardIndexCommuted p.hasSoftcap p.hasScaling
      p.logits p.upstream p.lse p.labels p.vocab p.logitsStride p.upstreamStride p.block p.softcap p.scaling
    else backward p⟩

def rowStep : ComputeStmt :=
  .assign .int [] "row_idx" (.alg (.castNatToInt (.programId 0)))
def axisStep (name : RegName) : ComputeStmt :=
  .assign .int [] name (.alg (.castNatToInt (.programId 1)))
def pointerStep (candidate : Bool) (p : Parameters) : ComputeStmt :=
  let a : Op .int [] := .ref .int [] "row_idx"
  let b : Op .int [] := .castNatToInt (.constNat p.logitsStride)
  .assign .ptr [] "logits_ptr" (.alg (.ptrAdd .nil (.ptrBase p.logits)
    (.castIntToNat (if candidate then .mul .int .nil b a else .mul .int .nil a b))))

/-- The only changed expression is signed integer multiplication before its
unchanged pointer-index conversion. No floating identity is used. -/
theorem pointer_step {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :
    step (engine M) (pointerStep Bool.false p) s =
      step (engine M) (pointerStep Bool.true p) s := by
  simp only [pointerStep, Bool.false_eq_true, ↓reduceIte, step, evalExpr]
  have product : evalOp (engine M) none
      (.mul .int .nil (.ref .int [] "row_idx") (.castNatToInt (.constNat p.logitsStride))) s =
    evalOp (engine M) none
      (.mul .int .nil (.castNatToInt (.constNat p.logitsStride)) (.ref .int [] "row_idx")) s := by
    change (s.regs .int [] "row_idx" >>= fun x =>
        some (bop Int.mul .nil x (fun _ => (p.logitsStride : Int)))) =
      (s.regs .int [] "row_idx" >>= fun x =>
        some (bop Int.mul .nil (fun _ => (p.logitsStride : Int)) x))
    cases s.regs .int [] "row_idx" with
    | none => rfl
    | some x =>
      apply congrArg some
      funext i
      exact Int.mul_comm _ _
  change ((evalOp (engine M) none
      (.mul .int .nil (.ref .int [] "row_idx") (.castNatToInt (.constNat p.logitsStride))) s >>= fun y =>
      some (fun i => (y i).toNat)) >>= fun y =>
      some (bop (fun (q : RegionName × Nat) (n : Nat) => (q.1,q.2+n))
        .nil (fun _ => (p.logits,0)) y)) >>= _ = _
  rw [product]
  rfl

theorem forward_split (candidate : Bool) (p : Parameters) :
    (forwardProgram candidate p).kernel.surfaceBody =
      [rowStep,pointerStep candidate p] ++ (forward p).surfaceBody.drop 2 := by
  rcases p with ⟨_,_,_,_,_,_,_,_,_,_,_,_,_,cap,scale⟩
  cases cap <;> cases scale <;> cases candidate <;> rfl

theorem chunked_split (candidate : Bool) (p : Parameters) :
    (chunkedProgram candidate p).kernel.surfaceBody =
      [rowStep,axisStep "chunk_idx",pointerStep candidate p] ++ (chunked p).surfaceBody.drop 3 := by
  rcases p with ⟨_,_,_,_,_,_,_,_,_,_,_,_,_,cap,scale⟩
  cases cap <;> cases scale <;> cases candidate <;> rfl

theorem backward_split (candidate : Bool) (p : Parameters) :
    (backwardProgram candidate p).kernel.surfaceBody =
      [rowStep,axisStep "block_idx",pointerStep candidate p] ++ (backward p).surfaceBody.drop 3 := by
  rcases p with ⟨_,_,_,_,_,_,_,_,_,_,_,_,_,cap,scale⟩
  cases cap <;> cases scale <;> cases candidate <;> rfl

theorem forward_equivalent (p : Parameters) :
    forwardProgram Bool.false p ≡[assumptions] forwardProgram Bool.true p := by
  apply Spec.FloatingPoint.ofStructural (structural := Equivalent)
    (lhs := forwardProgram Bool.false p) (rhs := forwardProgram Bool.true p) rfl
  · rcases p with ⟨_,_,_,_,_,_,_,_,_,_,_,_,_,cap,scale⟩
    cases cap <;> cases scale <;> rfl
  intro α _ M s
  change run (engine M) (forwardProgram Bool.false p).kernel.surfaceBody s =
    run (engine M) (forwardProgram Bool.true p).kernel.surfaceBody s
  rw [forward_split,forward_split,run_append,run_append]
  simp only [run, pointer_step]

theorem chunked_equivalent (p : Parameters) :
    chunkedProgram Bool.false p ≡[assumptions] chunkedProgram Bool.true p := by
  apply Spec.FloatingPoint.ofStructural (structural := Equivalent)
    (lhs := chunkedProgram Bool.false p) (rhs := chunkedProgram Bool.true p) rfl
  · rcases p with ⟨_,_,_,_,_,_,_,_,_,_,_,_,_,cap,scale⟩
    cases cap <;> cases scale <;> rfl
  intro α _ M s
  change run (engine M) (chunkedProgram Bool.false p).kernel.surfaceBody s =
    run (engine M) (chunkedProgram Bool.true p).kernel.surfaceBody s
  rw [chunked_split,chunked_split,run_append,run_append]
  simp only [run, pointer_step]

theorem backward_equivalent (p : Parameters) :
    backwardProgram Bool.false p ≡[assumptions] backwardProgram Bool.true p := by
  apply Spec.FloatingPoint.ofStructural (structural := Equivalent)
    (lhs := backwardProgram Bool.false p) (rhs := backwardProgram Bool.true p) rfl
  · rcases p with ⟨_,_,_,_,_,_,_,_,_,_,_,_,_,cap,scale⟩
    cases cap <;> cases scale <;> rfl
  intro α _ M s
  change run (engine M) (backwardProgram Bool.false p).kernel.surfaceBody s =
    run (engine M) (backwardProgram Bool.true p).kernel.surfaceBody s
  rw [backward_split,backward_split,run_append,run_append]
  simp only [run, pointer_step]

#print_fp_assumptions forward_equivalent
#print_fp_assumptions chunked_equivalent
#print_fp_assumptions backward_equivalent
#axiomsClean forward_equivalent
#axiomsClean chunked_equivalent
#axiomsClean backward_equivalent
end
end VeriTile.Bench.Repairs.FastCeLoss.SourceEquivalence
