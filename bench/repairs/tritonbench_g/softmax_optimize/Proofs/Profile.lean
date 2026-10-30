import bench.repairs.tritonbench_g.softmax_optimize.Proofs.Graph
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.Words
namespace VeriTile.Bench.Repairs.SoftmaxOptimize.Profile
open Graph
abbrev Word := BitVec 32
abbrev Instructions := Ksoftmax.Graph.Operations
namespace W
abbrev finite := Ksoftmax.Words.finite
abbrev AddDomain := Ksoftmax.Words.AddDomain
abbrev SubDomain := Ksoftmax.Words.SubDomain
abbrev MulDomain := Ksoftmax.Words.MulDomain
end W

/-- Concrete FP32 constants, comparison and exp instruction lowering. -/
def operations (ops : Instructions) : Graph.Operations Word where
  zero := 0
  negInf := 0xff800000
  maximum := ops.max32
  subtract := ops.sub32
  exponential := fun a => ops.ex2Approx32 (ops.mul32 a 0x3fb8aa3b)
  multiply := ops.mul32
  add := ops.add32
  isNegInf := fun a => decide (a=0xff800000)
  fusedMultiplyAdd := ops.fmaRN32
  divide := ops.divFull32

/-- Exact recursive operand domain; the inactive `where` branch does not
contribute a value to subsequent instructions. -/
def Domain (ops : Instructions) (input : ι → Word) : Expr ι → Prop
  | .input _ | .zero | .negInf => True
  | .maximum a b | .divide a b => Domain ops input a ∧ Domain ops input b
  | .subtract a b => Domain ops input a ∧ Domain ops input b ∧
      W.SubDomain (eval (operations ops) input a) (eval (operations ops) input b)
  | .exponential a => Domain ops input a ∧ W.MulDomain (eval (operations ops) input a) 0x3fb8aa3b
  | .multiply a b => Domain ops input a ∧ Domain ops input b ∧
      W.MulDomain (eval (operations ops) input a) (eval (operations ops) input b)
  | .add a b => Domain ops input a ∧ Domain ops input b ∧
      W.AddDomain (eval (operations ops) input a) (eval (operations ops) input b)
  | .fma a b c => Domain ops input a ∧ Domain ops input b ∧ Domain ops input c ∧
      W.finite (eval (operations ops) input a) ∧ W.finite (eval (operations ops) input b) ∧
      W.finite (eval (operations ops) input c)
  | .ifNegInf test yes no => Domain ops input test ∧
      if eval (operations ops) input test=0xff800000 then Domain ops input yes else Domain ops input no

abbrev LocalExact := Ksoftmax.Words.LocalExact

theorem exact_if_local (ops : Instructions) (ref : Ksoftmax.Words.PrimitiveReference)
    (ports : LocalExact ops ref) (input : ι → Word) (e : Expr ι) (domain : Domain ops input e) :
    eval (operations ops) input e=eval (operations (Ksoftmax.Words.referenceOps ref)) input e := by
  induction e with
  | input i => rfl
  | zero => rfl
  | negInf => rfl
  | maximum a b ia ib =>
    exact (ports.maximum _ _).trans (congrArg₂ ref.maximum32 (ia domain.1) (ib domain.2))
  | subtract a b ia ib =>
    exact (ports.sub _ _ domain.2.2).trans (congrArg₂ Ksoftmax.Words.softwareSub (ia domain.1) (ib domain.2.1))
  | exponential a ia =>
    change ops.ex2Approx32 (ops.mul32 _ _) = ref.exp2Approx32 (Ksoftmax.Words.softwareMul _ _)
    rw [ports.exp2,ports.mul _ _ domain.2,ia domain.1]
  | multiply a b ia ib =>
    exact (ports.mul _ _ domain.2.2).trans (congrArg₂ Ksoftmax.Words.softwareMul (ia domain.1) (ib domain.2.1))
  | add a b ia ib =>
    exact (ports.add _ _ domain.2.2).trans (congrArg₂ Ksoftmax.Words.softwareAdd (ia domain.1) (ib domain.2.1))
  | divide a b ia ib =>
    exact (ports.division _ _).trans (congrArg₂ ref.divisionFull32 (ia domain.1) (ib domain.2))
  | fma a b c ia ib ic =>
    change ops.fmaRN32 _ _ _ = Ksoftmax.Words.softwareFMA _ _ _
    rw [ports.fma _ _ _ domain.2.2.2.1 domain.2.2.2.2.1 domain.2.2.2.2.2,
      ia domain.1,ib domain.2.1,ic domain.2.2.1]
  | ifNegInf test yes no it iy ino =>
    have ht := it domain.1
    change (if decide (eval (operations ops) input test=0xff800000) then eval (operations ops) input yes else eval (operations ops) input no)=
      (if decide (eval (operations (Ksoftmax.Words.referenceOps ref)) input test=0xff800000) then
        eval (operations (Ksoftmax.Words.referenceOps ref)) input yes else eval (operations (Ksoftmax.Words.referenceOps ref)) input no)
    simp only [decide_eq_true_eq]
    rw [←ht]
    by_cases h : eval (operations ops) input test=0xff800000
    · simp only [h,if_true]
      exact iy (by simpa only [if_pos h] using domain.2)
    · simp only [h,if_false]
      exact ino (by simpa only [if_neg h] using domain.2)

#specNonCircular Domain avoiding [VeriTile.Triton.exec]
#axiomsClean exact_if_local
end VeriTile.Bench.Repairs.SoftmaxOptimize.Profile
