import VeriTile.Triton.Float.Exponential
import VeriTile.Triton.Float.ScalarProbeExact

/-! Exact certificates for the original supplemental scalar exponential probes.

The source is `experiments/floating_point/supplement/kernels.py`: EXP-ZERO is
`store(rnd(exp(+0)))` versus `store(+1)`; EXP-NEG-INF-SUB is
`store(rnd(exp(rnd(-Inf - load(a)))))` versus `store(+0)`. `load`, every `rnd`,
and the final storage cast are retained. The three existing profiles have
FP32 arithmetic and either FP32 or BF16 storage; only the BF16 profile rounds
intermediate nodes through BF16. This is not BF16 exponentiation.

The tiny syntax fixes these source expressions. A certificate proves their
execution equality for every residual implementation in the selected
documented exponential model. It does not accept an arbitrary equation or
theorem-name string as evidence. The current catalogue uses the two tl rules
at all three profiles and the two libdevice rules only at FP32.
-/
namespace VeriTile.Triton.FP.ExponentialProbeExact

open ScalarProbeExact
abbrev Backend := Exponential.Backend

inductive Rule where
  | zero | negInfSub
  deriving DecidableEq, Repr

def primitive : Backend → ExponentialExact.Primitive
  | .tl => .ptxEx2Approx
  | .libdevice => .libdeviceExpf

def exponential (backend : Backend) :
    ExponentialExact.Residual (primitive backend) → Value .fp32 → Value .fp32 :=
  match backend with
  | .tl => ExponentialExact.tlExp
  | .libdevice => ExponentialExact.libdeviceExp

/-- The intrinsic identity is a type index, not erased by the syntax. -/
inductive Expr (backend : Backend) where
  | input | zero | one | negInf
  | sub (a b : Expr backend)
  | rnd (a : Expr backend)
  | exp (a : Expr backend)

def lhs (backend : Backend) : Rule → Expr backend
  | .zero => .rnd (.exp .zero)
  | .negInfSub => .rnd (.exp (.rnd (.sub .negInf .input)))

def rhs (backend : Backend) : Rule → Expr backend
  | .zero => .one
  | .negInfSub => .zero

def eval (p : Profile) (backend : Backend)
    (rest : ExponentialExact.Residual (primitive backend))
    (input : Value p.storageFormat) : Expr backend → Value .fp32
  | .input => load p input
  | .zero => Value.zero .fp32
  | .one => ExponentialExact.positiveOne
  | .negInf => Value.infinity .fp32 true
  | .sub a b => FP.sub .ieeeValue (eval p backend rest input a) (eval p backend rest input b)
  | .rnd a => rnd p (eval p backend rest input a)
  | .exp a => exponential backend rest (eval p backend rest input a)

def run (p : Profile) (backend : Backend)
    (rest : ExponentialExact.Residual (primitive backend))
    (input : Value p.storageFormat) (e : Expr backend) : Value p.storageFormat :=
  store p (eval p backend rest input e)

def guard (rule : Rule) {f : Format} (input : Value f) : Prop :=
  match rule with
  | .zero => True
  | .negInfSub => input.toRat?.isSome = true

/-- Quantified equality of the fixed source expressions, including the final
storage conversion. The constant rule has no input restriction. -/
def Certificate (p : Profile) (backend : Backend) (rule : Rule) : Prop :=
  ∀ (rest : ExponentialExact.Residual (primitive backend))
    (input : Value p.storageFormat), guard rule input →
      run p backend rest input (lhs backend rule) = run p backend rest input (rhs backend rule)

theorem rnd_zero (p : Profile) : rnd p (Value.zero .fp32) = Value.zero .fp32 := by
  cases p <;> decide +kernel

theorem rnd_one (p : Profile) :
    rnd p ExponentialExact.positiveOne = ExponentialExact.positiveOne := by
  cases p <;> decide +kernel

theorem rnd_neg_inf (p : Profile) :
    rnd p (Value.infinity .fp32 true) = Value.infinity .fp32 true := by
  cases p <;> decide +kernel

theorem store_zero (p : Profile) :
    store p (Value.zero .fp32) = Value.zero p.storageFormat := by
  cases p <;> decide +kernel

theorem store_one (p : Profile) :
    store p ExponentialExact.positiveOne = ScalarOpsLaws.one p.storageFormat := by
  cases p <;> decide +kernel

theorem exponential_zero (backend : Backend)
    (rest : ExponentialExact.Residual (primitive backend)) :
    exponential backend rest (Value.zero .fp32) = ExponentialExact.positiveOne := by
  cases backend
  · exact ExponentialExact.tl_exp_zero rest false
  · exact ExponentialExact.libdevice_exp_zero rest false

theorem exponential_neg_inf (backend : Backend)
    (rest : ExponentialExact.Residual (primitive backend)) :
    exponential backend rest (Value.infinity .fp32 true) = Value.zero .fp32 := by
  cases backend
  · exact ExponentialExact.tl_exp_neg_inf rest
  · exact ExponentialExact.libdevice_exp_neg_inf rest

theorem zero_certificate (p : Profile) (backend : Backend) : Certificate p backend .zero := by
  intro rest input _
  simp only [run, lhs, rhs, eval, exponential_zero, rnd_one]

theorem neg_inf_sub_certificate (p : Profile) (backend : Backend) :
    Certificate p backend .negInfSub := by
  intro rest input hi
  have hfinite : ExponentialExact.finite (load p input) :=
    (ExponentialExact.finite_iff_toRat _).mpr (load_finite p input hi)
  simp only [run, lhs, rhs, eval, ExponentialExact.neg_inf_sub .ieeeValue _ hfinite,
    rnd_neg_inf, exponential_neg_inf, rnd_zero]

theorem certificate (p : Profile) (backend : Backend) (rule : Rule) :
    Certificate p backend rule := by
  cases rule
  · exact zero_certificate p backend
  · exact neg_inf_sub_certificate p backend

#axiomsClean rnd_zero
#axiomsClean rnd_one
#axiomsClean rnd_neg_inf
#axiomsClean store_zero
#axiomsClean store_one
#axiomsClean zero_certificate
#axiomsClean neg_inf_sub_certificate
#axiomsClean certificate

end VeriTile.Triton.FP.ExponentialProbeExact
