import VeriTile.Triton.DSL
import VeriTile.Triton.Float.ProfiledSoundness
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.RegisterDTypeWhere
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
set_option maxHeartbeats 2000000

set_option hygiene false in
macro "registerCast%" "[" dt:tritonDType "]" : term =>
  `(fun (X : RegionName) => triton {
    tl.region X = $dt
    g = tl.load(X)
    zero = float(0)
    z = (zero).to(g.dtype)
  })

set_option hygiene false in
macro "explicitCast%" "[" dt:tritonDType "]" : term =>
  `(fun (X : RegionName) => triton {
    tl.region X = $dt
    g = tl.load(X)
    zero = float(0)
    z = (zero).to($dt:tritonDType)
  })

/-- A register's dtype selects the same conversion as the explicit source type,
including the separate fp32/fp64 precision profiles. -/
theorem all_register_casts :
    registerCast%[tl.float64] = explicitCast%[tl.float64] ∧
    registerCast%[tl.float32] = explicitCast%[tl.float32] ∧
    registerCast%[tl.float16] = explicitCast%[tl.float16] ∧
    registerCast%[tl.bfloat16] = explicitCast%[tl.bfloat16] ∧
    registerCast%[tl.float8e4nv] = explicitCast%[tl.float8e4nv] ∧
    registerCast%[tl.float8e5] = explicitCast%[tl.float8e5] :=
  ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩

def wideWhere (X : RegionName) : ComputeKernel := triton {
  tl.region X = tl.float64
  g = tl.load(X)
  y = tl.where(true, g, g)
  out = y + y
}

def initial {α : Type} [Inhabited α] (value : α) : State α where
  mem := fun _ _ => .mk .real value
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun dtype _ _ => defaultValue dtype

/-- The consumer of where still computes in fp64 under any floating algebra. -/
theorem where_preserves_fp64 {α : Type} [Inhabited α] (M : Algebra α) (value : α) :
    ∃ t, run (engine M) (wideWhere "X").surfaceBody (initial value) = some t ∧
      t.regs .real [] "out" = some (fun _ =>
        M.binary (some .fp64) .real .add value value) := by
  simp [wideWhere, ComputeKernel.surfaceBody, run, step,
    FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, initial,
    State.setReg, Cell.read, Region.cast, engine,
    Algebra.withDefaultPrecision, resolvePrecision, ComputeDType.eraseDType,
    Option.bind, FP.Structural.numeric, FP.Structural.bop]
  funext i
  rfl

def mixedWhere (X Y : RegionName) : ComputeKernel := triton {
  tl.region X = tl.float16, Y = tl.bfloat16
  x = tl.load(X)
  y = tl.load(Y)
  out = tl.where(true, x, y)
}

/-- Mixed fp16/bf16 branches retain fp16 storage and explicit bf16 conversion.
The narrow result has no outer compute tag, so each operand keeps its fp32
child context; erasing these scopes would change the cast backend profile. -/
theorem mixed_half_where (X Y : RegionName) :
    (mixedWhere X Y).surfaceBody.getLast? = some
      (.assign .fp16 [] "out" (.alg (.where (.constBool Bool.true)
        (.withPrecision .fp16 .fp32 (.ref .fp16 [] "x"))
        (.withPrecision .fp16 .fp32 (.castFloat .bf16 .fp16 (.ref .bf16 [] "y")))))) := rfl

#axiomsClean all_register_casts
#axiomsClean where_preserves_fp64
#axiomsClean mixed_half_where
end VeriTile.Tests.RegisterDTypeWhere
