import VeriTile.Triton.Float.Erasure

/-! Proof-only insertion of real-to-real identity casts on named assignments.
This is a view of an existing kernel, never an authoritative source. A caller
must check its complete execution connection; RMSNorm's uncast_var_run and
uncast_var_flat prove it through both full source contexts, including failure.
No floating cast is removed from the original or FP candidate. -/
namespace VeriTile.Triton.RealCastView
mutual
 def castBinding (name : RegName) : Stmt → Stmt
  | .assign .real sh target value =>
    .assign .real sh target (if target = name then .castFloat .real .real value else value)
  | .forLoop idx bound body => .forLoop idx bound (castBindings name body)
  | .forRange idx a b c body => .forRange idx a b c (castBindings name body)
  | .forRangeDyn idx a b c body => .forRangeDyn idx a b c (castBindings name body)
  | .forRangeInt idx a b c body => .forRangeInt idx a b c (castBindings name body)
  | .ifThenElse cond yes no => .ifThenElse cond (castBindings name yes) (castBindings name no)
  | statement => statement
 termination_by statement => sizeOf statement
 def castBindings (name : RegName) : List Stmt → List Stmt
  | [] => []
  | head :: tail => castBinding name head :: castBindings name tail
 termination_by statements => sizeOf statements
end

def ofKernel (name : RegName) (source : Kernel) : ComputeKernel :=
  ComputeKernel.fromKernelBody source.inputs source.outputs (castBindings name source.body)
end VeriTile.Triton.RealCastView
