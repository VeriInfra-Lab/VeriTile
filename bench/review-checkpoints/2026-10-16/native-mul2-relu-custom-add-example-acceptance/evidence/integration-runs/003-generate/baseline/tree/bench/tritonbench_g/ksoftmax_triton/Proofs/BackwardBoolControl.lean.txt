import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardStaticControl

namespace VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward.StaticControl
open VeriTile.Triton

/-- Keep a source constexpr flag universally quantified during proof-only
control-flow normalization, rather than enumerating flag combinations. -/
theorem if_bool (condition : Bool) (body : List Stmt) (s : BlockState) :
    stepStmt (.ifThen (.constBool condition) body) s =
      if condition then stepStmts body s else some s := by
  cases condition <;> simp [if_true, if_false]

theorem ifElse_bool (condition : Bool) (left right : List Stmt) (s : BlockState) :
    stepStmt (.ifThenElse (.constBool condition) left right) s =
      if condition then stepStmts left s else stepStmts right s := by
  cases condition <;> simp [ifElse_true, ifElse_false]

/-- Distribute only interpreter sequencing, avoiding unrestricted rewriting
of conditional expressions as arbitrary function applications. -/
theorem bind_if {α β : Type} (condition : Bool) (left right : Option α)
    (next : α → Option β) :
    (if condition then left else right).bind next =
      if condition then left.bind next else right.bind next := by
  cases condition <;> rfl

theorem bind_fun_if {α β : Type} (condition : Bool) (value : Option α)
    (left right : α → Option β) :
    value.bind (fun v => if condition then left v else right v) =
      if condition then value.bind left else value.bind right := by
  cases condition <;> rfl

#axiomsClean if_bool
#axiomsClean ifElse_bool
#axiomsClean bind_if
#axiomsClean bind_fun_if
end VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward.StaticControl
