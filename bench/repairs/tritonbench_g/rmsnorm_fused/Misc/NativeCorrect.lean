import bench.repairs.tritonbench_g.rmsnorm_fused.Proofs.IONative

namespace VeriTile.Bench.Repairs.RmsnormFused.FPEquiv
open VeriTile Triton FP.Structural Ordered
open VeriTile.Bench.TritonBenchG.RmsNormTriton.Observed (Term graphAlgebra)
open scoped VeriTile.Triton.FP.Structural.ProgramIO
noncomputable section

/-- The complete canonical source grid, with actual epsilon ABI words and
per-column native observations. IO.lean exposes the same typed spans as Correct. -/
def nativeIO (shape : Shape) (I : Instructions) (P : Policy) (R : Reductions)
    (epsilon : Word) (rows : List Nat) : ProgramIO Word (Term Word) :=
  Interface.programIO graphAlgebra shape (.input epsilon) rows Term.input
    (fun i => interpret I P R (i%shape.width))

/-- Independent ordered native recurrence; primitive meanings, compiler fusion
choices and reduction topology retain their existing model boundary. -/
def native_formula (shape : Shape) (I : Instructions) (P : Policy) (R : Reductions)
    (epsilon : Word) (values : Interface.Inputs Word shape)
    (i : Fin (shape.rows*shape.width)) : Word :=
  Ordered.reference I P R shape.width shape.block epsilon
    (fun j => Interface.inputAt values ((i.val/shape.width)*shape.width+j))
    (Interface.weightAt values) (i.val%shape.width)

/-- Native observations of every output in the full scheduled source grid.
All FP32 words remain admitted. This is source-to-native interpretation;
there is no floating transformation law or experiment to defer. -/
theorem kernel_native (shape : Shape) (valid : shape.Valid)
    (I : Instructions) (P : Policy) (R : Reductions) (epsilonValue : ℝ)
    (epsilonWord : Word) (_abi : I.scalarABI epsilonValue=epsilonWord)
    (rows : List Nat) (all : rows.Perm shape.grid) :
    nativeIO shape I P R epsilonWord rows ⊨ fun values =>
      Interface.result (native_formula shape I P R epsilonWord values) := by
  exact Ordered.io_native shape valid I P R epsilonWord rows all

#stmtSurfaceSubset kernel_native ⊆
  [Shape, Shape.Valid, Shape.grid, Shape.width, Shape.rows, Instructions,
   Instructions.scalarABI, Policy, Reductions, Word, nativeIO,
   ProgramIO.Implements, ProgramIO.Inputs, Interface.result, native_formula, Term,
   VeriTile.Bench.TritonBenchG.RmsNormTriton.Observed.instInhabitedTerm]
#specNonCircular native_formula avoiding [source,Runtime.value,Runtime.run,FP.Structural.exec,Ordered.lower,Ordered.interpret]
#axiomsClean kernel_native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsnormFused.FPEquiv
