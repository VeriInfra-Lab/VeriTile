import bench.tritonbench_g.uniform_sampling.Proofs.Mathematical_Kernels
import VeriTile.Triton.Float.Erasure

/-! Checked projection of the complete expanded source. Projection success
is explicit: no error-default kernel can stand in for an unsupported source.
Floating erasure retains all integer operations, ten Philox rounds, all four
uniform decoders, and the four independent masked stores.
-/
namespace VeriTile.Bench.TritonBenchG.UniformSampling.Erasure
open VeriTile.Triton TypedSource
set_option maxHeartbeats 5000000
set_option maxRecDepth 20000

def erasedBody (body : List ComputeStmt) : Except EraseDTypeError (List Stmt) := do
  return Stmt.eraseDTypeList (← ComputeStmt.listToAlgorithm? body)

theorem eraseList_append (before after : List Stmt) :
    Stmt.eraseDTypeList (before ++ after) =
      Stmt.eraseDTypeList before ++ Stmt.eraseDTypeList after := by
  induction before with
  | nil => simp only [List.nil_append, Stmt.eraseDTypeList]
  | cons head rest ih => simp only [List.cons_append, Stmt.eraseDTypeList, ih]

theorem erasedBody_append (before after : List ComputeStmt) :
    erasedBody (before ++ after) = (do
      let a ← erasedBody before
      let b ← erasedBody after
      return a ++ b) := by
  simp only [erasedBody, ComputeStmt.listToAlgorithm?_append]
  cases ComputeStmt.listToAlgorithm? before <;>
    cases ComputeStmt.listToAlgorithm? after <;>
    simp only [bind, pure, Except.bind, Except.pure, eraseList_append]

theorem erasedBody_eq_of_parts (a b : List ComputeStmt) (n : Nat)
    (hp : a.take n = b.take n) (hs : erasedBody (a.drop n) = erasedBody (b.drop n)) :
    erasedBody a = erasedBody b := by
  conv_lhs => rw [← List.take_append_drop n a]
  conv_rhs => rw [← List.take_append_drop n b]
  rw [erasedBody_append, erasedBody_append, hp, hs]

theorem erased_projection_mk (inputs outputs : List RegionName) (body : List ComputeStmt) :
    (ComputeKernel.mk inputs outputs body).eraseDType.toAlgorithm? = (do
      return Kernel.mk inputs outputs (← erasedBody body)) := by
  cases h : ComputeStmt.listToAlgorithm? body <;>
    simp [ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, h, erasedBody,
      Kernel.eraseDType, ComputeKernel.fromKernelBody, bind, pure, Except.bind, Except.pure]

theorem erased_projection_of_ok (ck : ComputeKernel) (k : Kernel)
    (h : ck.toAlgorithm? = .ok k) :
    ck.eraseDType.toAlgorithm? = .ok k.eraseDType := by
  simp only [ComputeKernel.eraseDType, h, ComputeKernel.toAlgorithm?_fromKernelBody]

end VeriTile.Bench.TritonBenchG.UniformSampling.Erasure
