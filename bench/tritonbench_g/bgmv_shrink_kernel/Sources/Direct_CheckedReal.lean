import bench.tritonbench_g.bgmv_shrink_kernel.Sources.Contract
import bench.tritonbench_g.bgmv_shrink_kernel.Sources.Direct_AtomicOrder
import bench.tritonbench_g.bgmv_shrink_kernel.Sources.Direct_Linearization

namespace VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.CheckedReal
open VeriTile.Triton
open scoped BigOperators
noncomputable section

/-- Independent GEMV in original dense caller coordinates. -/
def math (h : Call) (initial : BlockState) (indices : Nat → Int) (b : Fin h.batches) (n : Fin h.N) : ℝ :=
  let old:=initial.readMem "Y" (b.val*h.columns+n.val)
  if indices b.val=-1 then old else old+h.scaling*∑ k : Fin h.K,
    initial.readMem "X" (b.val*h.K+k.val)*initial.readMem "W" ((indices b.val).toNat*(h.N*h.K)+n.val*h.K+k.val)

def initial (h : Call) (u : BlockState) (b : Fin h.batches) (n : Fin h.N) : ℝ :=
  u.readMem "Y" (Geometry.outputOffset h b.val n.val)
def step (h : Call) (u : BlockState) (indices : Nat → Int)
    (b : Fin h.batches) (n : Fin h.N) (split : Fin 64) (old : ℝ) :=
  old+RealExecution.contribution h u b.val (indices b.val).toNat n.val split.val
abbrev Run (h : Call) (orders : Workset.Orders h) (u : BlockState) (indices : Nat → Int) :=
  Linearization.Run h orders indices (initial h u) (step h u indices)

theorem readback (h : Call) (orders : Workset.Orders h) (u : BlockState) (indices : Nat → Int)
    (run : Run h orders u indices) (b : Fin h.batches) (n : Fin h.N) :
    run.output b n=math h u indices b n := by
  rw [Linearization.Run.readback]
  unfold math initial
  by_cases skip : indices b.val=-1
  · simp [skip,Geometry.outputOffset]
  · simp only [skip,if_false]
    exact AtomicOrder.contribution_readback h orders u b n (indices b.val).toNat

#axiomsClean readback
#specNonCircular math avoiding [Run,RealExecution.contribution,VeriTile.Triton.exec]
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.CheckedReal
