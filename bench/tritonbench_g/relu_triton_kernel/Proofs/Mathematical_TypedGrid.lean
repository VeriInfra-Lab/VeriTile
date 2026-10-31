import bench.tritonbench_g.relu_triton_kernel.Proofs.Mathematical_TypedValue

namespace VeriTile.Bench.TritonBenchG.ReluTritonKernel.TypedValue
open VeriTile.Triton FP.Structural GatedGrid
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def reference {α : Type} [Inhabited α] (M : Algebra α)
    (cmp16 cmp32 : α → α → Bool) (k : Kind) (x y : RegionName)
    (N B : Nat) (mem : Memory α) : Memory α :=
  fun r o => if r = y ∧ o < B ∧ o < N then
    .mk .fp32 (resultValue M cmp16 cmp32 k (mem x o)) else mem r o

theorem reference_frame {α : Type} [Inhabited α] (M : Algebra α)
    (cmp16 cmp32 : α → α → Bool) (k : Kind) (x y : RegionName)
    (N B : Nat) (mem : Memory α) (r : RegionName) (o : Nat)
    (outside : r ≠ y ∨ B ≤ o ∨ N ≤ o) :
    reference M cmp16 cmp32 k x y N B mem r o = mem r o := by
  simp only [reference]
  rw [if_neg]
  intro h
  rcases outside with hr | hb | hn
  · exact hr h.1
  · omega
  · omega

#specNonCircular reference avoiding [VeriTile.Bench.Repairs.ReluTritonKernel.kernel, FP.Structural.exec]
#axiomsClean reference_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ReluTritonKernel.TypedValue
