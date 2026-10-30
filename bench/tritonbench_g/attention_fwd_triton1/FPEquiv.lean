import bench.tritonbench_g.attention_fwd_triton1.Proofs.Pair

namespace VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.FPEquiv
open VeriTile.Triton FP.Structural
open AttentionFwdTriton1.Whole001
open scoped VeriTile.Spec
noncomputable section
set_option autoImplicit false

/-- Complete original source schedule, all heads/chunks and both flags. -/
def original {α : Type} [Inhabited α] (ops : Ops α) (literal : ℝ → α)
    (c : Shape) (scale : ℝ) (storeFlag ifcond : Bool) :=
  ProgramIO.Implementation.original (KernelIO.io ops literal c scale storeFlag ifcond) (fun _ => True)

/-- Actual fixed-BT candidate: four temporal offsets use `i << 5`.
The source graph, native observer, typed IO and full grid otherwise coincide. -/
def shifted {α : Type} [Inhabited α] (ops : Ops α) (literal : ℝ → α)
    (c : Shape) (scale : ℝ) (storeFlag ifcond : Bool) :=
  ProgramIO.Implementation.ofVariant (Pair.variant ops literal c scale storeFlag ifcond) (fun _ => True)

/-- Exact integer strength reduction of the full repaired FP32 program.
No floating arithmetic rewrite or numerical input restriction is used. -/
specification grid_equiv {α : Type} [Inhabited α] (ops : Ops α) (literal : ℝ → α)
    (c : Shape) (scale : ℝ) (storeFlag ifcond : Bool) :
    original ops literal c scale storeFlag ifcond ≡[[]]
      shifted ops literal c scale storeFlag ifcond :=
  ProgramIO.Implementation.of_equivalent (Pair.variant ops literal c scale storeFlag ifcond)
    (Pair.equivalent ops literal c scale storeFlag ifcond)

#axiomsClean grid_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.FPEquiv
