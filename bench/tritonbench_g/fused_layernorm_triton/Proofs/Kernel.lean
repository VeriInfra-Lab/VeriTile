import bench.tritonbench_g.fused_layernorm_triton.Proofs.NativeIndex
import bench.tritonbench_g.fused_layernorm_triton.Proofs.ReadableWhole

namespace VeriTile.Bench.Repairs.FusedLayernormTriton.KernelProof
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.FusedLayernormTriton FPStages Whole
noncomputable section

abbrev legacyTree := @ReadableStatistics.legacyTree

theorem leaves {N : Nat} (t : Kernels.ReductionTree N) :
    (legacyTree t).leaves = t.leaves := by
  induction t with
  | leaf i => rfl
  | node l r hl hr => simp [legacyTree, ReadableStatistics.legacyTree, ReductionTree.leaves, Kernels.ReductionTree.leaves, hl, hr]

theorem complete {N : Nat} (t : Kernels.ReductionTree N) (h : t.Complete) :
    (legacyTree t).Complete := by
  simpa only [ReductionTree.Complete, leaves] using h

def configuration (c : Contract.Configuration) : Whole.Configuration :=
  ⟨c.block,legacyTree c.tree,c.allowed⟩

/-- Successful selected kernel launch. The Python wrapper is outside this run. -/
def launch {α : Type} [Inhabited α] (M : Algebra α) (eq32 : α → α → Bool)
    (X W Bias Y Mean Rstd : RegionName) (S D : Nat) (c : Contract.Configuration)
    (s : State α) : Option (State α) :=
  runRows (withEquality M eq32) (Contract.source X W Bias Y Mean Rstd S D c) (List.range S) s

theorem succeeds {α : Type} [Inhabited α] (M : Algebra α) (eq32 : α → α → Bool)
    (X W Bias Y Mean Rstd : RegionName) (S D : Nat) (c : Contract.Configuration)
    (s : State α) (sep : Separation X W Bias Y Mean Rstd) :
    ∃ final, launch M eq32 X W Bias Y Mean Rstd S D c s = some final ∧
      GridEffect M eq32 s final X W Bias Y Mean Rstd D c.block (legacyTree c.tree) (List.range S) := by
  exact ReadableWhole.whole_grid Bool.false M eq32 s X W Bias Y Mean Rstd S D c.block c.tree c.positive sep _

def launchSource {α : Type} [Inhabited α] (hoist : Bool) (M : Algebra α) (eq32 : α → α → Bool)
    (X W Bias Y Mean Rstd : RegionName) (S D : Nat) (c : Contract.Configuration)
    (s : State α) : Option (State α) :=
  runRows (withEquality M eq32)
    (SourcePieces.source hoist X W Bias Y Mean Rstd S D c.block c.tree) (List.range S) s

theorem succeedsSource {α : Type} [Inhabited α] (hoist : Bool) (M : Algebra α) (eq32 : α → α → Bool)
    (X W Bias Y Mean Rstd : RegionName) (S D : Nat) (c : Contract.Configuration)
    (s : State α) (sep : Separation X W Bias Y Mean Rstd) :
    ∃ final,launchSource hoist M eq32 X W Bias Y Mean Rstd S D c s=some final ∧
      GridEffect M eq32 s final X W Bias Y Mean Rstd D c.block (legacyTree c.tree) (List.range S) :=
  ReadableWhole.whole_grid hoist M eq32 s X W Bias Y Mean Rstd S D c.block c.tree c.positive sep _

#axiomsClean succeeds
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FusedLayernormTriton.KernelProof
