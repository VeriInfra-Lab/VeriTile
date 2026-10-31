import VeriTile.Triton.Float.Loop
import bench.tritonbench_g.pow_scalar_tensor.Proofs.Source
import bench.tritonbench_g.pow_scalar_tensor.Proofs.Mathematical_WholeHost

/-! Complete kernel execution, signed grid coverage and observational
integer-index commutation. All scalar primitives remain opaque and unchanged.
The existing signed tile/grid result and frame proofs are reused. -/
namespace VeriTile.Bench.Repairs.PowScalarTensor.Checked
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.PowScalarTensor WholeTile
open VeriTile.Bench.TritonBenchG.ReluStridedBuffer
open SignedView StructuralGrid StructuralControl
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 2000000

def kernelIds (c : Parameters) (s : State α) : List Nat :=
  if c.oneTile then [s.pids 0]
  else (List.range c.tiles).map (fun j => s.pids 0 + j * s.numPids 0)

private theorem range_assign [Inhabited α] (M : Algebra α)
    (idx : RegName) (start stop stride : Nat) (name : RegName)
    (e : Op dtype shape) (body : List ComputeStmt) (s : State α) :
    FP.Structural.range M idx start stop stride (.assign dtype shape name (.alg e) :: body) s =
      FP.Structural.range M idx start stop stride (.alg (.assign dtype shape name e) :: body) s := by
  apply range_congr_on_reachable
  intro cur t _ _ _
  simp only [run,step,stepAlg,evalExpr]

theorem kernel_execution [Inhabited α] (M : Algebra α) (c : Parameters)
    (candidate : Bool) (s : State α) :
    ∃ actualFinal flatFinal,
      FP.Structural.exec M (sourceKernel c candidate) s = some actualFinal ∧
      runTiles (flatTile (tileOutput M (config c candidate))) (kernelIds c s) s = some flatFinal ∧
      ObsEq actualFinal flatFinal := by
  cases branch : c.oneTile
  · obtain ⟨actual,flat,ran,flatRan,obs⟩ := grid_branch_execution M
      (tileOutput M (config c candidate)) (perTile c candidate) (tile_implementation M c candidate)
      c.n c.B c.tiles s
    refine ⟨actual,flat,?_,?_,obs⟩
    · cases candidate <;>
        simpa [sourceKernel,kernel,commutedKernel,branch,gridStrideKernel,gridStrideCommuted,
          perTile,oneTileKernel,oneTileCommuted,FP.Structural.exec,StructuralControl.gridBody,
          StructuralControl.loopBody,SourceForRange.tileIdStmt,SourceForRange.numTilesStmt,
          ComputeKernel.surfaceBody,run,step,stepAlg,evalExpr,evalOp_unfold,numeric,range_assign] using ran
    · simpa only [kernelIds,branch,Bool.false_eq_true,if_false] using flatRan
  · obtain ⟨actual,ran,obs⟩ := one_tile_execution M
      (tileOutput M (config c candidate)) (perTile c candidate) (tile_implementation M c candidate) c.n c.B s
    refine ⟨actual,flatState (tileOutput M (config c candidate)) (s.pids 0) s,?_,?_,obs⟩
    · cases candidate <;>
        simpa [sourceKernel,kernel,commutedKernel,branch,oneTileKernel,oneTileCommuted,perTile,
          FP.Structural.exec,StructuralControl.oneTileBody,SourceForRange.numTilesStmt,
          ComputeKernel.surfaceBody,run,step,stepAlg,evalExpr,evalOp_unfold,numeric] using ran
    · simp [kernelIds,branch,runTiles,flatTile]

/-- Both actual integer-index variants succeed and agree on every memory
cell, for arbitrary interpretations of the unchanged scalar operations. -/
theorem commuted_memory [Inhabited α] (M : Algebra α) (c : Parameters) (s : State α) :
    ∃ original commuted, FP.Structural.exec M (kernel c) s = some original ∧
      FP.Structural.exec M (commutedKernel c) s = some commuted ∧ original.mem = commuted.mem := by
  obtain ⟨original,flat,ran,flatRan,obs⟩ := kernel_execution M c Bool.false s
  obtain ⟨commuted,flat',ran',flatRan',obs'⟩ := kernel_execution M c Bool.true s
  have equal : flat = flat' := Option.some.inj (flatRan.symm.trans flatRan')
  subst flat'
  exact ⟨original,commuted,ran,ran',obs.1.trans obs'.1.symm⟩

def runGrid [Inhabited α] (M : Algebra α) (base : ℝ) (candidate : Bool)
    (a : HostPlan.Arguments) (s : State α) : Option (State α) :=
  runPrograms M (sourceKernel (parameters base a) candidate).surfaceBody a.C (List.range a.C) s

theorem grid_execution [Inhabited α] (M : Algebra α) (base : ℝ) (candidate : Bool)
    (a : HostPlan.Arguments) (s : State α) :
    ∃ actualFinal flatFinal, runGrid M base candidate a s = some actualFinal ∧
      runTiles (flatTile (tileOutput M (config (parameters base a) candidate)))
        (WholeSource.sourceTileIds a) s = some flatFinal ∧ ObsEq actualFinal flatFinal := by
  have hbody : ∀ pid state, ∃ actualFinal flatFinal,
      run M (sourceKernel (parameters base a) candidate).surfaceBody (atProgram a.C pid state) = some actualFinal ∧
      runTiles (flatTile (tileOutput M (config (parameters base a) candidate)))
        (WholeSource.idsPerProgram a pid) (atProgram a.C pid state) = some flatFinal ∧
      ObsEq actualFinal flatFinal := by
    intro pid state
    simpa only [FP.Structural.exec,kernelIds,parameters,atProgram,if_true,
      WholeSource.idsPerProgram] using
      kernel_execution M (parameters base a) candidate (atProgram a.C pid state)
  exact programs_execution M (tileOutput M (config (parameters base a) candidate)) a.C
    (sourceKernel (parameters base a) candidate).surfaceBody (WholeSource.idsPerProgram a) hbody
    (List.range a.C) s s (ObsEq.refl s)

/-- Full kernel grid: all logical outputs use entry input; input, exterior
cells and active-address bounds are preserved. No wrapper execution premise. -/
theorem grid_result [Inhabited α] (M : Algebra α) (base : ℝ) (candidate : Bool)
    (input output : HostPlan.Buffer) {n inputCapacity outputCapacity : Nat} (hn : 0 < n) (sx sy : Int)
    (hin : (input.view sx).Valid n inputCapacity)
    (hout : (output.view sy).Valid n outputCapacity) (hstride : sy ≠ 0)
    (hsep : SignedGrid.FootprintsDisjoint (input.view sx) (output.view sy) n) (s : State α) :
    ∃ final, runGrid M base candidate (HostPlan.rankOneArguments input output n sx sy) s = some final ∧
      (∀ i, i < n → final.mem output.region ((output.view sy).address i) =
        Cell.mk .fp32 (scalarResult M base ((s.mem input.region ((input.view sx).address i)).read .fp32))) ∧
      (∀ i, i < n → final.mem input.region ((input.view sx).address i) =
        s.mem input.region ((input.view sx).address i)) ∧
      (∀ r a, (r ≠ output.region ∨ ∀ i, i < n → a ≠ (output.view sy).address i) →
        final.mem r a = s.mem r a) ∧
      (∀ i, i < n → (input.view sx).address i < inputCapacity ∧ (output.view sy).address i < outputCapacity) := by
  let a := HostPlan.rankOneArguments input output n sx sy
  let c := config (parameters base a) candidate
  obtain ⟨flat,hflat,hvalues,hinput,hframe,hbounds⟩ := capped_grid_success hin hout hstride
    (HostPlan.block_pos hn) hsep (fun x => Cell.mk .fp32 (scalarResult M base x))
    (flatTile (tileOutput M c)) (WholeHost.tile_contract M c hout hstride)
    (cappedTileIds n (HostPlan.block n)) (List.Perm.refl _) s
  obtain ⟨actual,flat',hactual,hflat',hobs⟩ := grid_execution M base candidate a s
  have equal : flat = flat' := Option.some.inj (hflat.symm.trans (by
    simpa only [a,WholeHost.prepared_ids] using hflat'))
  subst flat'
  exact ⟨actual,hactual,by simpa only [hobs.1] using hvalues,
    by simpa only [hobs.1] using hinput,by simpa only [hobs.1] using hframe,hbounds⟩

#axiomsClean kernel_execution
#axiomsClean commuted_memory
#axiomsClean grid_result
end
end VeriTile.Bench.Repairs.PowScalarTensor.Checked
