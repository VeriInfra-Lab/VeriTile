import VeriTile.Triton.Float.Loop
import bench.repairs.tritonbench_g.relu_strided_buffer.Proofs.Source
import bench.tritonbench_g.relu_strided_buffer.Proofs.StructuralHost

/-! Complete execution of the shared kernel source. These statements quantify
kernel programs and their cells; they contain no wrapper, allocation or return
identity claim. Existing signed tile/grid readback and frame proofs are reused. -/
namespace VeriTile.Bench.Repairs.ReluStridedBuffer
open VeriTile.Triton FP.Structural
open TritonBenchG.ReluStridedBuffer
open SignedView StructuralTile StructuralGrid StructuralControl
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

theorem kernel_execution [Inhabited α] (M : Algebra α) (cmp : α → α → Bool)
    (hc : M.compareLt none .fp32 = some cmp) (c : Parameters) (s : State α) :
    ∃ actualFinal flatFinal,
      exec M (kernel c) s = some actualFinal ∧
      runTiles (flatTile (tileOutput M cmp (views c))) (kernelIds c s) s = some flatFinal ∧
      ObsEq actualFinal flatFinal := by
  cases branch : c.oneTile
  · obtain ⟨actual,flat,ran,flatRan,obs⟩ := grid_branch_execution M
      (tileOutput M cmp (views c)) (perTile c) (tile_implementation M cmp hc c)
      c.n c.B c.tiles s
    refine ⟨actual,flat,?_,?_,obs⟩
    · simpa [kernel,branch,gridStrideKernel,perTile,oneTileKernel,
        FP.Structural.exec,StructuralControl.gridBody,StructuralControl.loopBody,SourceForRange.tileIdStmt,SourceForRange.numTilesStmt,
        ComputeKernel.surfaceBody,run,step,stepAlg,evalExpr,evalOp_unfold,numeric,range_assign] using ran
    · simpa only [kernelIds,branch,Bool.false_eq_true,if_false] using flatRan
  · obtain ⟨actual,ran,obs⟩ := one_tile_execution M
      (tileOutput M cmp (views c)) (perTile c) (tile_implementation M cmp hc c) c.n c.B s
    refine ⟨actual,flatState (tileOutput M cmp (views c)) (s.pids 0) s,?_,?_,obs⟩
    · simpa [kernel,branch,oneTileKernel,perTile,FP.Structural.exec,StructuralControl.oneTileBody,
        SourceForRange.numTilesStmt,ComputeKernel.surfaceBody,run,step,stepAlg,evalExpr,evalOp_unfold,numeric,range_assign] using ran
    · simp [kernelIds,branch,runTiles,flatTile]

/-- The displayed signed-pointer DSL is an observational lowering of the
historical FP32 block-pointer source: successful execution, all memory cells,
and arbitrary masked-load observations agree. Only private pointer registers
change from cell indices to signed pre-conversion offsets. -/
theorem signed_source_lowering [Inhabited α] (M : Algebra α) (cmp : α → α → Bool)
    (hc : M.compareLt none .fp32 = some cmp) (c : Parameters) (s : State α) :
    ∃ displayed historical,
      FP.Structural.exec M (kernel c) s = some displayed ∧
      FP.Structural.exec M (StructuralTile.kernelSource (views c) c.oneTile c.tiles) s = some historical ∧
      displayed.mem = historical.mem ∧ displayed.undef = historical.undef := by
  obtain ⟨displayed,flat,ran,flatRan,obs⟩ := kernel_execution M cmp hc c s
  obtain ⟨historical,flat',oldRan,oldFlatRan,oldObs⟩ :=
    StructuralHost.kernel_execution M cmp hc (views c) c.oneTile c.tiles s
  have equal : flat = flat' := Option.some.inj (flatRan.symm.trans oldFlatRan)
  subst flat'
  exact ⟨displayed,historical,ran,oldRan,obs.1.trans oldObs.1.symm,obs.2.trans oldObs.2.symm⟩

/-- The candidate changes the unused integer cdiv to its host-folded constant.
Floating operations, masks, signed addresses and branch iteration order remain
identical. This exact discrete equality needs no numerical experiment. -/
theorem folded_execution [Inhabited α] (M : Algebra α) (c : Parameters) (s : State α) :
    exec M (kernel c) s = exec M (foldedKernel c) s := by
  cases branch : c.oneTile <;>
    simp [kernel,foldedKernel,branch,oneTileKernel,oneTileFolded,gridStrideKernel,gridStrideFolded,
      FP.Structural.exec,ComputeKernel.surfaceBody,run,step,stepAlg,evalExpr,evalOp_unfold,numeric] <;> rfl

/-- Single-CTA mathematical/value scope: every valid logical element covered
by this CTA's tiles is written, and all other cells preserve entry memory. -/
theorem kernel_result [Inhabited α] (M : Algebra α) (cmp : α → α → Bool)
    (hc : M.compareLt none .fp32 = some cmp) (c : Parameters)
    {capacity : Nat} (hout : (views c).outputView.Valid c.n capacity)
    (hstride : c.output.stride ≠ 0)
    (hsep : SignedGrid.FootprintsDisjoint (views c).inputView (views c).outputView c.n)
    (s : State α) :
    ∃ final, exec M (kernel c) s = some final ∧
      (∀ i, i < c.n → Covered c.B (kernelIds c s) i →
        final.mem c.output.region ((views c).outputView.address i) =
          Cell.mk .fp32 (scalarResult M cmp ((s.mem c.input.region ((views c).inputView.address i)).read .fp32))) ∧
      (∀ i, i < c.n → final.mem c.input.region ((views c).inputView.address i) =
        s.mem c.input.region ((views c).inputView.address i)) ∧
      (∀ r a, (∀ tile, tile ∈ kernelIds c s →
        r ≠ c.output.region ∨ ∀ lane : TileIndex [c.B], tile * c.B + lane.1.val < c.n →
          a ≠ (views c).outputView.address (tile * c.B + lane.1.val)) → final.mem r a = s.mem r a) := by
  obtain ⟨flat,flatRan,result,inputFrame,frame⟩ := tiles_success hout hstride hsep
    (fun x => Cell.mk .fp32 (scalarResult M cmp x)) (flatTile (tileOutput M cmp (views c)))
    (StructuralHost.tile_contract M cmp (views c) hout hstride) (kernelIds c s) s
  obtain ⟨actual,flat',ran,flatRan',obs⟩ := kernel_execution M cmp hc c s
  have equal : flat = flat' := Option.some.inj (flatRan.symm.trans flatRan')
  subst flat'
  exact ⟨actual,ran,by simpa only [obs.1] using result,
    by simpa only [obs.1] using inputFrame,by simpa only [obs.1] using frame⟩

/-- Kernel grid execution; input preparation and Python launch orchestration
are not modeled by this function. The supplied grid is the original capped
kernel grid, whose coverage theorem is reused below. -/
def runGrid [Inhabited α] (M : Algebra α) (a : HostPlan.Arguments) (s : State α) : Option (State α) :=
  runPrograms M (kernel (parameters a)).surfaceBody a.C (List.range a.C) s

theorem grid_execution [Inhabited α] (M : Algebra α) (cmp : α → α → Bool)
    (hc : M.compareLt none .fp32 = some cmp) (a : HostPlan.Arguments) (s : State α) :
    ∃ actualFinal flatFinal, runGrid M a s = some actualFinal ∧
      runTiles (flatTile (tileOutput M cmp (views (parameters a))))
        (StructuralHost.sourceTileIds a) s = some flatFinal ∧ ObsEq actualFinal flatFinal := by
  have hbody : ∀ pid state, ∃ actualFinal flatFinal,
      run M (kernel (parameters a)).surfaceBody (atProgram a.C pid state) = some actualFinal ∧
      runTiles (flatTile (tileOutput M cmp (views (parameters a))))
        (StructuralHost.idsPerProgram a pid) (atProgram a.C pid state) = some flatFinal ∧
      ObsEq actualFinal flatFinal := by
    intro pid state
    simpa only [FP.Structural.exec,kernelIds,parameters,atProgram,if_true,
      StructuralHost.idsPerProgram] using
      kernel_execution M cmp hc (parameters a) (atProgram a.C pid state)
  exact programs_execution M (tileOutput M cmp (views (parameters a))) a.C
    (kernel (parameters a)).surfaceBody (StructuralHost.idsPerProgram a) hbody
    (List.range a.C) s s (ObsEq.refl s)

/-- Every logical output, entry-relative input/exterior frames and access
bounds for the complete kernel grid. No host success premise is required. -/
theorem grid_result [Inhabited α] (M : Algebra α) (cmp : α → α → Bool)
    (hc : M.compareLt none .fp32 = some cmp) (input output : HostPlan.Buffer)
    {n inputCapacity outputCapacity : Nat} (hn : 0 < n) (sx sy : Int)
    (hin : (input.view sx).Valid n inputCapacity)
    (hout : (output.view sy).Valid n outputCapacity) (hstride : sy ≠ 0)
    (hsep : SignedGrid.FootprintsDisjoint (input.view sx) (output.view sy) n) (s : State α) :
    ∃ final, runGrid M (HostPlan.rankOneArguments input output n sx sy) s = some final ∧
      (∀ i, i < n → final.mem output.region ((output.view sy).address i) =
        Cell.mk .fp32 (scalarResult M cmp ((s.mem input.region ((input.view sx).address i)).read .fp32))) ∧
      (∀ i, i < n → final.mem input.region ((input.view sx).address i) =
        s.mem input.region ((input.view sx).address i)) ∧
      (∀ r a, (r ≠ output.region ∨ ∀ i, i < n → a ≠ (output.view sy).address i) →
        final.mem r a = s.mem r a) ∧
      (∀ i, i < n → (input.view sx).address i < inputCapacity ∧ (output.view sy).address i < outputCapacity) := by
  let a := HostPlan.rankOneArguments input output n sx sy
  let c := views (parameters a)
  obtain ⟨flat,hflat,hvalues,hinput,hframe,hbounds⟩ := capped_grid_success hin hout hstride
    (HostPlan.block_pos hn) hsep (fun x => Cell.mk .fp32 (scalarResult M cmp x))
    (flatTile (tileOutput M cmp c)) (StructuralHost.tile_contract M cmp c hout hstride)
    (cappedTileIds n (HostPlan.block n)) (List.Perm.refl _) s
  obtain ⟨actual,flat',hactual,hflat',hobs⟩ := grid_execution M cmp hc a s
  have equal : flat = flat' := Option.some.inj (hflat.symm.trans (by
    simpa only [a,StructuralHost.prepared_ids] using hflat'))
  subst flat'
  exact ⟨actual,hactual,by simpa only [hobs.1] using hvalues,
    by simpa only [hobs.1] using hinput,by simpa only [hobs.1] using hframe,hbounds⟩

def runFoldedGrid [Inhabited α] (M : Algebra α) (a : HostPlan.Arguments) (s : State α) : Option (State α) :=
  runPrograms M (foldedKernel (parameters a)).surfaceBody a.C (List.range a.C) s

theorem grid_folded_execution [Inhabited α] (M : Algebra α) (a : HostPlan.Arguments) (s : State α) :
    runGrid M a s = runFoldedGrid M a s := by
  have go : ∀ pids s, runPrograms M (kernel (parameters a)).surfaceBody a.C pids s =
      runPrograms M (foldedKernel (parameters a)).surfaceBody a.C pids s := by
    intro pids
    induction pids with
    | nil => intro s; rfl
    | cons pid rest ih =>
      intro s
      change (FP.Structural.exec M (kernel (parameters a)) (atProgram a.C pid s)).bind _ =
        (FP.Structural.exec M (foldedKernel (parameters a)) (atProgram a.C pid s)).bind _
      rw [folded_execution]
      congr 1
      funext t
      exact ih t
  exact go (List.range a.C) s

#axiomsClean kernel_execution
#axiomsClean signed_source_lowering
#axiomsClean folded_execution
#axiomsClean kernel_result
#axiomsClean grid_result
#axiomsClean grid_folded_execution
end
end VeriTile.Bench.Repairs.ReluStridedBuffer
