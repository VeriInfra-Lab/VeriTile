import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.MathGrid
namespace VeriTile.Bench.Repairs.LayerNormOps.MathReturn
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.LayerNormLiger
noncomputable section
set_option maxHeartbeats 1000000

theorem loop_region_frame (M : Algebra α) [Inhabited α] (p : BackwardParameters)
    (s : State α) (r : RegionName) (o : Nat) (hx : r≠p.dx) (hy : r≠p.y) (hr : r≠p.dresIn) :
    (Backward.accumulated M p s).state.mem r o=s.mem r o := by
  apply Folds.preserves (Backward.update M p) (fun a => a.state.mem r o=s.mem r o)
  · intro row a ha
    rw [update_region_frame _ _ _ _ _ _ hx hy hr]
    exact ha
  · rfl

theorem partial_frame (M : Algebra α) [Inhabited α] (p : BackwardParameters)
    (s : State α) (r : RegionName) (o : Nat) (hx : r≠p.dx) (hy : r≠p.y) (hr : r≠p.dresIn)
    (hw : r≠p.dw ∨ ∀ i : TileIndex [p.block], i.1.val<p.columns → o≠s.pids 0*p.columns+i.1.val)
    (hb : r≠p.db ∨ ∀ i : TileIndex [p.block], i.1.val<p.columns → o≠s.pids 0*p.columns+i.1.val) :
    (Backward.reference M p s).mem r o=s.mem r o := by
  unfold Backward.reference Backward.finish
  cases h : p.hasBias
  · simp only [h,Bool.false_eq_true,if_false]
    rw [Backward.scatter_frame _ _ _ _ _ _ _ _ hw]
    exact loop_region_frame _ _ _ _ _ hx hy hr
  · simp only [if_true]
    rw [Backward.scatter_frame _ _ _ _ _ _ _ _ hb,Backward.scatter_frame _ _ _ _ _ _ _ _ hw]
    exact loop_region_frame _ _ _ _ _ hx hy hr

theorem grid_dw (rest : Algebra ℝ) (p : BackwardParameters) (s : State ℝ)
    (programs pid : Nat) (hp : pid<programs)
    (fresh : ∀ r ∈ gridReads p, Backward.RegionOutside p r)
    (distinct : [p.dx,p.dw,p.db,p.y,p.dresIn].Nodup)
    (i : TileIndex [p.block]) (hi : i.1.val<p.columns) :
    (Grid.reference (Backward.reference (Logical.algebra rest) p) s programs).mem
      p.dw (pid*p.columns+i.1.val)=.mk .fp32
        (∑ row ∈ Finset.Ico (pid*p.rowsPerProgram) (min ((pid+1)*p.rowsPerProgram) p.rows), Logical.rowDW p s row i.1.val) := by
  have hf : Logical.InputsFresh p := fun r hr => fresh r (by simp only [List.mem_cons,List.mem_singleton] at hr; simp [gridReads,backwardReads]; tauto)
  have hwb : p.dw≠p.db := by simp_all
  have hwx : p.dw≠p.dx := by simp_all [ne_comm]
  have hwy : p.dw≠p.y := by simp_all
  have hwr : p.dw≠p.dresIn := by simp_all
  rw [Grid.readback _ s programs pid hp p.dw (pid*p.columns+i.1.val) ?_]
  · change (Backward.finish (Logical.algebra rest) p pid
      (Backward.accumulated (Logical.algebra rest) p (before rest p s pid))).mem _ _ = _
    rw [Backward.finish_dw _ _ _ _ i hi hwb,Logical.loop_dw rest p _ hf i hi]
    change Cell.mk .fp32 _=Cell.mk .fp32 _
    congr 1
    simp only [Backward.start,Backward.stop,before,Grid.launch,if_true,Logical.algebra,LayerNormLiger.Logical.algebra]
    apply Finset.sum_congr rfl
    intro row hrow
    exact (grid_values rest p s pid row i.1.val fresh).2.2.1
  · intro later hl hu mem
    apply partial_frame _ _ _ _ _ hwx hwy hwr (Or.inr ?_) (Or.inl hwb)
    intro j hj
    simp only [Grid.launch,if_true]
    exact Grid.row_offset_ne _ _ _ hi hj pid later (by omega)

theorem grid_db (rest : Algebra ℝ) (p : BackwardParameters) (s : State ℝ)
    (programs pid : Nat) (hp : pid<programs)
    (fresh : ∀ r ∈ gridReads p, Backward.RegionOutside p r)
    (distinct : [p.dx,p.dw,p.db,p.y,p.dresIn].Nodup) (hb : p.hasBias=Bool.true)
    (i : TileIndex [p.block]) (hi : i.1.val<p.columns) :
    (Grid.reference (Backward.reference (Logical.algebra rest) p) s programs).mem
      p.db (pid*p.columns+i.1.val)=.mk .fp32
        (∑ row ∈ Finset.Ico (pid*p.rowsPerProgram) (min ((pid+1)*p.rowsPerProgram) p.rows), Logical.rowDB p s row i.1.val) := by
  have hf : Logical.InputsFresh p := fun r hr => fresh r (by simp only [List.mem_cons,List.mem_singleton] at hr; simp [gridReads,backwardReads]; tauto)
  have hbw : p.db≠p.dw := by simp_all [ne_comm]
  have hbx : p.db≠p.dx := by simp_all [ne_comm]
  have hby : p.db≠p.y := by simp_all
  have hbr : p.db≠p.dresIn := by simp_all
  rw [Grid.readback _ s programs pid hp p.db (pid*p.columns+i.1.val) ?_]
  · change (Backward.finish (Logical.algebra rest) p pid
      (Backward.accumulated (Logical.algebra rest) p (before rest p s pid))).mem _ _ = _
    rw [Backward.finish_db _ _ _ _ i hi hb,Logical.loop_db rest p _ hf i hi hb]
    change Cell.mk .fp32 _=Cell.mk .fp32 _
    congr 1
    simp only [Backward.start,Backward.stop,before,Grid.launch,if_true,Logical.algebra,LayerNormLiger.Logical.algebra]
    apply Finset.sum_congr rfl
    intro row hrow
    exact (grid_values rest p s pid row i.1.val fresh).2.2.2
  · intro later hl hu mem
    apply partial_frame _ _ _ _ _ hbx hby hbr (Or.inl hbw) (Or.inr ?_)
    intro j hj
    simp only [Grid.launch,if_true]
    exact Grid.row_offset_ne _ _ _ hi hj pid later (by omega)

def scalar : Host.Scalar ℝ := ⟨0,(·+·),id⟩

theorem tree_range (n : Nat) (p : ReductionPlan.Plan n) (f : Nat → ℝ) :
    p.tree.eval 0 (·+·) (fun i => f i.val)=∑ i ∈ Finset.range n, f i := by
  rw [p.real_sum,← Fin.sum_univ_def,Fin.sum_univ_eq_sum_range]

theorem return_dw (rest : Algebra ℝ) (fp : ForwardParameters) (p : BackwardParameters)
    (s : State ℝ) (rows programs : Nat) (scope : Scope.Supported fp p rows programs)
    (fresh : ∀ r ∈ gridReads p, Backward.RegionOutside p r)
    (distinct : [p.dx,p.dw,p.db,p.y,p.dresIn].Nodup)
    (pw pb : ReductionPlan.Plan programs) (dw db : RegionName) (res : Bool) (hout : dw≠db)
    (i : Fin p.columns) :
    (Host.backwardFinish scalar p programs pw pb dw db res
      (Grid.reference (Backward.reference (Logical.algebra rest) p) s programs)).state.mem dw i.val=
      .mk .fp32 (∑ row ∈ Finset.range rows, Logical.rowDW p s row i.val) := by
  rw [Host.finish_dw _ _ _ _ _ _ _ _ _ i hout]
  have hb : p.columns≤p.block := by rw [scope.sameColumns,scope.sameBlock]; exact scope.blockCovers
  let j : TileIndex [p.block] := (⟨i.val,Nat.lt_of_lt_of_le i.isLt hb⟩,PUnit.unit)
  have hcol : LayerNormLiger.Host.column (Grid.reference (Backward.reference (Logical.algebra rest) p) s programs)
      (Host.partialMatrix p p.dw programs) i.val=(fun pid =>
        ∑ row ∈ Finset.Ico (pid.val*p.rowsPerProgram) (min ((pid.val+1)*p.rowsPerProgram) p.rows), Logical.rowDW p s row i.val) := by
    funext pid
    dsimp [LayerNormLiger.Host.column,Host.partialMatrix]
    rw [grid_dw rest p s programs pid.val pid.isLt fresh distinct j i.isLt]
    rfl
  unfold LayerNormLiger.Host.sumValue
  rw [hcol]
  change Cell.mk .fp32 (pw.tree.eval 0 (·+·) _)=_
  exact congrArg (Cell.mk .fp32) ((tree_range programs pw (fun pid => ∑ row ∈ Finset.Ico (pid*p.rowsPerProgram) (min ((pid+1)*p.rowsPerProgram) p.rows), Logical.rowDW p s row i.val)).trans (Scope.partition_sum fp p rows programs scope (fun row => Logical.rowDW p s row i.val)))

theorem return_db (rest : Algebra ℝ) (fp : ForwardParameters) (p : BackwardParameters)
    (s : State ℝ) (rows programs : Nat) (scope : Scope.Supported fp p rows programs)
    (fresh : ∀ r ∈ gridReads p, Backward.RegionOutside p r)
    (distinct : [p.dx,p.dw,p.db,p.y,p.dresIn].Nodup) (hb : p.hasBias=Bool.true)
    (pw pb : ReductionPlan.Plan programs) (dw db : RegionName) (res : Bool) (hout : p.db≠dw)
    (i : Fin p.columns) :
    (Host.backwardFinish scalar p programs pw pb dw db res
      (Grid.reference (Backward.reference (Logical.algebra rest) p) s programs)).state.mem db i.val=
      .mk .fp32 (∑ row ∈ Finset.range rows, Logical.rowDB p s row i.val) := by
  rw [Host.finish_db _ _ _ _ _ _ _ _ _ i hb hout]
  have hblock : p.columns≤p.block := by rw [scope.sameColumns,scope.sameBlock]; exact scope.blockCovers
  let j : TileIndex [p.block] := (⟨i.val,Nat.lt_of_lt_of_le i.isLt hblock⟩,PUnit.unit)
  have hcol : LayerNormLiger.Host.column (Grid.reference (Backward.reference (Logical.algebra rest) p) s programs)
      (Host.partialMatrix p p.db programs) i.val=(fun pid =>
        ∑ row ∈ Finset.Ico (pid.val*p.rowsPerProgram) (min ((pid.val+1)*p.rowsPerProgram) p.rows), Logical.rowDB p s row i.val) := by
    funext pid
    dsimp [LayerNormLiger.Host.column,Host.partialMatrix]
    rw [grid_db rest p s programs pid.val pid.isLt fresh distinct hb j i.isLt]
    rfl
  unfold LayerNormLiger.Host.sumValue
  rw [hcol]
  change Cell.mk .fp32 (pb.tree.eval 0 (·+·) _)=_
  exact congrArg (Cell.mk .fp32) ((tree_range programs pb (fun pid => ∑ row ∈ Finset.Ico (pid*p.rowsPerProgram) (min ((pid+1)*p.rowsPerProgram) p.rows), Logical.rowDB p s row i.val)).trans (Scope.partition_sum fp p rows programs scope (fun row => Logical.rowDB p s row i.val)))

#axiomsClean grid_dw
#axiomsClean grid_db
#axiomsClean return_dw
#axiomsClean return_db
end
end VeriTile.Bench.Repairs.LayerNormOps.MathReturn
