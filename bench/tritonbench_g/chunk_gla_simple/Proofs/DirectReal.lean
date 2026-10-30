import bench.tritonbench_g.chunk_gla_simple.Sources.Execution
import bench.tritonbench_g.chunk_gla_simple.Kernels
import bench.tritonbench_g.chunk_gla_simple.Proofs.Layout

namespace VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.DirectReal
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGlaSimple Math Real
open VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.Host
noncomputable section
set_option maxHeartbeats 8000000

def kernel (c : Shape) (scale : ℝ) : Kernel :=
  (ChunkGlaSimpleMinDotTile.kernel c .fp32 .fp32 .fp32 .fp32 scale).eraseDType.toAlgKernel

/-- Candidate BK specialization of the unchanged generic readable source.
Only the candidate Shape is used; the predecessor's small-BK host instance is not reused. -/
theorem source_binding (c : Shape) (scale : ℝ) :
    ChunkGlaSimpleMinDotTile.kernel c .fp32 .fp32 .fp32 .fp32 scale =
      VeriTile.Bench.Repairs.ChunkGlaSimple.Program.typed (layout c) regions scale := rfl

theorem perProgram (c : Shape) (scale : ℝ) (s : BlockState) (idx : GridIndex (grid c)) :
    ∃ t, exec (kernel c scale) (s.withGridIndex idx)=some t ∧
      (∀ i : TileIndex [c.BT,c.BV],active (layout c) (launchIndex c idx) i.1.val i.2.1.val →
        t.readMem "o" (outputAddress (layout c) (launchIndex c idx) i.1.val i.2.1.val)=
          expected (layout c) (launchIndex c idx) (inputs regions s) scale i.1.val i.2.1.val) ∧
      BlockState.WriteWithin (Real.footprint (layout c) (launchIndex c idx) regions) (s.withGridIndex idx) t := by
  have h:=VeriTile.Bench.Repairs.ChunkGlaSimple.Execution.region_executes (layout c) regions scale (s.withGridIndex idx) (noCollision c _)
  simpa only [kernel,launch_grid] using h

def frames (c : Shape) (scale : ℝ) (s : BlockState) : Kernel.GridFrames (kernel c scale) (grid c) s :=
  fun idx => {
    final:=(perProgram c scale s idx).choose
    writes:=Real.footprint (layout c) (launchIndex c idx) regions
    h_exec:=(perProgram c scale s idx).choose_spec.1
    h_writeWithin:=(perProgram c scale s idx).choose_spec.2.2 }

theorem disjoint (c : Shape) (scale : ℝ) (s : BlockState) : Kernel.GridWritesDisjoint (frames c scale s) := by
  intro x y ne cell hx hy
  rcases hx with ⟨_,i,hi,hei⟩
  rcases hy with ⟨_,j,hj,hej⟩
  exact ne (launchIndex_injective c (address_injective c _ _ i j hi hj (hei.symm.trans hej)).1)

def footprint (c : Shape) : WriteFootprint := fun cell => cell.1="o" ∧ cell.2<c.heads*c.T*c.V

theorem frames_bound (c : Shape) (scale : ℝ) (s : BlockState) (cell : RegionName×Nat)
    (hw : Kernel.GridWriteFootprint (frames c scale s) cell) : footprint c cell := by
  rcases hw with ⟨idx,hr,i,hi,he⟩
  refine ⟨hr,?_⟩
  rw [he]
  simpa [outputAddress,layout] using triple_lt (launch_lt c idx).2.2 hi.1 hi.2

def result (c : Shape) (scale : ℝ) (s : BlockState) : BlockState :=
  Kernel.mergeFrames (grid c) s (frames c scale s)

theorem source (c : Shape) (scale : ℝ) (s : BlockState) :
    Nonempty (Kernel.GridLaunchedOrdinary (kernel c scale) (grid c) s (result c scale s)) :=
  ⟨⟨frames c scale s,disjoint c scale s,rfl⟩⟩

theorem readback (c : Shape) (ht : 0<c.BT) (scale : ℝ) (s : BlockState)
    (h : Fin c.heads) (t : Fin c.T) (v : Fin c.V) :
    (result c scale s).readMem "o" (h.val*(c.T*c.V)+t.val*c.V+v.val)=
      Host.formula c ht (inputs regions s) scale h t v := by
  let idx:=owner c ht h t v
  let i:=lane c ht t v
  have hi:=owner_active c ht h t v
  have hw : (frames c scale s idx).writes ("o",outputAddress (layout c) (launchIndex c idx) i.1.val i.2.1.val) :=
    ⟨rfl,i,hi,rfl⟩
  have hm:=Kernel.mergeFrames_mem_written (disjoint c scale s) idx _ _ hw
  have hv:=(perProgram c scale s idx).choose_spec.2.1 i hi
  change (frames c scale s idx).final.readMem _ _=_ at hv
  have value : (result c scale s).readMem "o" (outputAddress (layout c) (launchIndex c idx) i.1.val i.2.1.val)=
      expected (layout c) (launchIndex c idx) (inputs regions s) scale i.1.val i.2.1.val := by
    unfold BlockState.readMem result
    rw [hm]
    exact hv
  simpa only [idx,i,owner_address,Host.formula] using value

theorem frame (c : Shape) (scale : ℝ) (s : BlockState) :
    BlockState.WriteWithin (footprint c) s (result c scale s) := by
  intro region off miss
  exact Kernel.mergeFrames_writeWithin (frames c scale s) region off
    (fun hw => miss (frames_bound c scale s _ hw))

/-- Full repaired regional source grid and independent formula. Read-only
input aliasing is handled by physical projection/commit, not by input copies. -/
theorem complete (c : Shape) (ht : 0<c.BT) (scale : ℝ) (s : BlockState) :
    Nonempty (Kernel.GridLaunchedOrdinary (kernel c scale) (grid c) s (result c scale s)) ∧
    (∀ h t v,(result c scale s).readMem "o" (h.val*(c.T*c.V)+t.val*c.V+v.val)=
      Host.formula c ht (inputs regions s) scale h t v) ∧
    BlockState.WriteWithin (footprint c) s (result c scale s) :=
  ⟨source c scale s,readback c ht scale s,frame c scale s⟩

#axiomsClean complete
end
end VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.DirectReal
