import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.SplitGeometry
import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.InterGrid
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.SplitGrid
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGlaFwd
open ProducerGeometry (Shape bt_pos)
open SplitGeometry
open RealSplitMath (columns enabled rowStart rowActive)
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

def layout (c : Shape) : RealSplitMath.Layout := ⟨c.T*c.K,c.K,c.T,c.K,c.BT,16,c.BK,c.NC⟩
def regions : RealSplitProgram.Regions := ⟨"q","k","g","scratch"⟩
def kernel (c : Shape) (scale : ℝ) : Kernel :=
  (SourceBridge.splitNew (layout c) regions scale).eraseDType.toAlgKernel

theorem source_address (c : Shape) (l : RealSplitMath.Launch) (hl : l.heads=c.heads) (i : TileIndex [16,16]) :
    RealSplitMath.outputAddress (layout c) l i.1.val i.2.1.val=address c l i := by
  simp only [RealSplitMath.outputAddress,layout,address,row,RealSplitMath.rowStart,hl,Nat.add_mul,Nat.mul_assoc]
  omega

theorem launch_grid (c : Shape) (s : BlockState) (q : GridIndex (grid c)) :
    RealSplitProgram.launch (layout c) (s.withGridIndex q)=launchIndex c q := by
  simp [RealSplitProgram.launch,launchIndex,BlockState.withGridIndex,GridIndex.toPids,grid,Grid.rank,layout]

theorem perProgram (c : Shape) (scale : ℝ) (s : BlockState)
    (hu : s.undef=fun _ _ => 0) (q : GridIndex (grid c)) :
    ∃ t, exec (kernel c scale) (s.withGridIndex q)=some t ∧
      (enabled (layout c) (launchIndex c q) → ∀ j, j<columns (layout c) (launchIndex c q) →
        ∀ i : Fin 16, rowActive (layout c) (launchIndex c q) i.val →
        t.readMem "scratch" (RealSplitMath.outputAddress (layout c) (launchIndex c q) i.val j)=
          RealSplitMath.expected (layout c) (launchIndex c q) (Logical.splitInputs (InterGrid.inputs s)) scale i.val j) ∧
      BlockState.WriteWithin (RealSplitExecution.footprint (layout c) (launchIndex c q) "scratch") (s.withGridIndex q) t := by
  have h := LocalReal.split_executes (layout c) regions scale (s.withGridIndex q) (by simpa using hu)
    (by decide) (by decide)
  simpa only [kernel,launch_grid,regions,RealSplitSetup.inputs,Logical.splitInputs,InterGrid.inputs,
    BlockState.withGridIndex,BlockState.readMem] using h

def frames (c : Shape) (scale : ℝ) (s : BlockState) (hu : s.undef=fun _ _ => 0) :
    Kernel.GridFrames (kernel c scale) (grid c) s := fun q => {
  final := (perProgram c scale s hu q).choose
  writes := RealSplitExecution.footprint (layout c) (launchIndex c q) "scratch"
  h_exec := (perProgram c scale s hu q).choose_spec.1
  h_writeWithin := (perProgram c scale s hu q).choose_spec.2.2 }

theorem footprint_tile (c : Shape) (q : GridIndex (grid c)) (cell : RegionName×Nat)
    (hc : RealSplitExecution.footprint (layout c) (launchIndex c q) "scratch" cell) :
    cell.1="scratch" ∧ ∃ i : TileIndex [16,16], active c (launchIndex c q) i ∧ cell.2=address c (launchIndex c q) i := by
  rcases hc with ⟨hen,hr,j,hj,i,hi,he⟩
  have hj16 : j<16 := Nat.lt_of_lt_of_le hj (Nat.min_le_left _ _)
  let ij : TileIndex [16,16] := (i.1,⟨j,hj16⟩,⟨⟩)
  refine ⟨hr,ij,hi,?_⟩
  change cell.2=RealSplitMath.outputAddress (layout c) (launchIndex c q) ij.1.val ij.2.1.val at he
  rw [source_address c _ rfl] at he
  exact he

theorem disjoint (c : Shape) (scale : ℝ) (s : BlockState) (hu : s.undef=fun _ _ => 0) :
    Kernel.GridWritesDisjoint (frames c scale s hu) := by
  intro x y hne cell hx hy
  obtain ⟨_,i,hi,hei⟩ := footprint_tile c x cell hx
  obtain ⟨_,j,hj,hej⟩ := footprint_tile c y cell hy
  exact hne (address_injective c x y i j hi hj (hei.symm.trans hej)).1

def result (c : Shape) (scale : ℝ) (s : BlockState) (hu : s.undef=fun _ _ => 0) : BlockState :=
  Kernel.mergeFrames (grid c) s (frames c scale s hu)
def launched (c : Shape) (scale : ℝ) (s : BlockState) (hu : s.undef=fun _ _ => 0) :
    Kernel.GridLaunchedOrdinary (kernel c scale) (grid c) s (result c scale s hu) :=
  ⟨frames c scale s hu,disjoint c scale s hu,rfl⟩

def tileValue (c : Shape) (a : Logical.Inputs) (scale : ℝ) (f h t j : Nat) : ℝ :=
  RealSplitMath.expected (layout c) ⟨t/c.BT,t%c.BT/16,h,f,c.heads⟩
    (Logical.splitInputs a) scale (t%c.BT%16) j

theorem readback_streamed (c : Shape) (scale : ℝ) (s : BlockState) (hu : s.undef=fun _ _ => 0)
    (f : Fin (Geometry.ceilDiv c.K c.BK)) (h : Fin c.heads) (t : Fin c.T) (j : Fin 16)
    (hc : t.val/c.BT*c.BT+t.val%c.BT/16*16+j.val<c.T) :
    (result c scale s hu).readMem "scratch" (((f.val*c.heads+h.val)*c.T+t.val)*16+j.val)=
      tileValue c (InterGrid.inputs s) scale f.val h.val t.val j.val := by
  let q := owner c f h t
  let i := ownerLane c t j
  have hrow : rowStart (layout c) (launchIndex c q)+i.1.val=t.val := owner_row c f h t j
  have ht := t.isLt
  have hen : enabled (layout c) (launchIndex c q) := by unfold enabled; change _<c.T; omega
  have hi : rowActive (layout c) (launchIndex c q) i.1.val := by unfold rowActive; rw [hrow]; exact ht
  have hjc : j.val<columns (layout c) (launchIndex c q) := by
    have hj := j.isLt
    simp only [q,owner_launch,columns,layout]
    omega
  have hv := (perProgram c scale s hu q).choose_spec.2.1 hen j.val hjc i.1 hi
  change (frames c scale s hu q).final.readMem _ _ = _ at hv
  have hw : (frames c scale s hu q).writes ("scratch",
      RealSplitMath.outputAddress (layout c) (launchIndex c q) i.1.val j.val) :=
    ⟨hen,rfl,j.val,hjc,(i.1,⟨⟩),hi,rfl⟩
  have hm := Kernel.mergeFrames_mem_written (disjoint c scale s hu) q _ _ hw
  have hv' : (result c scale s hu).readMem "scratch"
      (RealSplitMath.outputAddress (layout c) (launchIndex c q) i.1.val j.val)=
      RealSplitMath.expected (layout c) (launchIndex c q) (Logical.splitInputs (InterGrid.inputs s)) scale i.1.val j.val := by
    unfold result BlockState.readMem
    rw [hm]
    exact hv
  change (result c scale s hu).readMem "scratch"
      (RealSplitMath.outputAddress (layout c) (launchIndex c q) i.1.val i.2.1.val)=_ at hv'
  rw [source_address c _ rfl] at hv'
  dsimp only [q,i] at hv'
  rw [owner_address] at hv'
  simpa only [owner_launch,ownerLane,tileValue] using hv'

theorem unwritten (c : Shape) (scale : ℝ) (s : BlockState) (hu : s.undef=fun _ _ => 0)
    (r : RegionName) (off : Nat)
    (hn : ∀ q, ¬RealSplitExecution.footprint (layout c) (launchIndex c q) "scratch" (r,off)) :
    (result c scale s hu).mem r off=s.mem r off := by
  apply Kernel.mergeFrames_mem_eq_of_not_written
  rintro ⟨q,hq⟩
  exact hn q hq

#specNonCircular tileValue avoiding [exec]
#axiomsClean readback_streamed
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.SplitGrid
