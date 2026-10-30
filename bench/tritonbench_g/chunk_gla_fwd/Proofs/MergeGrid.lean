import bench.tritonbench_g.chunk_gla_fwd.Proofs.SplitGrid
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.MergeGrid
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGlaFwd
open ProducerGeometry (Shape bt_pos)
open DiagonalGeometry (grid owner lane)
open scoped BigOperators
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

def layout (c : Shape) : RealMergeProgram.Layout := ⟨c.T,c.BT,16,Geometry.ceilDiv c.K c.BK⟩
def launchIndex (c : Shape) (q : GridIndex (grid c)) : RealMergeProgram.Launch :=
  let l := DiagonalGeometry.launchIndex c q
  ⟨l.chunk,l.block,l.head,c.heads⟩
def kernel (c : Shape) : Kernel :=
  (SourceBridge.mergeNew (layout c) "scratch" "attention").eraseDType.toAlgKernel

theorem launch_grid (c : Shape) (s : BlockState) (q : GridIndex (grid c)) :
    RealMergeProgram.launch (s.withGridIndex q)=launchIndex c q := by
  simp [RealMergeProgram.launch,launchIndex,DiagonalGeometry.launchIndex,BlockState.withGridIndex,
    GridIndex.toPids,grid,Grid.rank]

theorem source_address (c : Shape) (q : GridIndex (grid c)) (i : TileIndex [16,16]) :
    RealMergeProgram.outputAddress (layout c) (launchIndex c q) i=
      DiagonalGeometry.address c (DiagonalGeometry.launchIndex c q) i := rfl

theorem source_active (c : Shape) (q : GridIndex (grid c)) (i : TileIndex [16,16]) :
    RealMergeProgram.active (layout c) (launchIndex c q) i ↔
      DiagonalGeometry.active c (DiagonalGeometry.launchIndex c q) i := Iff.rfl

theorem noCollision (c : Shape) (q : GridIndex (grid c)) :
    RealMergeProgram.NoCollision (layout c) (launchIndex c q) := by
  intro i j hi hj he
  exact (DiagonalGeometry.address_injective c q q i j hi hj he).2

theorem perProgram (c : Shape) (s : BlockState) (hu : s.undef=fun _ _ => 0) (q : GridIndex (grid c)) :
    ∃ t, exec (kernel c) (s.withGridIndex q)=some t ∧
      (∀ i, RealMergeProgram.active (layout c) (launchIndex c q) i →
        t.readMem "attention" (RealMergeProgram.outputAddress (layout c) (launchIndex c q) i)=
          RealMergeProgram.expected (layout c) (launchIndex c q) (s.readMem "scratch") i) ∧
      BlockState.WriteWithin (RealMergeProgram.footprint (layout c) (launchIndex c q) "attention") (s.withGridIndex q) t := by
  have h := LocalReal.merge_executes (layout c) "scratch" "attention" (s.withGridIndex q)
    (by simpa using hu) (by rw [launch_grid]; exact noCollision c q)
  simpa only [kernel,launch_grid,BlockState.withGridIndex,BlockState.readMem] using h

def frames (c : Shape) (s : BlockState) (hu : s.undef=fun _ _ => 0) :
    Kernel.GridFrames (kernel c) (grid c) s := fun q => {
  final := (perProgram c s hu q).choose
  writes := RealMergeProgram.footprint (layout c) (launchIndex c q) "attention"
  h_exec := (perProgram c s hu q).choose_spec.1
  h_writeWithin := (perProgram c s hu q).choose_spec.2.2 }

theorem disjoint (c : Shape) (s : BlockState) (hu : s.undef=fun _ _ => 0) :
    Kernel.GridWritesDisjoint (frames c s hu) := by
  intro x y hne cell hx hy
  rcases hx with ⟨_,_,i,hi,hei⟩
  rcases hy with ⟨_,_,j,hj,hej⟩
  exact hne (DiagonalGeometry.address_injective c x y i j hi hj (hei.symm.trans hej)).1

def result (c : Shape) (s : BlockState) (hu : s.undef=fun _ _ => 0) : BlockState :=
  Kernel.mergeFrames (grid c) s (frames c s hu)
def launched (c : Shape) (s : BlockState) (hu : s.undef=fun _ _ => 0) :
    Kernel.GridLaunchedOrdinary (kernel c) (grid c) s (result c s hu) :=
  ⟨frames c s hu,disjoint c s hu,rfl⟩

theorem readback (c : Shape) (s : BlockState) (hu : s.undef=fun _ _ => 0)
    (h : Fin c.heads) (t : Fin c.T) (j : Fin c.BT) (hsame : j.val/16=t.val%c.BT/16) :
    (result c s hu).readMem "attention" (h.val*c.T*c.BT+t.val*c.BT+j.val)=
      ∑ f ∈ Finset.range (Geometry.ceilDiv c.K c.BK),
        s.readMem "scratch" (((f*c.heads+h.val)*c.T+t.val)*16+j.val%16) := by
  let q := owner c h t
  let i := lane c t j
  have hi : RealMergeProgram.active (layout c) (launchIndex c q) i := DiagonalGeometry.owner_active c h t j hsame
  have hen : RealMergeProgram.enabled (layout c) (launchIndex c q) := by
    have ht := hi.1
    unfold RealMergeProgram.enabled
    omega
  have hw : (frames c s hu q).writes ("attention",RealMergeProgram.outputAddress (layout c) (launchIndex c q) i) :=
    ⟨rfl,hen,i,hi,rfl⟩
  have hm := Kernel.mergeFrames_mem_written (disjoint c s hu) q _ _ hw
  have hv := (perProgram c s hu q).choose_spec.2.1 i hi
  change (frames c s hu q).final.readMem _ _ = _ at hv
  have hv' : (result c s hu).readMem "attention" (RealMergeProgram.outputAddress (layout c) (launchIndex c q) i)=
      RealMergeProgram.expected (layout c) (launchIndex c q) (s.readMem "scratch") i := by
    unfold result BlockState.readMem
    rw [hm]
    exact hv
  rw [source_address] at hv'
  dsimp only [q,i] at hv'
  rw [DiagonalGeometry.owner_address c h t j hsame] at hv'
  rw [hv']
  apply Finset.sum_congr rfl
  intro f _
  congr 1
  have hr := DiagonalGeometry.owner_row c h t j
  change _=t.val at hr
  simp only [RealMergeProgram.inputAddress,layout,launchIndex]
  change (h.val+f*c.heads)*c.T*16+(DiagonalGeometry.row c (DiagonalGeometry.launchIndex c q) i.1.val)*16+i.2.1.val= _
  dsimp only [q,i]
  rw [DiagonalGeometry.owner_row]
  simp only [lane,Nat.add_mul]
  omega

theorem unwritten (c : Shape) (s : BlockState) (hu : s.undef=fun _ _ => 0)
    (r : RegionName) (off : Nat)
    (hn : ∀ q, ¬RealMergeProgram.footprint (layout c) (launchIndex c q) "attention" (r,off)) :
    (result c s hu).mem r off=s.mem r off := by
  apply Kernel.mergeFrames_mem_eq_of_not_written
  rintro ⟨q,hq⟩
  exact hn q hq

#axiomsClean readback
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.MergeGrid
