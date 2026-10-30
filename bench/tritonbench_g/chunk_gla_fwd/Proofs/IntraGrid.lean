import bench.tritonbench_g.chunk_gla_fwd.Proofs.DiagonalGeometry
import bench.tritonbench_g.chunk_gla_fwd.Proofs.InterGrid
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.IntraGrid
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGlaFwd
open ProducerGeometry (Shape bt_pos)
open DiagonalGeometry
open RealIntraMath (columns enabled rowStart rowActive)
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

def layout (c : Shape) (BK : Nat) : RealIntraMath.Layout := ⟨c.T*c.K,c.K,c.T,c.K,c.BT,16,BK⟩
def regions : RealIntraProgram.Regions := ⟨"q","k","g","attention"⟩
def kernel (c : Shape) (BK : Nat) (scale : ℝ) : Kernel :=
  (SourceBridge.intraNew (layout c BK) regions scale).eraseDType.toAlgKernel

theorem source_address (c : Shape) (BK : Nat) (l : RealIntraMath.Launch) (i : TileIndex [16,16]) :
    RealIntraMath.outputAddress (layout c BK) l i.1.val i.2.1.val=address c l i := by
  simp [RealIntraMath.outputAddress,layout,address,row,column,RealIntraMath.rowStart,Nat.add_assoc]

theorem launch_grid (c : Shape) (s : BlockState) (q : GridIndex (grid c)) :
    RealIntraProgram.launch (s.withGridIndex q)=launchIndex c q := by
  simp [RealIntraProgram.launch,launchIndex,BlockState.withGridIndex,GridIndex.toPids,grid,Grid.rank]

theorem columns_le (c : Shape) (BK : Nat) (l : RealIntraMath.Launch) : columns (layout c BK) l ≤ c.BT := by
  have hp := c.nc_pos
  have he := c.bt_eq
  have hm : columns (layout c BK) l ≤ 16 := Nat.min_le_left _ _
  omega

theorem perProgram (c : Shape) (BK : Nat) (hK : c.K ≤ BK) (scale : ℝ) (s : BlockState)
    (hu : s.undef=fun _ _ => 0) (q : GridIndex (grid c)) :
    ∃ t, exec (kernel c BK scale) (s.withGridIndex q)=some t ∧
      (enabled (layout c BK) (launchIndex c q) → ∀ j, j<columns (layout c BK) (launchIndex c q) →
        ∀ i : Fin 16, rowActive (layout c BK) (launchIndex c q) i.val →
        t.readMem "attention" (RealIntraMath.outputAddress (layout c BK) (launchIndex c q) i.val j)=
          RealIntraMath.expected (layout c BK) (launchIndex c q) (Logical.intraInputs (InterGrid.inputs s)) scale i.val j) ∧
      BlockState.WriteWithin (RealIntraExecution.footprint (layout c BK) (launchIndex c q) "attention") (s.withGridIndex q) t := by
  have h := LocalReal.intra_executes (layout c BK) regions scale (s.withGridIndex q) (by simpa using hu)
    (by decide) (by decide) (columns_le c BK _) hK
  simpa only [kernel,launch_grid,regions,RealIntraSetup.inputs,Logical.intraInputs,InterGrid.inputs,
    BlockState.withGridIndex,BlockState.readMem] using h

def frames (c : Shape) (BK : Nat) (hK : c.K ≤ BK) (scale : ℝ) (s : BlockState) (hu : s.undef=fun _ _ => 0) :
    Kernel.GridFrames (kernel c BK scale) (grid c) s := fun q => {
  final := (perProgram c BK hK scale s hu q).choose
  writes := RealIntraExecution.footprint (layout c BK) (launchIndex c q) "attention"
  h_exec := (perProgram c BK hK scale s hu q).choose_spec.1
  h_writeWithin := (perProgram c BK hK scale s hu q).choose_spec.2.2 }

theorem footprint_tile (c : Shape) (BK : Nat) (q : GridIndex (grid c)) (cell : RegionName×Nat)
    (hc : RealIntraExecution.footprint (layout c BK) (launchIndex c q) "attention" cell) :
    cell.1="attention" ∧ ∃ i : TileIndex [16,16], active c (launchIndex c q) i ∧ cell.2=address c (launchIndex c q) i := by
  rcases hc with ⟨hen,hr,j,hj,i,hi,he⟩
  have hj16 : j<16 := Nat.lt_of_lt_of_le hj (Nat.min_le_left _ _)
  let ij : TileIndex [16,16] := (i.1,⟨j,hj16⟩,⟨⟩)
  refine ⟨hr,ij,⟨hi,local_bound c q ⟨j,hj16⟩⟩,?_⟩
  change cell.2=RealIntraMath.outputAddress (layout c BK) (launchIndex c q) ij.1.val ij.2.1.val at he
  rw [source_address] at he
  exact he

theorem disjoint (c : Shape) (BK : Nat) (hK : c.K ≤ BK) (scale : ℝ) (s : BlockState) (hu : s.undef=fun _ _ => 0) :
    Kernel.GridWritesDisjoint (frames c BK hK scale s hu) := by
  intro x y hne cell hx hy
  obtain ⟨_,i,hi,hei⟩ := footprint_tile c BK x cell hx
  obtain ⟨_,j,hj,hej⟩ := footprint_tile c BK y cell hy
  exact hne (address_injective c x y i j hi hj (hei.symm.trans hej)).1

def result (c : Shape) (BK : Nat) (hK : c.K ≤ BK) (scale : ℝ) (s : BlockState) (hu : s.undef=fun _ _ => 0) : BlockState :=
  Kernel.mergeFrames (grid c) s (frames c BK hK scale s hu)
def launched (c : Shape) (BK : Nat) (hK : c.K ≤ BK) (scale : ℝ) (s : BlockState) (hu : s.undef=fun _ _ => 0) :
    Kernel.GridLaunchedOrdinary (kernel c BK scale) (grid c) s (result c BK hK scale s hu) :=
  ⟨frames c BK hK scale s hu,disjoint c BK hK scale s hu,rfl⟩

theorem readback_causal (c : Shape) (BK : Nat) (hK : c.K ≤ BK) (scale : ℝ) (s : BlockState)
    (hu : s.undef=fun _ _ => 0) (h : Fin c.heads) (t : Fin c.T) (j : Fin c.BT)
    (hsame : j.val/16=t.val%c.BT/16) (hcausal : j.val ≤ t.val%c.BT) :
    (result c BK hK scale s hu).readMem "attention" (h.val*c.T*c.BT+t.val*c.BT+j.val)=
      Logical.attention c.T c.K c.BT (InterGrid.inputs s) scale h.val t.val j.val := by
  let q := owner c h t
  let i := lane c t j
  have hrow : rowStart (layout c BK) (launchIndex c q)+i.1.val=t.val := owner_row c h t j
  have hcol : (launchIndex c q).block*16+i.2.1.val=j.val := owner_column c h t j hsame
  have hij : i.2.1.val ≤ i.1.val := by
    have hd := Nat.div_add_mod' (t.val%c.BT) 16
    have hj := Nat.div_add_mod' j.val 16
    dsimp [i,lane]
    omega
  have ht := t.isLt
  have hen : enabled (layout c BK) (launchIndex c q) := by unfold enabled; change _<c.T; omega
  have hi : rowActive (layout c BK) (launchIndex c q) i.1.val := by unfold rowActive; rw [hrow]; exact ht
  have hjc : i.2.1.val<columns (layout c BK) (launchIndex c q) := by
    have hii := i.1.isLt
    unfold columns
    dsimp [rowStart,layout] at hrow ⊢
    omega
  have hv := (perProgram c BK hK scale s hu q).choose_spec.2.1 hen i.2.1.val hjc i.1 hi
  change (frames c BK hK scale s hu q).final.readMem _ _ = _ at hv
  have hw : (frames c BK hK scale s hu q).writes ("attention",
      RealIntraMath.outputAddress (layout c BK) (launchIndex c q) i.1.val i.2.1.val) :=
    ⟨hen,rfl,i.2.1.val,hjc,(i.1,⟨⟩),hi,rfl⟩
  have hm := Kernel.mergeFrames_mem_written (disjoint c BK hK scale s hu) q _ _ hw
  have hv' : (result c BK hK scale s hu).readMem "attention"
      (RealIntraMath.outputAddress (layout c BK) (launchIndex c q) i.1.val i.2.1.val)=
      RealIntraMath.expected (layout c BK) (launchIndex c q) (Logical.intraInputs (InterGrid.inputs s)) scale i.1.val i.2.1.val := by
    unfold result BlockState.readMem
    rw [hm]
    exact hv
  rw [Logical.intra_expected _ _ _ _ _ _ rfl rfl,if_pos hij,hrow] at hv'
  have hcolumn : rowStart (layout c BK) (launchIndex c q)+i.2.1.val=t.val/c.BT*c.BT+j.val := by
    unfold rowStart
    change t.val/c.BT*c.BT+(launchIndex c q).block*16+i.2.1.val= _
    omega
  rw [hcolumn] at hv'
  rw [source_address] at hv'
  dsimp only [q,i] at hv'
  rw [owner_address c h t j hsame] at hv'
  simpa only [owner_launch,Logical.attention,hcausal,ite_true] using hv'

#axiomsClean readback_causal
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.IntraGrid
