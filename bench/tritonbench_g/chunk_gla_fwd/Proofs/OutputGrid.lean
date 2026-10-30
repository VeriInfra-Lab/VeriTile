import bench.tritonbench_g.chunk_gla_fwd.Proofs.LocalReal
import bench.tritonbench_g.chunk_gla_fwd.Proofs.Geometry
import VeriTile.Triton.Launch.Composition

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.OutputGrid
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGlaFwd
open RealOutputIO
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

structure Shape where
  heads : Nat
  T : Nat
  K : Nat
  V : Nat
  BT : Nat
  BK : Nat
  BV : Nat
  bt_pos : 0 < BT
  bv_pos : 0 < BV

def layout (c : Shape) : Layout :=
  ⟨c.T*c.K,c.K,c.T*c.V,c.V,Geometry.ceilDiv c.T c.BT*c.K*c.V,c.V,
    c.T,c.K,c.V,c.BT,c.BK,c.BV⟩
def regions : Regions := ⟨"q","v","g","h","o","attention"⟩
def grid (c : Shape) : Grid := ⟨[Geometry.ceilDiv c.V c.BV,Geometry.ceilDiv c.T c.BT,c.heads]⟩
def kernel (c : Shape) (scale : ℝ) : Kernel :=
  (SourceBridge.outputNew (layout c) regions scale).eraseDType.toAlgKernel

def launchIndex (c : Shape) (q : GridIndex (grid c)) : Launch :=
  ⟨(q ⟨0,by norm_num [grid,Grid.rank]⟩).val,(q ⟨1,by norm_num [grid,Grid.rank]⟩).val,(q ⟨2,by norm_num [grid,Grid.rank]⟩).val⟩
def index (c : Shape) (v : Fin (Geometry.ceilDiv c.V c.BV))
    (t : Fin (Geometry.ceilDiv c.T c.BT)) (h : Fin c.heads) : GridIndex (grid c) := by
  intro a
  rcases a with ⟨n,hn⟩
  cases n with
  | zero => exact v
  | succ n =>
    cases n with
    | zero => exact t
    | succ n =>
      cases n with
      | zero => exact h
      | succ n => simp [grid,Grid.rank] at hn; omega

theorem launch_grid (c : Shape) (s : BlockState) (q : GridIndex (grid c)) :
    launch (s.withGridIndex q)=launchIndex c q := by
  simp [launch,launchIndex,BlockState.withGridIndex,GridIndex.toPids,grid,Grid.rank]

theorem launchIndex_injective (c : Shape) : Function.Injective (launchIndex c) := by
  intro x y he
  have hv := congrArg Launch.value he
  have ht := congrArg Launch.chunk he
  have hh := congrArg Launch.head he
  funext a
  fin_cases a
  · exact Fin.ext hv
  · exact Fin.ext ht
  · exact Fin.ext hh

theorem address_injective (c : Shape) (x y : GridIndex (grid c))
    (i j : TileIndex [c.BT,c.BV])
    (hi : active (layout c) (launchIndex c x) i)
    (hj : active (layout c) (launchIndex c y) j)
    (he : outputAddress (layout c) (launchIndex c x) i=outputAddress (layout c) (launchIndex c y) j) :
    x=y ∧ i=j := by
  have he' : Geometry.attentionAddress c.T c.V (launchIndex c x).head
      ((launchIndex c x).chunk*c.BT+i.1.val) ((launchIndex c x).value*c.BV+i.2.1.val) =
      Geometry.attentionAddress c.T c.V (launchIndex c y).head
      ((launchIndex c y).chunk*c.BT+j.1.val) ((launchIndex c y).value*c.BV+j.2.1.val) := by
    simpa only [outputAddress,layout,Geometry.attentionAddress,Nat.add_mul,Nat.mul_assoc] using he
  obtain ⟨hh,hr,hv⟩ := Geometry.attention_injective hi.1 hj.1 hi.2 hj.2 he'
  obtain ⟨ht,hir⟩ := Geometry.packed_injective i.1.isLt j.1.isLt hr
  obtain ⟨hv',hiv⟩ := Geometry.packed_injective i.2.1.isLt j.2.1.isLt hv
  refine ⟨launchIndex_injective c ?_,?_⟩
  · cases hx : launchIndex c x
    cases hy : launchIndex c y
    simp_all
  · exact Prod.ext (Fin.ext hir) (Prod.ext (Fin.ext hiv) (Subsingleton.elim _ _))

def owner (c : Shape) (h : Fin c.heads) (t : Fin c.T) (v : Fin c.V) : GridIndex (grid c) :=
  index c ⟨v.val/c.BV,Geometry.owner_lt c.bv_pos v.isLt⟩
    ⟨t.val/c.BT,Geometry.owner_lt c.bt_pos t.isLt⟩ h
def lane (c : Shape) (t : Fin c.T) (v : Fin c.V) : TileIndex [c.BT,c.BV] :=
  (⟨t.val%c.BT,Nat.mod_lt _ c.bt_pos⟩,⟨v.val%c.BV,Nat.mod_lt _ c.bv_pos⟩,⟨⟩)

@[simp] theorem owner_launch (c : Shape) (h : Fin c.heads) (t : Fin c.T) (v : Fin c.V) :
    launchIndex c (owner c h t v)=⟨v.val/c.BV,t.val/c.BT,h.val⟩ := rfl

theorem owner_active (c : Shape) (h : Fin c.heads) (t : Fin c.T) (v : Fin c.V) :
    active (layout c) (launchIndex c (owner c h t v)) (lane c t v) := by
  simpa [active,layout,lane,Nat.div_add_mod'] using And.intro t.isLt v.isLt

theorem owner_address (c : Shape) (h : Fin c.heads) (t : Fin c.T) (v : Fin c.V) :
    outputAddress (layout c) (launchIndex c (owner c h t v)) (lane c t v)=
      h.val*(c.T*c.V)+t.val*c.V+v.val := by
  simp [outputAddress,layout,lane,Nat.div_add_mod']

def returned (c : Shape) (s : BlockState) (h : Fin c.heads) (t : Fin c.T) (v : Fin c.V) : ℝ :=
  s.readMem "o" (h.val*(c.T*c.V)+t.val*c.V+v.val)

def formula (c : Shape) (a : Inputs) (scale : ℝ) (h : Fin c.heads) (t : Fin c.T) (v : Fin c.V) : ℝ :=
  RealOutputIO.formula (layout c) scale (launchIndex c (owner c h t v)) a (lane c t v)

def footprint (c : Shape) : WriteFootprint :=
  fun cell => cell.1="o" ∧ cell.2<c.heads*c.T*c.V

theorem perProgram (c : Shape) (scale : ℝ) (s : BlockState) (hu : s.undef=fun _ _ => 0)
    (q : GridIndex (grid c)) :
    ∃ t, exec (kernel c scale) (s.withGridIndex q)=some t ∧
      (∀ i, active (layout c) (launchIndex c q) i →
        t.readMem "o" (outputAddress (layout c) (launchIndex c q) i)=
          RealOutputIO.formula (layout c) scale (launchIndex c q) (LocalReal.outputInputs regions s) i) ∧
      BlockState.WriteWithin (LocalReal.outputFootprint (layout c) (launchIndex c q) "o") (s.withGridIndex q) t := by
  have h := LocalReal.output_executes (layout c) regions scale (s.withGridIndex q)
    (by simpa using hu) (by
      intro i j hi hj he
      rw [launch_grid] at hi hj he
      exact (address_injective c q q i j hi hj he).2)
  simpa only [kernel,launch_grid,regions,LocalReal.outputInputs,BlockState.withGridIndex,BlockState.readMem] using h

def frames (c : Shape) (scale : ℝ) (s : BlockState) (hu : s.undef=fun _ _ => 0) :
    Kernel.GridFrames (kernel c scale) (grid c) s := fun q => {
  final := (perProgram c scale s hu q).choose
  writes := LocalReal.outputFootprint (layout c) (launchIndex c q) "o"
  h_exec := (perProgram c scale s hu q).choose_spec.1
  h_writeWithin := (perProgram c scale s hu q).choose_spec.2.2 }

theorem disjoint (c : Shape) (scale : ℝ) (s : BlockState) (hu : s.undef=fun _ _ => 0) :
    Kernel.GridWritesDisjoint (frames c scale s hu) := by
  intro x y hne cell hx hy
  rcases hx with ⟨_,i,hi,hei⟩
  rcases hy with ⟨_,j,hj,hej⟩
  exact hne (address_injective c x y i j hi hj (hei.symm.trans hej)).1

theorem frame_bound (c : Shape) (scale : ℝ) (s : BlockState) (hu : s.undef=fun _ _ => 0) :
    ∀ cell, Kernel.GridWriteFootprint (frames c scale s hu) cell → footprint c cell := by
  rintro cell ⟨q,hr,i,hi,he⟩
  refine ⟨hr,?_⟩
  rw [he]
  have hh : (launchIndex c q).head<c.heads := (q ⟨2,by norm_num [grid,Grid.rank]⟩).isLt
  have h := Geometry.output_bound hh hi.1 hi.2
  simpa only [Geometry.outputAddress,outputAddress,layout,Nat.add_mul,Nat.mul_assoc] using h

/-- Constructed whole consumer grid, every returned cell, and the finite output
frame. The formula still takes the producer attention array as an explicit input. -/
theorem executes (c : Shape) (scale : ℝ) (s : BlockState) (hu : s.undef=fun _ _ => 0) :
    ∃ t, Nonempty (Kernel.GridLaunchedOrdinary (kernel c scale) (grid c) s t) ∧
      (∀ h row v, returned c t h row v=formula c (LocalReal.outputInputs regions s) scale h row v) ∧
      BlockState.WriteWithin (footprint c) s t := by
  let fs := frames c scale s hu
  let t := Kernel.mergeFrames (grid c) s fs
  have hd := disjoint c scale s hu
  refine ⟨t,⟨⟨fs,hd,rfl⟩⟩,?_,?_⟩
  · intro h row v
    let q := owner c h row v
    let i := lane c row v
    have hi := owner_active c h row v
    have hw : (fs q).writes ("o",outputAddress (layout c) (launchIndex c q) i) := ⟨rfl,i,hi,rfl⟩
    have hm := Kernel.mergeFrames_mem_written hd q _ _ hw
    have hv := (perProgram c scale s hu q).choose_spec.2.1 i hi
    change (fs q).final.readMem _ _ = _ at hv
    have hr : t.readMem "o" (outputAddress (layout c) (launchIndex c q) i)=
        RealOutputIO.formula (layout c) scale (launchIndex c q) (LocalReal.outputInputs regions s) i := by
      unfold BlockState.readMem
      rw [hm]
      exact hv
    simpa only [q,i,owner_address,returned,formula] using hr
  · intro region off hn
    exact Kernel.mergeFrames_writeWithin fs region off
      (fun hw => hn (frame_bound c scale s hu _ hw))

#specNonCircular formula avoiding [exec]
#axiomsClean executes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.OutputGrid
