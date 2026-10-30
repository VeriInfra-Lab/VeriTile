import bench.tritonbench_g.chunk_gla_fwd.Proofs.ProducerGeometry
import VeriTile.Triton.Launch.Composition
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.InterGrid
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGlaFwd
open ProducerGeometry RealInterMath
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

def inputs (s : BlockState) : Logical.Inputs :=
  ⟨s.readMem "q",s.readMem "k",s.readMem "g",s.readMem "v",s.readMem "h"⟩
def kernel (c : Shape) (scale : ℝ) : Kernel :=
  (SourceBridge.interNew (layout c) regions scale).eraseDType.toAlgKernel

theorem perProgram (c : Shape) (scale : ℝ) (s : BlockState) (hu : s.undef=fun _ _ => 0)
    (q : GridIndex (grid c)) :
    ∃ t, exec (kernel c scale) (s.withGridIndex q)=some t ∧
      (enabled (layout c) (launchIndex c q) → ∀ i : TileIndex [16,16],
        active (layout c) (launchIndex c q) i.1.val i.2.1.val →
        t.readMem "attention" (outputAddress (layout c) (launchIndex c q) i.1.val i.2.1.val)=
          Logical.contraction c.T c.K (inputs s) scale (launchIndex c q).head
            (rowStart (layout c) (launchIndex c q)+i.1.val) (columnStart (layout c) (launchIndex c q)+i.2.1.val)) ∧
      BlockState.WriteWithin (RealInterFinal.footprint (layout c) (launchIndex c q) "attention") (s.withGridIndex q) t := by
  obtain ⟨t,he,ho,hf⟩ := LocalReal.inter_executes (layout c) regions scale (s.withGridIndex q) (by simpa using hu)
  refine ⟨t,he,?_,?_⟩
  · intro hen i hi
    have hc : columnStart (layout c) (launchIndex c q)+i.2.1.val<c.T := by
      have hij := hen.2
      have hr := hi.1
      have hj := i.2.1.isLt
      dsimp [columnStart,rowStart,rowBlock,columnBlock,layout] at hij hr ⊢
      omega
    have h := ho (by simpa [launch_grid] using hen) i (by simpa [launch_grid] using hi)
    rw [launch_grid] at h
    change t.readMem "attention" _=expected (layout c) (launchIndex c q) (Logical.interInputs (inputs s)) scale _ _ at h
    rw [Logical.inter_expected _ _ _ _ _ _ rfl rfl c.bk_pos hi.1 hc] at h
    exact h
  · simpa only [launch_grid,regions] using hf

def frames (c : Shape) (scale : ℝ) (s : BlockState) (hu : s.undef=fun _ _ => 0) :
    Kernel.GridFrames (kernel c scale) (grid c) s := fun q => {
  final := (perProgram c scale s hu q).choose
  writes := RealInterFinal.footprint (layout c) (launchIndex c q) "attention"
  h_exec := (perProgram c scale s hu q).choose_spec.1
  h_writeWithin := (perProgram c scale s hu q).choose_spec.2.2 }

theorem disjoint (c : Shape) (scale : ℝ) (s : BlockState) (hu : s.undef=fun _ _ => 0) :
    Kernel.GridWritesDisjoint (frames c scale s hu) := by
  intro x y hne cell hx hy
  rcases hx with ⟨_,_,i,hi,hei⟩
  rcases hy with ⟨_,_,j,hj,hej⟩
  exact hne (ProducerGeometry.address_injective c x y i j hi hj (hei.symm.trans hej)).1

def result (c : Shape) (scale : ℝ) (s : BlockState) (hu : s.undef=fun _ _ => 0) : BlockState :=
  Kernel.mergeFrames (grid c) s (frames c scale s hu)

def launched (c : Shape) (scale : ℝ) (s : BlockState) (hu : s.undef=fun _ _ => 0) :
    Kernel.GridLaunchedOrdinary (kernel c scale) (grid c) s (result c scale s hu) :=
  ⟨frames c scale s hu,disjoint c scale s hu,rfl⟩

theorem owner_enabled (c : Shape) (h : Fin c.heads) (t : Fin c.T) (j : Fin c.BT)
    (hlower : j.val/16<t.val%c.BT/16) : enabled (layout c) (launchIndex c (owner c h t j)) := by
  constructor
  · have hr := owner_row c h t j
    have ht := t.isLt
    change _<c.T
    omega
  · have hj : j.val/16<c.NC := quotient_bound (by simpa [c.bt_eq] using j.isLt) (by decide)
    simp only [owner_launch,rowBlock,columnBlock,layout,Nat.add_mod,Nat.mul_mod_left,zero_add,Nat.mod_eq_of_lt hj]
    have hp : (t.val%c.BT/16*c.NC+j.val/16)/c.NC=t.val%c.BT/16 := by
      rw [Nat.mul_comm _ c.NC,Nat.mul_add_div c.nc_pos,Nat.div_eq_of_lt hj,Nat.add_zero]
    rw [hp]
    exact hlower

theorem readback (c : Shape) (scale : ℝ) (s : BlockState) (hu : s.undef=fun _ _ => 0)
    (h : Fin c.heads) (t : Fin c.T) (j : Fin c.BT) (hlower : j.val/16<t.val%c.BT/16) :
    (result c scale s hu).readMem "attention" (h.val*c.T*c.BT+t.val*c.BT+j.val)=
      Logical.contraction c.T c.K (inputs s) scale h.val t.val (t.val/c.BT*c.BT+j.val) := by
  let q := owner c h t j
  let i := lane c t j
  have hen := owner_enabled c h t j hlower
  have hi := owner_active c h t j
  have hw : (frames c scale s hu q).writes ("attention",outputAddress (layout c) (launchIndex c q) i.1.val i.2.1.val) :=
    ⟨rfl,hen,i,hi,rfl⟩
  have hm := Kernel.mergeFrames_mem_written (disjoint c scale s hu) q _ _ hw
  have hv := (perProgram c scale s hu q).choose_spec.2.1 hen i hi
  change (frames c scale s hu q).final.readMem _ _ = _ at hv
  have hc : columnStart (layout c) (launchIndex c q)+i.2.1.val=t.val/c.BT*c.BT+j.val := by
    unfold columnStart
    rw [Nat.add_assoc]
    change t.val/c.BT*c.BT+_= _
    exact congrArg (fun x => t.val/c.BT*c.BT+x) (owner_column c h t j)
  have hr : (result c scale s hu).readMem "attention"
      (outputAddress (layout c) (launchIndex c q) i.1.val i.2.1.val)=
      Logical.contraction c.T c.K (inputs s) scale (launchIndex c q).head
        (rowStart (layout c) (launchIndex c q)+i.1.val) (columnStart (layout c) (launchIndex c q)+i.2.1.val) := by
    unfold result BlockState.readMem
    rw [hm]
    exact hv
  rw [hc] at hr
  dsimp only [q,i] at hr
  rw [owner_address,owner_row] at hr
  simpa only [owner_launch] using hr

theorem unwritten (c : Shape) (scale : ℝ) (s : BlockState) (hu : s.undef=fun _ _ => 0)
    (r : RegionName) (off : Nat)
    (hn : ∀ q, ¬RealInterFinal.footprint (layout c) (launchIndex c q) "attention" (r,off)) :
    (result c scale s hu).mem r off=s.mem r off := by
  apply Kernel.mergeFrames_mem_eq_of_not_written
  rintro ⟨q,hq⟩
  exact hn q hq

#axiomsClean readback
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.InterGrid
