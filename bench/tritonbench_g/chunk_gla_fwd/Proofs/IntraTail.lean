import bench.tritonbench_g.chunk_gla_fwd.Proofs.DiagonalCells
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.IntraTail
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGlaFwd
open ProducerGeometry (Shape bt_pos)
open DiagonalGeometry IntraGrid
open RealIntraMath (columns enabled rowStart rowActive)
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

theorem readback_streamed (c : Shape) (BK : Nat) (hK : c.K ≤ BK) (scale : ℝ) (s : BlockState)
    (hu : s.undef=fun _ _ => 0) (h : Fin c.heads) (t : Fin c.T) (j : Fin c.BT)
    (hsame : j.val/16=t.val%c.BT/16) (hc : t.val/c.BT*c.BT+j.val<c.T) :
    (result c BK hK scale s hu).readMem "attention" (h.val*c.T*c.BT+t.val*c.BT+j.val)=
      Logical.attention c.T c.K c.BT (InterGrid.inputs s) scale h.val t.val j.val := by
  let q := owner c h t
  let i := lane c t j
  have hrow : rowStart (layout c BK) (launchIndex c q)+i.1.val=t.val := owner_row c h t j
  have hcol : (launchIndex c q).block*16+i.2.1.val=j.val := owner_column c h t j hsame
  have ht := t.isLt
  have hen : enabled (layout c BK) (launchIndex c q) := by unfold enabled; change _<c.T; omega
  have hi : rowActive (layout c BK) (launchIndex c q) i.1.val := by unfold rowActive; rw [hrow]; exact ht
  have hjc : i.2.1.val<columns (layout c BK) (launchIndex c q) := by
    have hj := i.2.1.isLt
    simp only [q,owner_launch,columns,layout]
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
  rw [Logical.intra_expected _ _ _ _ _ _ rfl rfl,hrow] at hv'
  have hcolumn : rowStart (layout c BK) (launchIndex c q)+i.2.1.val=t.val/c.BT*c.BT+j.val := by
    unfold rowStart
    change t.val/c.BT*c.BT+(launchIndex c q).block*16+i.2.1.val= _
    omega
  rw [hcolumn] at hv'
  rw [source_address] at hv'
  dsimp only [q,i] at hv'
  rw [owner_address c h t j hsame] at hv'
  have hdt := Nat.div_add_mod' (t.val%c.BT) 16
  have hdj := Nat.div_add_mod' j.val 16
  have hij : (lane c t j).2.1.val≤(lane c t j).1.val ↔ j.val≤t.val%c.BT := by
    dsimp [lane]
    omega
  simpa only [owner_launch,Logical.attention,hij] using hv' 


theorem tail_unwritten (c : Shape) (BK : Nat) (hK : c.K≤BK) (scale : ℝ)
    (s : BlockState) (hu : s.undef=fun _ _ => 0) (h : Fin c.heads) (t : Fin c.T) (j : Fin c.BT)
    (hsame : j.val/16=t.val%c.BT/16) (hc : ¬t.val/c.BT*c.BT+j.val<c.T) :
    (result c BK hK scale s hu).mem "attention" (h.val*c.T*c.BT+t.val*c.BT+j.val)=
      s.mem "attention" (h.val*c.T*c.BT+t.val*c.BT+j.val) := by
  apply Kernel.mergeFrames_mem_eq_of_not_written
  rintro ⟨q,hw⟩
  rcases hw with ⟨hen,hr,z,hz,il,hil,he⟩
  have hz16 : z<16 := Nat.lt_of_lt_of_le hz (Nat.min_le_left _ _)
  let iz : TileIndex [16,16] := (il.1,⟨z,hz16⟩,⟨⟩)
  have ha := owner_active c h t j hsame
  have hae : address c (launchIndex c (owner c h t)) (lane c t j)=address c (launchIndex c q) iz := by
    rw [owner_address c h t j hsame]
    change _=RealIntraMath.outputAddress (layout c BK) (launchIndex c q) iz.1.val iz.2.1.val at he
    rw [source_address] at he
    exact he
  have hizi : active c (launchIndex c q) iz := ⟨hil,local_bound c q ⟨z,hz16⟩⟩
  obtain ⟨hqo,hii⟩ := address_injective c _ _ _ _ ha hizi hae
  have hzj : z=j.val%16 := by
    have h := congrArg (fun x : TileIndex [16,16] => x.2.1.val) hii
    exact h.symm
  have hcols : z<columns (layout c BK) (launchIndex c (owner c h t)) := by
    simpa only [hqo] using hz
  rw [owner_launch] at hcols
  dsimp [columns,layout] at hcols
  have hdj := Nat.div_add_mod' j.val 16
  exact hc (by omega)

theorem readback (c : Shape) (BK : Nat) (hK : c.K≤BK) (scale : ℝ)
    (s : BlockState) (hu : s.undef=fun _ _ => 0)
    (hzero : ∀ (h : Fin c.heads) (t : Fin c.T) (j : Fin c.BT),
      j.val/16=t.val%c.BT/16 → s.readMem "attention" (h.val*c.T*c.BT+t.val*c.BT+j.val)=0)
    (h : Fin c.heads) (t : Fin c.T) (j : Fin c.BT) (hsame : j.val/16=t.val%c.BT/16) :
    (result c BK hK scale s hu).readMem "attention" (h.val*c.T*c.BT+t.val*c.BT+j.val)=
      Logical.attention c.T c.K c.BT (InterGrid.inputs s) scale h.val t.val j.val := by
  by_cases hc : t.val/c.BT*c.BT+j.val<c.T
  · exact readback_streamed c BK hK scale s hu h t j hsame hc
  · have ht := t.isLt
    have hdt := Nat.div_add_mod' t.val c.BT
    have hnc : ¬j.val≤t.val%c.BT := by omega
    simp only [Logical.attention,hnc,ite_false]
    unfold BlockState.readMem
    rw [tail_unwritten c BK hK scale s hu h t j hsame hc]
    exact hzero h t j hsame

#axiomsClean readback_streamed
#axiomsClean readback
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.IntraTail
