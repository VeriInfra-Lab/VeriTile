import bench.tritonbench_g.triton_matmul.Proofs.Model
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false
abbrev groupBase := MatmulTritonAutotune.Whole.groupBase
abbrev groupSize := MatmulTritonAutotune.Whole.groupSize
abbrev tileCol := MatmulTritonAutotune.Whole.tileCol

/-- The original source rotates rows inside each group by absolute pid % size.
The final short group remains bijective; its schedule is preserved. -/
def tileRow (rows cols gm pid : Nat) := groupBase rows cols gm pid + pid%groupSize rows cols gm pid

theorem grouped_bounds (rows cols gm pid : Nat) (hgm : 0<gm) (hp : pid<rows*cols) :
    0<groupSize rows cols gm pid ∧ tileRow rows cols gm pid<rows ∧ tileCol rows cols gm pid<cols := by
  obtain ⟨hs,_,hc⟩ := MatmulTritonAutotune.Whole.grouped_bounds rows cols gm pid hgm hp
  change 0<groupSize rows cols gm pid at hs
  have hcols : 0<cols := by nlinarith
  have hb : groupBase rows cols gm pid<rows := by
    have h := Nat.div_mul_le_self pid (gm*cols)
    unfold groupBase MatmulTritonAutotune.Whole.groupBase
    nlinarith
  have hm := Nat.mod_lt pid hs
  have hle : groupSize rows cols gm pid≤rows-groupBase rows cols gm pid := min_le_left _ _
  exact ⟨hs,by unfold tileRow; omega,hc⟩

theorem grouped_injective (rows cols gm a b : Nat)
    (hgm : 0<gm) (ha : a<rows*cols) (hb : b<rows*cols)
    (hr : tileRow rows cols gm a=tileRow rows cols gm b)
    (hc : tileCol rows cols gm a=tileCol rows cols gm b) : a=b := by
  have group_of_row (pid : Nat) (hp : pid<rows*cols) :
      tileRow rows cols gm pid/gm=pid/(gm*cols) := by
    have hs := (grouped_bounds rows cols gm pid hgm hp).1
    have hm := Nat.mod_lt pid hs
    have hle : groupSize rows cols gm pid≤gm := min_le_right _ _
    unfold tileRow groupBase MatmulTritonAutotune.Whole.groupBase
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ hgm]
    rw [Nat.div_eq_of_lt (show pid%groupSize rows cols gm pid<gm by omega)]
    simp
  have hq : a/(gm*cols)=b/(gm*cols) := by rw [←group_of_row a ha,←group_of_row b hb,hr]
  have hbase : groupBase rows cols gm a=groupBase rows cols gm b := by simp [groupBase,MatmulTritonAutotune.Whole.groupBase,hq]
  have hsize : groupSize rows cols gm a=groupSize rows cols gm b := by simp [groupSize,MatmulTritonAutotune.Whole.groupSize,hbase]
  have hm : a%groupSize rows cols gm a=b%groupSize rows cols gm a := by
    unfold tileRow at hr
    rw [←hbase,←hsize] at hr
    exact Nat.add_left_cancel hr
  have hmod : a%(gm*cols)%groupSize rows cols gm a=b%(gm*cols)%groupSize rows cols gm a := by
    apply Nat.ModEq.add_left_cancel' ((gm*cols)*(a/(gm*cols)))
    change ((gm*cols)*(a/(gm*cols))+a%(gm*cols))%_=((gm*cols)*(a/(gm*cols))+b%(gm*cols))%_
    rw [Nat.div_add_mod, hq, Nat.div_add_mod]
    exact hm
  unfold tileCol MatmulTritonAutotune.Whole.tileCol at hc
  change a%(gm*cols)/groupSize rows cols gm a=b%(gm*cols)/groupSize rows cols gm b at hc
  rw [←hsize] at hc
  exact Nat.ext_div_mod hq (Nat.ext_div_mod hc hmod)

def coordinate (rows cols gm : Nat) (hgm : 0<gm) (p : Fin (rows*cols)) : Fin rows × Fin cols :=
  (⟨tileRow rows cols gm p.val,(grouped_bounds rows cols gm p.val hgm p.isLt).2.1⟩,
   ⟨tileCol rows cols gm p.val,(grouped_bounds rows cols gm p.val hgm p.isLt).2.2⟩)

theorem coordinate_bijective (rows cols gm : Nat) (hgm : 0<gm) : Function.Bijective (coordinate rows cols gm hgm) := by
  apply (Fintype.bijective_iff_injective_and_card _).mpr
  refine ⟨?_,by simp⟩
  intro a b he
  apply Fin.ext
  exact grouped_injective rows cols gm a.val b.val hgm a.isLt b.isLt
    (congrArg (fun x => x.1.val) he) (congrArg (fun x => x.2.val) he)

theorem grouped_covered (rows cols gm i j : Nat) (hgm : 0<gm) (hi : i<rows) (hj : j<cols) :
    ∃ pid, pid<rows*cols ∧ tileRow rows cols gm pid=i ∧ tileCol rows cols gm pid=j := by
  obtain ⟨p,hp⟩ := (coordinate_bijective rows cols gm hgm).2 (⟨i,hi⟩,⟨j,hj⟩)
  exact ⟨p.val,p.isLt,congrArg (fun x => x.1.val) hp,congrArg (fun x => x.2.val) hp⟩

@[simp] theorem source_pidM {α : Type} (r : Request) (pid : Nat) (s : State α) :
    FPValue.pidM r.parameters (instanceState r pid s)=tileRow (tileRows r) (tileColumns r) 8 pid := by
  simp [FPValue.pidM,FPValue.firstM,FPValue.perGroup,FPValue.blocksN,FPValue.blocksM,
    FPValue.groupSize,FPValue.BM,FPValue.BN,Request.parameters,SourceScope.parameters,instanceState,
    tileRow,groupBase,groupSize,tileRows,tileColumns,MatmulTritonAutotune.Whole.groupBase,MatmulTritonAutotune.Whole.groupSize]

@[simp] theorem source_pidN {α : Type} (r : Request) (pid : Nat) (s : State α) :
    FPValue.pidN r.parameters (instanceState r pid s)=tileCol (tileRows r) (tileColumns r) 8 pid := by
  simp [FPValue.pidN,FPValue.firstM,FPValue.perGroup,FPValue.blocksN,FPValue.blocksM,
    FPValue.groupSize,FPValue.BM,FPValue.BN,Request.parameters,SourceScope.parameters,instanceState,
    tileCol,groupBase,groupSize,tileRows,tileColumns,MatmulTritonAutotune.Whole.tileCol,MatmulTritonAutotune.Whole.groupBase,MatmulTritonAutotune.Whole.groupSize]

theorem offset_injective (r : Request) {i j i' j' : Nat}
    (hj : j<r.columns) (hj' : j'<r.columns)
    (h : outputOffset r i j=outputOffset r i' j') : i=i' ∧ j=j' := by
  have hn : 0<max r.columns 1 := by omega
  have hjn : j<max r.columns 1 := lt_of_lt_of_le hj (le_max_left _ _)
  have hjn' : j'<max r.columns 1 := lt_of_lt_of_le hj' (le_max_left _ _)
  unfold outputOffset at h
  have hm := congrArg (fun n => n%max r.columns 1) h
  have hd := congrArg (fun n => n/max r.columns 1) h
  simp [Nat.add_mod,Nat.mod_eq_of_lt hjn,Nat.mod_eq_of_lt hjn'] at hm
  subst j'
  constructor
  · exact Nat.eq_of_mul_eq_mul_left hn (Nat.add_right_cancel h)
  · rfl

theorem active_injective {α : Type} (r : Request) (s : State α)
    (i j : FPValue.CIndex r.parameters)
    (hi : FPValue.active r.parameters s i) (hj : FPValue.active r.parameters s j)
    (he : FPValue.cOffset r.parameters s i=FPValue.cOffset r.parameters s j) : i=j :=
  SourceScope.active_injective r.A r.B r.C r.storage r.rows r.columns r.inner
    r.strideAM r.strideAK r.strideBK r.strideBN s i j hi hj he

theorem cell_covered {α : Type} (r : Request) (s : State α)
    (row col : Nat) (hr : row<r.rows) (hc : col<r.columns) :
    ∃ pid, pid<programs r ∧ ∃ i : FPValue.CIndex r.parameters,
      FPValue.row r.parameters (instanceState r pid s) i.1=row ∧
      FPValue.col r.parameters (instanceState r pid s) i.2.1=col := by
  have hrb : row/128<tileRows r := by simp [tileRows]; omega
  have hcb : col/256<tileColumns r := by simp [tileColumns]; omega
  obtain ⟨pid,hp,hm,hn⟩ := grouped_covered (tileRows r) (tileColumns r) 8 (row/128) (col/256) (by decide) hrb hcb
  refine ⟨pid,hp,(⟨row%128,Nat.mod_lt _ (by decide)⟩,
    ⟨col%256,Nat.mod_lt _ (by decide)⟩,PUnit.unit),?_,?_⟩
  · change FPValue.pidM r.parameters (instanceState r pid s)*128+row%128=row
    rw [source_pidM,hm]
    simpa [Nat.mul_comm] using Nat.div_add_mod row 128
  · change FPValue.pidN r.parameters (instanceState r pid s)*256+col%256=col
    rw [source_pidN,hn]
    simpa [Nat.mul_comm] using Nat.div_add_mod col 256


#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair
