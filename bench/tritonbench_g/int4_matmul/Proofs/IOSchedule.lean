import bench.tritonbench_g.int4_matmul.IO

namespace VeriTile.Bench.Repairs.Int4Matmul.IOProof
open VeriTile.Triton FP.Structural KernelExecution
noncomputable section
set_option maxHeartbeats 4000000

@[simp] theorem program_mem (s : Shape) (t : Tile) (p : IO.Program) :
    p∈IO.programs s t ↔ p.1<blocksM s t*blocksN s t ∧ p.2<t.splits := by
  rcases p with ⟨pid,split⟩
  simp [IO.programs,List.mem_product]

theorem programs_nodup (s : Shape) (t : Tile) : (IO.programs s t).Nodup := by
  exact List.Nodup.product List.nodup_range List.nodup_range

theorem lane_owns (s : Shape) (t : Tile) (ht : t.Valid) (l : Lane) (valid : l.Valid s t) :
    IO.owns s t (l.pid,l.split) (laneWork s t l).row (laneWork s t l).col := by
  simp only [IO.owns,laneWork,div_coordinate _ _ _ ht.bmpos valid.2.2.1,
    div_coordinate _ _ _ ht.bnpos valid.2.2.2,and_self]

theorem own_lane (s : Shape) (t : Tile) (ht : t.Valid) (p : IO.Program)
    (listed : p∈IO.programs s t) (row col : Nat) (own : IO.owns s t p row col) :
    ∃ l : Lane,l.Valid s t ∧ (l.pid,l.split)=p ∧
      (laneWork s t l).row=row ∧ (laneWork s t l).col=col := by
  have hp := (program_mem s t p).mp listed
  refine ⟨⟨p.1,p.2,row%t.bm,col%t.bn⟩,⟨hp.1,hp.2,Nat.mod_lt _ ht.bmpos,Nat.mod_lt _ ht.bnpos⟩,
    Prod.eta p,?_,?_⟩
  · change GroupedGrid.row _ _ _ p.1*t.bm+row%t.bm=row
    rw [←own.1,Nat.div_add_mod']
  · change GroupedGrid.col _ _ _ p.1*t.bn+col%t.bn=col
    rw [←own.2,Nat.div_add_mod']

theorem events_partition (s : Shape) (t : Tile) (row col : Nat) :
    (IO.events s t row col ++ (IO.programs s t).filter
      (fun p=>decide (p∉IO.events s t row col))).Perm (IO.programs s t) := by
  classical
  have disjoint : List.Disjoint (IO.events s t row col)
      ((IO.programs s t).filter (fun p=>decide (p∉IO.events s t row col))) := by
    intro p hp hq
    exact (of_decide_eq_true (List.mem_filter.mp hq).2) hp
  apply List.perm_ext_iff_of_nodup
    (List.nodup_append.mpr ⟨(programs_nodup s t).filter _,(programs_nodup s t).filter _,by
      intro a ha b hb eq
      subst b
      exact disjoint ha hb⟩)
    (programs_nodup s t) |>.mpr
  intro p
  simp only [IO.events,List.mem_append,List.mem_filter,decide_eq_true_eq]
  tauto

theorem order_perm (s : Shape) (hs : s.Valid) (t : Tile) (ht : t.Valid)
    (lanes : List Lane) (complete : lanes.Perm (rawLanes s t)) (row col : Nat) :
    (IO.order s t lanes row col).Perm (IO.events s t row col) := by
  have raw : (rawLanes s t).Nodup :=
    (List.nodup_map_iff (toLane_injective s t)).mpr (Finset.nodup_toList _)
  have ln : lanes.Nodup := complete.nodup_iff.mpr raw
  have ln' : (IO.order s t lanes row col).Nodup := by
    apply (List.nodup_map_iff_inj_on (ln.filter _)).mpr
    intro a ha b hb same
    have ha' := of_decide_eq_true (List.mem_filter.mp ha).2
    have hb' := of_decide_eq_true (List.mem_filter.mp hb).2
    have hp : a.pid=b.pid := congrArg Prod.fst same
    have hsp : a.split=b.split := congrArg Prod.snd same
    have hi : a.i=b.i := by
      have h := ha'.1.trans hb'.1.symm
      dsimp only [laneWork] at h
      rw [hp] at h
      omega
    have hj : a.j=b.j := by
      have h := ha'.2.trans hb'.2.symm
      dsimp only [laneWork] at h
      rw [hp] at h
      omega
    cases a;cases b;simp_all
  apply List.perm_ext_iff_of_nodup ln' ((programs_nodup s t).filter _) |>.mpr
  intro p
  constructor
  · intro h
    obtain ⟨l,hl,rfl⟩ := List.mem_map.mp h
    have valid := (raw_mem s t l).mp (complete.mem_iff.mp (List.mem_filter.mp hl).1)
    have coordinates := of_decide_eq_true (List.mem_filter.mp hl).2
    apply List.mem_filter.mpr
    refine ⟨(program_mem s t _).mpr ⟨valid.1,valid.2.1⟩,?_⟩
    apply decide_eq_true
    simpa only [coordinates.1,coordinates.2] using lane_owns s t ht l valid
  · intro h
    have h := List.mem_filter.mp h
    obtain ⟨l,valid,same,hr,hc⟩ := own_lane s t ht p h.1 row col (of_decide_eq_true h.2)
    apply List.mem_map.mpr
    exact ⟨l,List.mem_filter.mpr ⟨complete.mem_iff.mpr ((raw_mem s t l).mpr valid),by simp [hr,hc]⟩,same⟩

theorem output_unique (x : Raw.Arguments) (t : Tile) (o : Ops α) (rest : Algebra α)
    (lanes : List Lane) (v : (IO.programIO x t o rest lanes).Inputs) :
    (IO.programIO x t o rest lanes).toMemoryIO.OutputUnique v := by
  rintro ⟨a,ai⟩ ⟨b,bi⟩ _ _ same
  have az : a=0 := Fin.eq_zero a
  have bz : b=0 := Fin.eq_zero b
  subst a;subst b
  have offset := congrArg Prod.snd same
  change ai.1.val*x.shape.n+ai.2.1.val=bi.1.val*x.shape.n+bi.2.1.val at offset
  have colsA := ai.2.1.isLt
  have colsB := bi.2.1.isLt
  have eq := address_injective x.shape ⟨ai.1.val,ai.2.1.val,0⟩ ⟨bi.1.val,bi.2.1.val,0⟩ colsA colsB offset
  have ae : ai=bi := by
    apply Prod.ext
    · exact Fin.ext eq.1
    · apply Prod.ext
      · exact Fin.ext eq.2
      · exact Subsingleton.elim _ _
  subst bi
  rfl

theorem schedule_complete (x : Raw.Arguments) (geometry : Raw.Geometry x) (t : Tile)
    (member : t∈Host.sourceTiles) (o : Ops α) (rest : Algebra α)
    (lanes : List Lane) (complete : lanes.Perm (rawLanes x.shape t))
    (v : (IO.programIO x t o rest lanes).Inputs) :
    (IO.programIO x t o rest lanes).ScheduleComplete v := by
  refine ⟨rfl,output_unique x t o rest lanes v,?_⟩
  rintro ⟨key,j⟩ _
  have hz : key=0 := Fin.eq_zero key
  subst key
  constructor
  · exact order_perm x.shape geometry.shape t (Host.all_tiles_valid t member) lanes complete _ _
  · convert events_partition x.shape t j.1.val j.2.1.val using 1
    congr 1
    apply List.filter_congr
    intro p _
    apply Bool.eq_iff_iff.mpr
    simp only [decide_eq_true_eq]
    rfl

#axiomsClean schedule_complete
end
end VeriTile.Bench.Repairs.Int4Matmul.IOProof
