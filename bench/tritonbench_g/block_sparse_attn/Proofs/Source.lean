import bench.tritonbench_g.block_sparse_attn.Proofs.Memory
namespace VeriTile.Bench.Repairs.BlockSparseAttn.Source
set_option maxHeartbeats 8000000
noncomputable section

abbrev GridIndex (s : Shape) := Fin (ceil s.length s.blockM) × Fin (s.batches*s.heads) ×
  Fin s.blocks × Fin s.blockM × Fin s.blockD
def toLane (s : Shape) (i : GridIndex s) : Lane :=
  ⟨i.1,i.2.1,i.2.2.1,i.2.2.2.1,i.2.2.2.2⟩
def lanes (s : Shape) : List Lane := (Finset.univ : Finset (GridIndex s)).toList.map (toLane s)

theorem lanes_mem (s : Shape) (l : Lane) : l∈lanes s ↔ l.Valid s := by
  simp only [lanes,List.mem_map,Finset.mem_toList,Finset.mem_univ,true_and]
  constructor
  · rintro ⟨i,rfl⟩
    exact ⟨i.1.isLt,i.2.1.isLt,i.2.2.1.isLt,i.2.2.2.1.isLt,i.2.2.2.2.isLt⟩
  · rintro ⟨hm,hh,hc,hr,hd⟩
    exact ⟨(⟨l.rowProgram,hm⟩,⟨l.headProgram,hh⟩,⟨l.channelProgram,hc⟩,
      ⟨l.rowLane,hr⟩,⟨l.colLane,hd⟩),rfl⟩

theorem lanes_nodup (s : Shape) : (lanes s).Nodup := by
  apply List.Nodup.map _ (Finset.nodup_toList _)
  rintro ⟨a,b,c,d,e⟩ ⟨f,g,h,i,j⟩ he
  simp only [toLane,Lane.mk.injEq] at he
  have h1 : a=f := Fin.ext he.1
  have h2 : b=g := Fin.ext he.2.1
  have h3 : c=h := Fin.ext he.2.2.1
  have h4 : d=i := Fin.ext he.2.2.2.1
  have h5 : e=j := Fin.ext he.2.2.2.2
  simp [h1,h2,h3,h4,h5]

theorem store_mask (s : Shape) (heads : 0<s.heads) (l : Lane) (valid : l.Valid s) :
    (laneWork s l).active s ↔ l.rowProgram*s.blockM+l.rowLane<s.length := by
  have hb : l.headProgram/s.heads<s.batches :=
    (Nat.div_lt_iff_lt_mul heads).mpr valid.2.1
  have hh : l.headProgram%s.heads<s.heads := Nat.mod_lt _ heads
  have hc : l.channelProgram*s.blockD+l.colLane<s.dim := by
    have hblock:=valid.2.2.1
    have hp:=Nat.mul_le_mul_right s.blockD (show l.channelProgram+1≤s.blocks by omega)
    have hd:=valid.2.2.2.2
    unfold Shape.dim
    nlinarith
  simp only [Work.active,laneWork,hb,hh,hc,and_true,true_and]

/-- Actual 3D launch and the sole store mask. Each lane is retained once. -/
def work (s : Shape) : List Work :=
  ((lanes s).filter (fun l=>decide (l.rowProgram*s.blockM+l.rowLane<s.length))).map (laneWork s)

theorem work_complete (s : Shape) (hh : 0<s.heads) (hm : 0<s.blockM) (hd : 0<s.blockD) :
    Complete s (work s) := by
  have nd : (work s).Nodup := by
    apply List.Nodup.map_on _ ((lanes_nodup s).filter _)
    intro a ha b hb he
    exact lane_injective s a b ((lanes_mem s a).mp (List.mem_filter.mp ha).1)
      ((lanes_mem s b).mp (List.mem_filter.mp hb).1) he
  apply (List.perm_ext_iff_of_nodup nd (schedule_nodup s)).mpr
  intro w
  rw [schedule_mem]
  constructor
  · intro hw
    obtain ⟨l,hl,rfl⟩:=List.mem_map.mp hw
    obtain ⟨hl,hm⟩:=List.mem_filter.mp hl
    exact (store_mask s hh l ((lanes_mem s l).mp hl)).mpr (of_decide_eq_true hm)
  · intro active
    obtain ⟨l,valid,he⟩:=lane_coverage s hm hd w active
    apply List.mem_map.mpr
    refine ⟨l,?_,he⟩
    apply List.mem_filter.mpr
    refine ⟨(lanes_mem s l).mpr valid,?_⟩
    apply decide_eq_true
    exact (store_mask s hh l valid).mp (he.symm ▸ active)

/-- Source pointer arithmetic, retaining each independent inner stride. -/
def queryPointer (t : Strides) (b h pm lm block bd d : Nat) (bm : Nat) : Nat :=
  b*t.batch+h*t.head+(pm*bm+lm)*t.row+d*t.dim+block*bd*t.dim
def keyPointer (t : Strides) (b h : Nat) (column : Int) (bn lane block bd d : Nat) : Int :=
  (b*t.batch+h*t.head : Nat)+(lane*t.row+d*t.dim : Nat)+
    column*(bn:Int)*(t.row:Int)+(block*bd*t.dim : Nat)
def valuePointer (t : Strides) (b h : Nat) (column : Int) (bn lane block bd d : Nat) : Int :=
  (b*t.batch+h*t.head : Nat)+(lane*t.row+(block*bd+d)*t.dim : Nat)+
    column*(bn:Int)*(t.row:Int)

theorem query_pointer (t : Strides) (b h pm lm block bd d bm : Nat) :
    queryPointer t b h pm lm block bd d bm=address t b h (pm*bm+lm) (block*bd+d) := by
  unfold queryPointer address
  ring
theorem key_pointer (t : Strides) (b h : Nat) (column : Int) (bn lane block bd d : Nat) :
    keyPointer t b h column bn lane block bd d=
      signedAddress t b h (column*(bn:Int)+(lane:Int)) (block*bd+d) := by
  unfold keyPointer signedAddress
  push_cast
  ring
theorem value_pointer (t : Strides) (b h : Nat) (column : Int) (bn lane block bd d : Nat) :
    valuePointer t b h column bn lane block bd d=
      signedAddress t b h (column*(bn:Int)+(lane:Int)) (block*bd+d) := by
  unfold valuePointer signedAddress
  push_cast
  ring
theorem output_pointer (s : Shape) (l : Lane) :
    l.headProgram/s.heads*(s.heads*s.length*s.dim)+
      l.headProgram%s.heads*(s.length*s.dim)+(l.rowProgram*s.blockM+l.rowLane)*s.dim+
        l.channelProgram*s.blockD+l.colLane=(laneWork s l).target s := by
  unfold laneWork Work.target outputAddress
  ring

#axiomsClean lanes_mem
#axiomsClean lanes_nodup
#axiomsClean store_mask
#axiomsClean work_complete
#axiomsClean query_pointer
#axiomsClean key_pointer
#axiomsClean value_pointer
#axiomsClean output_pointer
end
end VeriTile.Bench.Repairs.BlockSparseAttn.Source
