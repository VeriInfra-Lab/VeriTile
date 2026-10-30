import bench.tritonbench_g.int8_dequant_matmul.IO
import VeriTile.Triton.Float.PerAddressProgramIO

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.IOGeometry
open VeriTile Triton FP.Structural IOPrograms
open IntScaledMatmul.Revision001.Geometry
noncomputable section
set_option maxHeartbeats 2000000

theorem programs_nodup (s : Shape) (c : Config) : (programs s c).Nodup :=
  (List.nodup_finRange _).product (List.nodup_finRange _)

theorem program_mem (s : Shape) (c : Config) (p : Program s c) : p ∈ programs s c := by
  exact List.mem_product.mpr ⟨by simp,by simp⟩

theorem rows_pos (s : Shape) (c : Config) (pid : Nat) (valid : pid<count (geometry s c)) :
    0<s.M := by
  by_contra empty
  have zero : s.M=0 := by omega
  have blocks : gridRows (geometry s c)=0 := by
    change (s.M+c.BM-1)/c.BM=0
    rw [zero,Nat.zero_add,Nat.div_eq_of_lt (by have h:=c.positiveM; omega)]
  simp only [count,blocks,Nat.zero_mul] at valid
  omega

theorem columns_pos (s : Shape) (c : Config) (pid : Nat) (valid : pid<count (geometry s c)) :
    0<s.N := by
  by_contra empty
  have zero : s.N=0 := by omega
  have blocks : gridColumns (geometry s c)=0 := by
    change (s.N+c.BN-1)/c.BN=0
    rw [zero,Nat.zero_add,Nat.div_eq_of_lt (by have h:=c.positiveN; omega)]
  simp only [count,blocks,Nat.mul_zero] at valid
  omega

theorem owner_owns (s : Shape) (c : Config) (cell : MatrixCell s) :
    Owns (geometry s c) (owner s c cell).val cell :=
  (Classical.choose_spec (Grouped.every_cell s c cell)).2

theorem owner_unique (s : Shape) (c : Config) (cell : MatrixCell s)
    (pid : Fin (count (geometry s c))) (owns : Owns (geometry s c) pid.val cell) :
    pid = owner s c cell :=
  Fin.ext (Grouped.only_one_owner s c pid.val (owner s c cell).val pid.isLt
    (owner s c cell).isLt cell owns (owner_owns s c cell))

theorem physical_owner (s : Shape) (c : Config) (p : PhysicalCommit s c) :
    p.pid = owner s c p.logical.1 := by
  apply owner_unique
  exact ⟨p.lane,rfl,rfl⟩

theorem physical_program (s : Shape) (c : Config) (p : PhysicalCommit s c) :
    physicalProgram p = (owner s c p.logical.1,p.logical.2) := by
  simp only [physicalProgram,physical_owner s c p]
  rfl

theorem events_mem (s : Shape) (c : Config) (cell : MatrixCell s) (p : Program s c) :
    p ∈ events s c cell ↔ p.1 = owner s c cell := by
  rcases p with ⟨pid,split⟩
  simp [events,eq_comm]

theorem events_nodup (s : Shape) (c : Config) (cell : MatrixCell s) :
    (events s c cell).Nodup :=
  (List.nodup_finRange _).map (fun _ _ same => congrArg Prod.snd same)

theorem events_partition (s : Shape) (c : Config) (cell : MatrixCell s) :
    (events s c cell ++ (programs s c).filter (fun p => decide (p ∉ events s c cell))).Perm
      (programs s c) := by
  classical
  have same : (events s c cell).Perm
      ((programs s c).filter (fun p => decide (p ∈ events s c cell))) := by
    apply (List.perm_ext_iff_of_nodup (events_nodup s c cell) ((programs_nodup s c).filter _)).mpr
    intro p
    simp [program_mem]
  have partition := List.filter_append_perm (fun p : Program s c => decide (p ∈ events s c cell))
    (programs s c)
  simpa only [decide_not] using (same.append_right _).trans partition

theorem order_complete (s : Shape) (c : Config) (schedule : Schedule s c) (cell : MatrixCell s) :
    (order s c schedule cell).Perm (events s c cell) :=
  (schedule.complete cell).map _

theorem output_unique (F H : Type) (s : Shape) (c : Config) (hasBias : Bool)
    (vals : (IOLayout.memory F H s c hasBias).Inputs) :
    (IOLayout.memory F H s c hasBias).OutputUnique vals := by
  intro left right _ _ same
  rcases left with ⟨o,left⟩
  rcases right with ⟨p,right⟩
  have ho : o=0 := Fin.eq_zero o
  have hp : p=0 := Fin.eq_zero p
  subst o
  subst p
  obtain ⟨r,n,u⟩ := left
  obtain ⟨r',n',u'⟩ := right
  cases u
  cases u'
  have address := congrArg Prod.snd same
  change TileShape.linearIndex [s.M,s.N] (r,n,PUnit.unit) =
    TileShape.linearIndex [s.M,s.N] (r',n',PUnit.unit) at address
  have pair : (r,n)=(r',n') := by
    apply outputAddress_injective s (Buffers.mk 0 0 0 0 0 0 0 0 0 0 0 0)
    apply Prod.ext
    · rfl
    · simpa [outputAddress,TileShape.linearIndex] using address
  cases pair
  rfl

#axiomsClean physical_program
#axiomsClean events_partition
#axiomsClean order_complete
#axiomsClean output_unique
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.IOGeometry
