import bench.tritonbench_g.swiglu_triton.Proofs.Physical

namespace VeriTile.Bench.Repairs.SwigluTriton.Index
open Memory Physical
noncomputable section
set_option maxHeartbeats 4000000

structure Chunk where
  first : Nat
  count : Nat

def chunks (first remaining : Nat) : List Chunk :=
  if empty : remaining=0 then []
  else
    let count := min remaining 65535
    ⟨first,count⟩ :: chunks (first+count) (remaining-count)
termination_by remaining
decreasing_by
  have positive : 0<min remaining 65535 := by omega
  exact Nat.sub_lt (by omega) positive

def Chunk.rows (chunk : Chunk) : List Nat := List.range' chunk.first chunk.count

theorem chunks_cover (first remaining : Nat) :
    (chunks first remaining).flatMap Chunk.rows=List.range' first remaining := by
  induction remaining using Nat.strong_induction_on generalizing first with
  | h remaining ih =>
    by_cases empty : remaining=0
    · subst remaining; simp [chunks]
    · have smaller : remaining-min remaining 65535<remaining := Nat.sub_lt (by omega) (by omega)
      rw [chunks]
      simp only [dif_neg empty,List.flatMap_cons,Chunk.rows]
      rw [ih _ smaller]
      rw [List.range'_append_1]
      congr 1
      omega

theorem chunks_bounds (first remaining : Nat) (chunk : Chunk) (member : chunk∈chunks first remaining) :
    first≤chunk.first ∧ 0<chunk.count ∧ chunk.count≤65535 ∧ chunk.first+chunk.count≤first+remaining := by
  induction remaining using Nat.strong_induction_on generalizing first with
  | h remaining ih =>
    by_cases empty : remaining=0
    · subst remaining; simp [chunks] at member
    · rw [chunks] at member
      simp only [dif_neg empty,List.mem_cons] at member
      rcases member with head | tail
      · subst chunk; simp only; omega
      · have smaller : remaining-min remaining 65535<remaining := Nat.sub_lt (by omega) (by omega)
        obtain ⟨lower,positive,maximum,upper⟩ := ih _ smaller _ tail
        exact ⟨by omega,positive,maximum,by omega⟩

theorem schedule (rows : Nat) :
    (chunks 0 rows).flatMap Chunk.rows=List.range rows := by
  rw [chunks_cover]
  exact List.range_eq_range'.symm

def signedRow (pid stride : Nat) : BitVec 64 :=
  BitVec.ofNat 64 pid * BitVec.ofNat 64 stride

def pointer (base first block pid lane : Nat) : BitVec 64 :=
  BitVec.ofNat 64 (base+4*(first*block)) +
    signedRow pid block * BitVec.ofNat 64 4 + BitVec.ofNat 64 (4*lane)

theorem signed_ofNat (width n : Nat) (positive : 0<width) (bound : n<2^(width-1)) :
    (BitVec.ofNat width n).toInt=(n:Int) := by
  have huge : n<2^width := lt_of_lt_of_le bound (Nat.pow_le_pow_right (by omega) (by omega))
  have exact : (BitVec.ofNat width n).toNat=n := by
    rw [BitVec.toNat_ofNat,Nat.mod_eq_of_lt huge]
  have power : 2^width=2^(width-1)*2 := by
    conv_lhs => rw [show width=(width-1)+1 by omega]
    rw [Nat.pow_succ]
  rw [BitVec.toInt_eq_toNat_of_lt (by rw [exact,power]; omega),exact]

/-- All row intermediates are i64 after the source's explicit pid cast.
The i32 ABI stride/columns and i32 range lanes are bounded by 65536. -/
theorem row_exact (block pid : Nat) (blockBound : block≤65536) (pidBound : pid<65535) :
    (BitVec.ofNat 32 block).toInt=(block:Int) ∧
    (BitVec.ofNat 32 pid).toInt=(pid:Int) ∧
    (signedRow pid block).toInt=(pid*block:Nat) := by
  have product := Nat.mul_le_mul (show pid≤65535 by omega) blockBound
  refine ⟨signed_ofNat 32 block (by decide) (by omega),
    signed_ofNat 32 pid (by decide) (by omega),?_⟩
  rw [signedRow,←BitVec.ofNat_mul]
  exact signed_ofNat 64 _ (by decide) (by omega)

theorem mask_exact (block cols lane : Nat) (colsBound : cols≤block)
    (blockBound : block≤65536) (laneBound : lane<block) :
    (BitVec.ofNat 32 lane).slt (BitVec.ofNat 32 cols)=decide (lane<cols) := by
  change decide ((BitVec.ofNat 32 lane).toInt<(BitVec.ofNat 32 cols).toInt)=_
  rw [signed_ofNat 32 lane (by decide) (by omega),signed_ofNat 32 cols (by decide) (by omega)]
  simp only [Int.ofNat_lt]

/-- The complete padded allocation bounds even inactive tail pointer
formation. The base is an unsigned 64-bit address: no spurious <2^63
restriction is imposed on an allocator-provided backing. -/
theorem pointer_exact (a : Allocation) (rows block : Nat) (capacity : rows*block≤a.words)
    (chunk : Chunk) (endBound : chunk.first+chunk.count≤rows)
    (pid lane : Nat) (pidBound : pid<chunk.count) (laneBound : lane<block) :
    (pointer a.baseByte chunk.first block pid lane).toNat=
      a.baseByte+4*((chunk.first+pid)*block+lane) ∧
    ∀ byte,byte<4 →
      a.baseByte≤(pointer a.baseByte chunk.first block pid lane).toNat+byte ∧
      (pointer a.baseByte chunk.first block pid lane).toNat+byte<a.baseByte+a.bytes ∧
      (pointer a.baseByte chunk.first block pid lane).toNat+byte<2^64 := by
  have index := index_bound rows block (chunk.first+pid) lane (by omega) laneBound
  have representable := a.representable
  have byteCapacity : 4*a.words≤a.bytes := by unfold Allocation.words; omega
  have arithmetic : a.baseByte+4*(chunk.first*block)+pid*block*4+4*lane=
      a.baseByte+4*((chunk.first+pid)*block+lane) := by ring
  have exact : (pointer a.baseByte chunk.first block pid lane).toNat=
      a.baseByte+4*((chunk.first+pid)*block+lane) := by
    unfold pointer signedRow
    rw [←BitVec.ofNat_mul,←BitVec.ofNat_mul,←BitVec.ofNat_add,←BitVec.ofNat_add,
      BitVec.toNat_ofNat,arithmetic,Nat.mod_eq_of_lt (by omega)]
  refine ⟨exact,?_⟩
  intro byte small
  rw [exact]
  omega

#axiomsClean chunks_cover
#axiomsClean chunks_bounds
#axiomsClean row_exact
#axiomsClean mask_exact
#axiomsClean pointer_exact
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SwigluTriton.Index
