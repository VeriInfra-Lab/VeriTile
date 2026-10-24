import bench.repairs.tritonbench_g.swiglu_triton.Proofs.Memory
import bench.repairs.tritonbench_g.swiglu_triton.Proofs.Index
import bench.tritonbench_g.adam_update_triton.Proofs.HostAutotune
namespace VeriTile.Bench.Repairs.AdamUpdateTriton.Index
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open VeriTile.Bench.Repairs.SwigluTriton.Index (signed_ofNat)
open VeriTile.Bench.TritonBenchG.AdamUpdateTriton.HostAutotune (Config)
noncomputable section

structure Chunk where
  first : Nat
  count : Nat

def chunks (first remaining : Nat) : List Chunk :=
  if empty : remaining=0 then []
  else
    let count := min remaining 8388480
    ⟨first,count⟩ :: chunks (first+count) (remaining-count)
termination_by remaining
decreasing_by
  have positive : 0<min remaining 8388480 := by omega
  exact Nat.sub_lt (by omega) positive

def Chunk.indices (chunk : Chunk) : List Nat := List.range' chunk.first chunk.count

theorem chunks_cover (first remaining : Nat) :
    (chunks first remaining).flatMap Chunk.indices=List.range' first remaining := by
  induction remaining using Nat.strong_induction_on generalizing first with
  | h remaining ih =>
    by_cases empty : remaining=0
    · subst remaining; simp [chunks]
    · have smaller : remaining-min remaining 8388480<remaining := Nat.sub_lt (by omega) (by omega)
      rw [chunks]
      simp only [dif_neg empty,List.flatMap_cons,Chunk.indices]
      rw [ih _ smaller]
      rw [List.range'_append_1]
      congr 1
      omega

theorem chunks_bounds (first remaining : Nat) (chunk : Chunk) (member : chunk∈chunks first remaining) :
    first≤chunk.first ∧ 0<chunk.count ∧ chunk.count≤8388480 ∧ chunk.first+chunk.count≤first+remaining := by
  induction remaining using Nat.strong_induction_on generalizing first with
  | h remaining ih =>
    by_cases empty : remaining=0
    · subst remaining; simp [chunks] at member
    · rw [chunks] at member
      simp only [dif_neg empty,List.mem_cons] at member
      rcases member with head | tail
      · subst chunk; simp only; omega
      · have smaller : remaining-min remaining 8388480<remaining := Nat.sub_lt (by omega) (by omega)
        obtain ⟨lower,positive,maximum,upper⟩ := ih _ smaller _ tail
        exact ⟨by omega,positive,maximum,by omega⟩

theorem schedule (rows : Nat) :
    (chunks 0 rows).flatMap Chunk.indices=List.range rows := by
  rw [chunks_cover]
  exact List.range_eq_range'.symm

def grid (count : Nat) (config : Config) : Nat := (count+config.block-1)/config.block

theorem grid_bound (count : Nat) (config : Config) (bounded : count≤8388480) :
    grid count config≤65535 := by
  cases config <;> simp only [grid,Config.block] <;> omega

def offset (pid block lane : Nat) : BitVec 32 :=
  BitVec.ofNat 32 pid * BitVec.ofNat 32 block + BitVec.ofNat 32 lane

theorem lane_bound (count : Nat) (config : Config) (pid lane : Nat)
    (bounded : count≤8388480) (hp : pid<grid count config) (hl : lane<config.block) :
    pid*config.block+lane<8388608 := by
  cases config <;> simp only [grid,Config.block] at * <;> omega

/-- Every i32 intermediate is exact, including inactive tail lanes. -/
theorem i32_exact (count : Nat) (config : Config) (pid lane : Nat)
    (bounded : count≤8388480) (hp : pid<grid count config) (hl : lane<config.block) :
    (BitVec.ofNat 32 pid).toInt=(pid:Int) ∧
    (BitVec.ofNat 32 config.block).toInt=(config.block:Int) ∧
    (BitVec.ofNat 32 lane).toInt=(lane:Int) ∧
    (BitVec.ofNat 32 pid*BitVec.ofNat 32 config.block).toInt=(pid*config.block:Nat) ∧
    (offset pid config.block lane).toInt=(pid*config.block+lane:Nat) := by
  have whole := lane_bound count config pid lane bounded hp hl
  have gb := grid_bound count config bounded
  have bb : config.block≤1024 := by cases config <;> decide
  refine ⟨signed_ofNat 32 pid (by decide) (by omega),
    signed_ofNat 32 config.block (by decide) (by omega),
    signed_ofNat 32 lane (by decide) (by omega),?_,?_⟩
  · rw [←BitVec.ofNat_mul]
    exact signed_ofNat 32 _ (by decide) (by omega)
  · rw [offset,←BitVec.ofNat_mul,←BitVec.ofNat_add]
    exact signed_ofNat 32 _ (by decide) (by omega)

theorem mask_exact (count : Nat) (config : Config) (pid lane : Nat)
    (bounded : count≤8388480) (hp : pid<grid count config) (hl : lane<config.block) :
    (offset pid config.block lane).slt (BitVec.ofNat 32 count)=decide (pid*config.block+lane<count) := by
  change decide ((offset pid config.block lane).toInt<(BitVec.ofNat 32 count).toInt)=_
  rw [(i32_exact count config pid lane bounded hp hl).2.2.2.2,
    signed_ofNat 32 count (by decide) (by omega)]
  simp only [Int.ofNat_lt]

/-- PTX signed-wide multiplication, followed by unsigned 64-bit pointer add.
Inactive pointers are modular values; they are never dereferenced. -/
def pointer (base first pid block lane : Nat) : BitVec 64 :=
  BitVec.ofNat 64 (base+4*first) +
    BitVec.ofInt 64 ((offset pid block lane).toInt*4)

theorem pointer_modular (base first count : Nat) (config : Config) (pid lane : Nat)
    (bounded : count≤8388480) (hp : pid<grid count config) (hl : lane<config.block) :
    (pointer base first pid config.block lane).toNat=
      (base+4*(first+pid*config.block+lane))%2^64 := by
  unfold pointer
  rw [(i32_exact count config pid lane bounded hp hl).2.2.2.2]
  have wide : BitVec.ofInt 64 (((pid*config.block+lane:Nat):Int)*4)=
      BitVec.ofNat 64 ((pid*config.block+lane)*4) := by
    simpa only [Int.natCast_mul] using
      (show BitVec.ofInt 64 (((pid*config.block+lane)*4:Nat):Int)=
        BitVec.ofNat 64 ((pid*config.block+lane)*4) from rfl)
  rw [wide,←BitVec.ofNat_add,BitVec.toNat_ofNat]
  congr 1
  ring

theorem active_pointer (allocation : Allocation) (total first count : Nat)
    (config : Config) (pid lane : Nat) (capacity : total≤allocation.words)
    (endBound : first+count≤total) (bounded : count≤8388480)
    (hp : pid<grid count config) (hl : lane<config.block) (active : pid*config.block+lane<count) :
    (pointer allocation.baseByte first pid config.block lane).toNat=
      (allocation.key (first+pid*config.block+lane)).address ∧
    allocation.contains (allocation.key (first+pid*config.block+lane)) := by
  have keyBound : first+pid*config.block+lane<allocation.words := by omega
  have contained := allocation.key_contains _ keyBound
  refine ⟨?_,contained⟩
  rw [pointer_modular _ _ _ config pid lane bounded hp hl]
  have repr := allocation.representable
  have bounds := contained.2.2
  simp only [Allocation.key] at bounds ⊢
  exact Nat.mod_eq_of_lt (by omega)

/-- The original total count remains the autotune key. Its i64 ABI fits for
any backed FP32 snapshot, independently of the bounded per-chunk i32 mask. -/
theorem total_i64 (allocation : Allocation) (total : Nat) (capacity : total≤allocation.words) :
    (BitVec.ofNat 64 total).toInt=(total:Int) := by
  have repr := allocation.representable
  unfold Allocation.words at capacity
  exact signed_ofNat 64 total (by decide) (by omega)

#axiomsClean chunks_cover
#axiomsClean chunks_bounds
#axiomsClean i32_exact
#axiomsClean mask_exact
#axiomsClean pointer_modular
#axiomsClean active_pointer
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AdamUpdateTriton.Index
