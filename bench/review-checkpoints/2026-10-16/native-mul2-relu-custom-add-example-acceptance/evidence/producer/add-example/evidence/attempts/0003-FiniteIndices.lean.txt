import bench.tritonbench_g.vector_addition.Proofs.FiniteRoutes
import bench.tritonbench_g.add_example.Kernels

namespace VeriTile.Bench.TritonBenchG.AddExample.Finite
open VeriTile.Triton VeriTile.Bench.TritonBenchG.VectorAddition
open DTypeScope TypedHost VectorAddition.Finite
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def gridCount (n : Nat) : Nat := (n+3)/4

def start32 (pid : Nat) : BitVec 32 := BitVec.ofNat 32 pid * 4

def offset32 (pid lane : Nat) : BitVec 32 := start32 pid + BitVec.ofNat 32 lane

/-- The original annotated n_elements remains runtime i32, including n=1.
The LP64 native launcher casts PyLong_AsLong's result to int32_t. -/
def extent32 (n : Nat) : BitVec 32 := BitVec.ofNat 32 n

def mask32 (n pid lane : Nat) : Bool := (offset32 pid lane).slt (extent32 n)

def pointer64 (pointer : BitVec 64) (d : DType) (pid lane : Nat) : BitVec 64 :=
  pointer + (offset32 pid lane).signExtend 64 * BitVec.ofNat 64 (bytes d)

def programLanes (n pid : Nat) : List Nat := (List.range 4).filter (mask32 n pid)

def allLanes (n : Nat) : List (Nat × Nat) :=
  (List.range (gridCount n)).flatMap fun pid => (programLanes n pid).map (pid,·)

theorem offset32_word (pid lane : Nat) : offset32 pid lane=BitVec.ofNat 32 (pid*4+lane) := by
  simp only [offset32,start32,show (4 : BitVec 32)=BitVec.ofNat 32 4 from rfl,
    BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat]

theorem extent32_periodic (n k : Nat) : extent32 (n+k*2^32)=extent32 n := by
  apply BitVec.eq_of_toNat_eq
  simp [extent32,BitVec.toNat_ofNat,Nat.add_mod,Nat.mul_mod]

theorem minimum_extent_all_masked (n : Nat) (hn : extent32 n=BitVec.ofNat 32 (2^31))
    (pid lane : Nat) : mask32 n pid lane=Bool.false := by
  have bound := BitVec.le_two_mul_toInt (x:=offset32 pid lane)
  rw [mask32,hn,BitVec.slt_eq_decide]
  have hm : (BitVec.ofNat 32 (2^31)).toInt=(-2147483648 : Int) := by decide
  rw [hm]
  norm_num at bound
  simp only [decide_eq_false_iff_not]
  omega

theorem n2p31_all_masked (pid lane : Nat) : mask32 (2^31) pid lane=Bool.false :=
  minimum_extent_all_masked _ rfl pid lane

theorem minimum_extent_empty_program (n pid : Nat)
    (hn : extent32 n=BitVec.ofNat 32 (2^31)) : programLanes n pid=[] := by
  simp [programLanes,minimum_extent_all_masked n hn]

theorem minimum_extent_empty_grid (n : Nat)
    (hn : extent32 n=BitVec.ofNat 32 (2^31)) : allLanes n=[] := by
  simp [allLanes,minimum_extent_empty_program n _ hn]

theorem zero_empty_grid : allLanes 0=[] := rfl

theorem nat32_signed (n : Nat) (bound : n<2^31) : (BitVec.ofNat 32 n).toInt=(n : Int) := by
  exact BitVec.toInt_ofInt_eq_self (w:=32) (n:=(n : Int))
    (by decide) (by omega) (by norm_num; omega)

theorem no_wrap_indices (n pid lane : Nat) (hn : n<2^31)
    (hp : pid<gridCount n) (hl : lane<4) :
    (offset32 pid lane).toInt=(pid*4+lane : Nat) ∧
    mask32 n pid lane=decide (pid*4+lane<n) := by
  have hg : gridCount n≤2^29 := by unfold gridCount; omega
  have hi : pid*4+lane<2^31 := by omega
  have ho : (offset32 pid lane).toInt=(pid*4+lane : Nat) := by
    rw [offset32_word]; exact nat32_signed _ hi
  refine ⟨ho,?_⟩
  rw [mask32,BitVec.slt_eq_decide,ho,extent32,nat32_signed n hn]
  simp only [Int.ofNat_lt]

theorem active_owner (n i : Nat) (hi : i<n) :
    i/4<gridCount n ∧ i%4<4 ∧ (i/4)*4+i%4=i := by
  unfold gridCount
  omega

theorem no_wrap_coverage (n : Nat) (hn : n<2^31) (i : Nat) :
    (∃ pid lane, (pid,lane)∈allLanes n ∧ pid*4+lane=i) ↔ i<n := by
  constructor
  · rintro ⟨pid,lane,h,rfl⟩
    simp only [allLanes,List.mem_flatMap,List.mem_map] at h
    obtain ⟨p,hp,l,hl,he⟩ := h
    cases he
    have hp' := List.mem_range.mp hp
    have hl' := List.mem_range.mp (List.mem_filter.mp hl).1
    have hm := (List.mem_filter.mp hl).2
    rw [(no_wrap_indices n pid lane hn hp' hl').2] at hm
    exact of_decide_eq_true hm
  · intro hi
    obtain ⟨hp,hl,he⟩ := active_owner n i hi
    refine ⟨i/4,i%4,?_,he⟩
    simp only [allLanes,List.mem_flatMap,List.mem_map]
    refine ⟨i/4,List.mem_range.mpr hp,i%4,?_,rfl⟩
    apply List.mem_filter.mpr
    refine ⟨List.mem_range.mpr hl,?_⟩
    rw [(no_wrap_indices n _ _ hn hp hl).2,he]
    exact decide_eq_true hi

#axiomsClean minimum_extent_all_masked
#axiomsClean minimum_extent_empty_grid
#axiomsClean no_wrap_indices
#axiomsClean no_wrap_coverage
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AddExample.Finite
