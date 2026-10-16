import bench.tritonbench_g.matmul_persistent_triton.Proofs.HintProfiles

/-! The actual source offset-vector cache is initialized with arange, refreshed
only at ki=0, and retained otherwise. The proof relates that recurrence to the
already derived reached coordinate expressions. This is still a handwritten
source/control projection, not a certified TTIR interpreter. -/
namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.HintCached
open NativeDomain NativeBits HintMath HintReached NativeConfig HintProfiles
set_option maxHeartbeats 4000000

structure Vectors where
  row : Nat → BitVec 32
  column : Nat → BitVec 32

def run (p : Parameters) (pid : Nat) : Nat → Vectors
  | 0 => ⟨fun lane => bits lane,fun lane => bits lane⟩
  | i+1 =>
    if (NativeLoop.run p pid (i+1)).ki == bits 0 then
      ⟨rowBits p pid i,columnBits p pid i⟩
    else run p pid i

theorem first_restart (p : Parameters) (h : Admissible p) (pid : Nat)
    (hp : pid < grid p) (hk : 0 < count p) (hi : 0 < trip p pid) :
    ((NativeLoop.run p pid 1).ki == bits 0) = Bool.true := by
  have hc := (NativeCoordinates.control_after p h pid hp hk 0 hi).2
  have he := eq_exact hc (bits_exact (z := 0) (by norm_num [Fits]))
  simpa using he

/-- This proof also covers inactive M/N lanes; the source vectors themselves
are cached before the K-only load masks are applied. -/
theorem cache_matches (p : Parameters) (h : Admissible p) (pid : Nat)
    (hp : pid < grid p) (hk : 0 < count p) (i : Nat) (hi : i < trip p pid) :
    (run p pid (i+1)).row = rowBits p pid i ∧
    (run p pid (i+1)).column = columnBits p pid i := by
  induction i with
  | zero =>
    simp only [run,first_restart p h pid hp hk hi,if_true,and_self]
  | succ i ih =>
    have hprev := ih (by omega)
    by_cases hr : ((NativeLoop.run p pid (i+1+1)).ki == bits 0) = Bool.true
    · simp only [run,hr,if_true,and_self]
    · have hr' : ((NativeLoop.run p pid (i+1+1)).ki == bits 0) = Bool.false :=
        Bool.eq_false_iff.mpr hr
      have hc : NativeCoordinates.run p pid (i+1+1) = NativeCoordinates.run p pid (i+1) := by
        change (if (NativeLoop.run p pid (i+1+1)).ki == bits 0 then _ else _) = _
        rw [hr']
        rfl
      have hv : run p pid (i+1+1) = run p pid (i+1) := by
        rw [run,hr']
        rfl
      rw [hv]
      constructor
      · rw [hprev.1]
        funext lane
        simp only [rowBits,hc]
      · rw [hprev.2]
        funext lane
        simp only [columnBits,hc]

def rowValues (p : Parameters) (pid i lane : Nat) : Int := (run p pid (i+1)).row lane |>.toInt
def columnValues (p : Parameters) (pid i lane : Nat) : Int := (run p pid (i+1)).column lane |>.toInt

theorem values_match (p : Parameters) (h : Admissible p) (pid : Nat)
    (hp : pid < grid p) (hk : 0 < count p) (i : Nat) (hi : i < trip p pid) :
    rowValues p pid i = HintReached.rowValues p pid i ∧
    columnValues p pid i = HintReached.columnValues p pid i := by
  have hc := cache_matches p h pid hp hk i hi
  constructor
  · funext lane
    simp only [rowValues,HintReached.rowValues,hc.1]
  · funext lane
    simp only [columnValues,HintReached.columnValues,hc.2]

theorem reached_promises (p : Parameters) (h : Admissible p)
    (hm : p.BM ∣ p.M) (hn : p.BN ∣ p.N) (pid : Nat)
    (hp : pid < grid p) (hk : 0 < count p) (i : Nat) (hi : i < trip p pid) :
    TilePromises p.BM (rowValues p pid i) ∧ TilePromises p.BN (columnValues p pid i) := by
  have hc := values_match p h pid hp hk i hi
  rw [hc.1,hc.2]
  exact HintReached.reached_promises p h hm hn pid hp hk i hi

theorem witness_promises (storage : TypedSource.Storage) (pid i : Nat)
    (hp : pid < grid (witness storage)) (hi : i < trip (witness storage) pid) :
    TilePromises 128 (rowValues (witness storage) pid i) ∧
    TilePromises 256 (columnValues (witness storage) pid i) := by
  have hc := values_match _ (witness_admissible storage) pid hp (witness_count_pos storage) i hi
  rw [hc.1,hc.2]
  exact witness_reached_promises storage pid i hp hi

theorem tail_counterexample (storage : TypedSource.Storage) (i : Nat)
    (hi : i < count (tailWitness storage)) :
    Aligned 128 (rowValues (tailWitness storage) 3 i) ∧
    Aligned 256 (columnValues (tailWitness storage) 3 i) ∧
    ¬ Contiguous 128 (rowValues (tailWitness storage) 3 i) ∧
    ¬ Contiguous 256 (columnValues (tailWitness storage) 3 i) := by
  have hb := tail_reached_bounds storage i hi
  have hc := values_match _ (tail_admissible storage) 3 hb.1 (tail_geometry storage).2.2.1 i hb.2
  rw [hc.1,hc.2]
  exact ⟨(tail_still_aligned storage i hi).1,(tail_still_aligned storage i hi).2,
    (tail_not_contiguous storage i hi).1,(tail_not_contiguous storage i hi).2⟩

theorem tail_hint_site_reached (storage : TypedSource.Storage) :
    ((NativeLoop.run (tailWitness storage) 3 1).ki == bits 0) = Bool.true := by
  have hk := (tail_geometry storage).2.2.1
  have hb := tail_reached_bounds storage 0 hk
  exact first_restart _ (tail_admissible storage) 3 hb.1 hk hb.2

end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.HintCached
