import bench.tritonbench_g.int8_dequant_matmul.Proofs.IOGeometry
import bench.tritonbench_g.int8_dequant_matmul.Proofs.IOStoreGeometry

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.IOSafety
open VeriTile Triton FP.Structural Quote
open IntScaledMatmul.Revision001
noncomputable section
set_option maxHeartbeats 2000000

/-- Capacities of the actual declared source windows. These contain no
payload condition and remain valid after every arbitrary patch of C. -/
structure Bounds (s : Shape) (hasBias : Bool) (bounds : RegionName → Nat) : Prop where
  a : ∀ r, r<s.M → ∀ k, k<s.K → r*s.K+k<bounds "A"
  b : ∀ k, k<s.K → ∀ n, n<s.N → k*s.N+n<bounds "B"
  x : ∀ r, r<s.M → r<bounds "state_x"
  w : ∀ n, n<s.N → n<bounds "state_w"
  bias : hasBias=Bool.true → ∀ n, n<s.N → n<bounds "bias"
  c : ∀ r, r<s.M → ∀ n, n<s.N → r*s.N+n<bounds "C"

theorem bounds_of_io (F H : Type) (s : Shape) (c : Config) (hasBias : Bool)
    (vals : (IOLayout.memory F H s c hasBias).Inputs) (bounds : RegionName → Nat)
    (hb : (IOLayout.memory F H s c hasBias).InBounds bounds vals) : Bounds s hasBias bounds := by
  constructor
  · intro r hr k hk
    have h := hb.2.1 (0 : Fin 6) (⟨r,hr⟩,⟨k,hk⟩,()) (by simp [IOLayout.memory])
    change TileShape.linearIndex [s.M,s.K] (⟨r,hr⟩,⟨k,hk⟩,())<bounds "A" at h
    simpa [TileShape.linearIndex] using h
  · intro k hk n hn
    have h := hb.2.1 (1 : Fin 6) (⟨k,hk⟩,⟨n,hn⟩,()) (by simp [IOLayout.memory])
    change TileShape.linearIndex [s.K,s.N] (⟨k,hk⟩,⟨n,hn⟩,())<bounds "B" at h
    simpa [TileShape.linearIndex] using h
  · intro r hr
    have h := hb.2.1 (2 : Fin 6) (⟨r,hr⟩,()) (by simp [IOLayout.memory])
    change TileShape.linearIndex [s.M] (⟨r,hr⟩,())<bounds "state_x" at h
    simpa [TileShape.linearIndex] using h
  · intro n hn
    have h := hb.2.1 (3 : Fin 6) (⟨n,hn⟩,()) (by simp [IOLayout.memory])
    change TileShape.linearIndex [s.N] (⟨n,hn⟩,())<bounds "state_w" at h
    simpa [TileShape.linearIndex] using h
  · intro bias n hn
    have h := hb.2.1 (4 : Fin 6) (⟨n,hn⟩,()) (by simp [IOLayout.memory,bias])
    change TileShape.linearIndex [s.N] (⟨n,hn⟩,())<bounds "bias" at h
    simpa [TileShape.linearIndex] using h
  · intro r hr n hn
    have h := hb.2.2.1 (0 : Fin 1) (⟨r,hr⟩,⟨n,hn⟩,()) (by trivial)
    change TileShape.linearIndex [s.M,s.N] (⟨r,hr⟩,⟨n,hn⟩,())<bounds "C" at h
    simpa [TileShape.linearIndex] using h

variable {α : Type} [Inhabited α] (M : Algebra α) (q : IntegerModel I F H)
  (s : Shape) (c : Config) (hasBias : Bool) (initial : State α)
  (valid : initial.pids 0<Geometry.count (geometry s c))
  (hz : initial.pids 1<c.S) (bounds : RegionName → Nat) (hb : Bounds s hasBias bounds)

include valid hz hb

theorem a_bound (t : Nat) (ht : t<steps s c) (i : TileIndex [c.BM,c.BK])
    (enabled : s.K%(c.BK*c.S)=0 ∨
      register (preludeExecution M q s c hasBias initial).val .int [c.BK] "rk" (i.2.1,())<
        (s.K-t*(c.BK*c.S) : Nat)) :
    (ap s c (preludeExecution M q s c hasBias initial).val t i).2 <
      bounds (ap s c (preludeExecution M q s c hasBias initial).val t i).1 := by
  have rk := prelude_rk M q s c hasBias initial
  have active : sourceK c t (initial.pids 1) i.2.1.val<s.K := by
    rcases enabled with even|mask
    · exact even_active s c even t _ _ ht hz i.2.1.isLt
    · rw [rk] at mask
      have mask' : initial.pids 1*c.BK+i.2.1.val<s.K-t*(c.BK*c.S) := by
        simpa only [←Int.natCast_mul,←Int.natCast_add,Int.ofNat_lt] using mask
      exact (source_mask s c t _ _ ht hz i.2.1.isLt).mp mask'
  simp only [ap,prelude_a M q s c hasBias initial valid]
  have h := hb.a (row s c (initial.pids 0) i.1%s.M)
    (Nat.mod_lt _ (IOGeometry.rows_pos s c _ valid)) _ active
  simpa [sourceK,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using h

theorem b_bound (t : Nat) (ht : t<steps s c) (i : TileIndex [c.BK,c.BN])
    (enabled : s.K%(c.BK*c.S)=0 ∨
      register (preludeExecution M q s c hasBias initial).val .int [c.BK] "rk" (i.1,())<
        (s.K-t*(c.BK*c.S) : Nat)) :
    (bp s c (preludeExecution M q s c hasBias initial).val t i).2 <
      bounds (bp s c (preludeExecution M q s c hasBias initial).val t i).1 := by
  have rk := prelude_rk M q s c hasBias initial
  have active : sourceK c t (initial.pids 1) i.1.val<s.K := by
    rcases enabled with even|mask
    · exact even_active s c even t _ _ ht hz i.1.isLt
    · rw [rk] at mask
      have mask' : initial.pids 1*c.BK+i.1.val<s.K-t*(c.BK*c.S) := by
        simpa only [←Int.natCast_mul,←Int.natCast_add,Int.ofNat_lt] using mask
      exact (source_mask s c t _ _ ht hz i.1.isLt).mp mask'
  simp only [bp,prelude_b M q s c hasBias initial valid]
  have h := hb.b _ active (col s c (initial.pids 0) i.2.1%s.N)
    (Nat.mod_lt _ (IOGeometry.columns_pos s c _ valid))
  simpa [sourceK,Nat.add_mul,Nat.mul_assoc,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using h

theorem bias_bound (enabled : hasBias=Bool.true) (i : Fin c.BN)
    (mask : register (preludeExecution M q s c hasBias initial).val .int [c.BN] "rn" (i,())<(s.N:Int)) :
    (register (preludeExecution M q s c hasBias initial).val .int [c.BN] "rn" (i,())).toNat<bounds "bias" := by
  rw [prelude_rn M q s c hasBias initial valid] at mask ⊢
  simp only [Int.toNat_natCast,Int.ofNat_lt] at mask ⊢
  exact hb.bias enabled _ mask

theorem output_bound (i : TileIndex [c.BM,c.BN])
    (enabled : Quote.active s c (preludeExecution M q s c hasBias initial).val i) :
    offset s c (preludeExecution M q s c hasBias initial).val i<bounds "C" := by
  obtain ⟨cell,addr,_⟩ := IOStoreGeometry.active_cell M q s c hasBias initial valid i enabled
  rw [addr]
  exact hb.c _ cell.1.isLt _ cell.2.isLt

#axiomsClean bounds_of_io
#axiomsClean a_bound
#axiomsClean b_bound
#axiomsClean bias_bound
#axiomsClean output_bound
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.IOSafety
