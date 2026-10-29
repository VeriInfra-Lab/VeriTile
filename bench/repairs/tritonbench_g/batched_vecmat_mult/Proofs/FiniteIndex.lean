import bench.repairs.tritonbench_g.batched_vecmat_mult.Proofs.IndexModel
import bench.tritonbench_g.batched_vecmat_mult.Proofs.HostAccess

namespace VeriTile.Bench.Repairs.BatchedVecmatMult.Index64
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.BatchedVecmatMult.ValueProofs

@[simp] theorem coordinate_eq (pid block lane : Nat) :
    coordinate pid block lane = word (pid*block+lane) := by
  simp [coordinate, ← BitVec.ofNat_mul, ← BitVec.ofNat_add]

@[simp] theorem outOffset_eq (p : Parameters) (pm pn i j : Nat) :
    outOffset p pm pn i j = word ((pm*p.BM+i)*p.N+(pn*p.BN+j)) := by
  simp [outOffset, ← BitVec.ofNat_mul, ← BitVec.ofNat_add]

@[simp] theorem aOffset_eq (p : Parameters) (pm c i e : Nat) :
    aOffset p pm c i e = word ((pm*p.BM+i)*p.K+(c*p.BK+e)) := by
  simp [aOffset, ← BitVec.ofNat_mul, ← BitVec.ofNat_add]

@[simp] theorem bOffset_eq (p : Parameters) (pm pn c i j e : Nat) :
    bOffset p pm pn c i j e =
      word ((pm*p.BM+i)*p.N*p.K+(pn*p.BN+j)*p.K+(c*p.BK+e)) := by
  simp [bOffset, ← BitVec.ofNat_mul, ← BitVec.ofNat_add]

theorem word_toNat {x : Nat} (h : x < 2^64) : (word x).toNat = x := by
  exact Nat.mod_eq_of_lt h

theorem word_toInt {x : Nat} (h : x < 2^63) : (word x).toInt = (x : Int) := by
  have hn : (word x).toNat = x := word_toNat (by omega)
  rw [BitVec.toInt_eq_toNat_of_lt (by rw [hn]; omega), hn]

/-- A CUDA program ID is signed-i32 nonnegative on an admitted grid. Its
source `.to(tl.int64)` is exact, including grids whose tile-coordinate product
exceeds i32. Grid setup errors are ordinary API outcomes. -/
theorem program_id_cast (pid : Nat) (h : pid < 2^31) :
    ((BitVec.ofNat 32 pid).signExtend 64).toInt = (pid : Int) := by
  rw [BitVec.toInt_signExtend_of_le (by decide)]
  have hn : (BitVec.ofNat 32 pid).toNat = pid := Nat.mod_eq_of_lt (by omega)
  rw [BitVec.toInt_eq_toNat_of_lt (by rw [hn]; omega), hn]

/-- Exact signed input conversion as a word equality; this also binds every
arange lane, whose compiler-supported block extent is below 2^31. -/
theorem signed32_cast_word (x : Nat) (h : x < 2^31) :
    (BitVec.ofNat 32 x).signExtend 64 = word x := by
  apply BitVec.eq_of_toInt_eq
  rw [program_id_cast x h, word_toInt (by omega)]

theorem arange_cast_word (block : Nat) (lane : Fin block) (hblock : block ≤ 2^31) :
    (BitVec.ofNat 32 lane.val).signExtend 64 = word lane.val := by
  exact signed32_cast_word lane.val (lt_of_lt_of_le lane.isLt hblock)

/-- Auxiliary unsigned conversion fact. Installed Triton does not use u32
for this wrapper's Python dimension values; its actual ABI is recorded in Host. -/
theorem unsigned32_cast_word (x : Nat) (h : x < 2^32) :
    (BitVec.ofNat 32 x).setWidth 64 = word x := by
  apply BitVec.eq_of_toNat_eq
  rw [BitVec.toNat_setWidth, BitVec.toNat_ofNat, Nat.mod_eq_of_lt h,
    Nat.mod_eq_of_lt (by omega), word_toNat (by omega)]

/-- The signed i64 loop bound equals natural floor division. A valid Triton
arange block is positive and at most the compiler's static tile limit; the
weaker representability fact below suffices. -/
theorem loop_bound_exact (K BK : Nat) (hK : K < 2^63) (hBK : BK < 2^63) :
    ((word K).sdiv (word BK)).toNat = K/BK := by
  have hk : (word K).msb = Bool.false := BitVec.msb_eq_false_iff_two_mul_lt.mpr
    (by rw [word_toNat (by omega)]; omega)
  have hb : (word BK).msb = Bool.false := BitVec.msb_eq_false_iff_two_mul_lt.mpr
    (by rw [word_toNat (by omega)]; omega)
  rw [BitVec.toNat_sdiv, hk, hb]
  change ((word K) / (word BK)).toNat = K/BK
  rw [BitVec.toNat_udiv, word_toNat (by omega), word_toNat (by omega)]

/-- Each dynamic induction value and the final increment to the stopping
value stay signed-i64 nonnegative; the loop visits exactly the natural range. -/
theorem loop_increment_exact (K BK c : Nat) (hK : K < 2^63) (hc : c < K/BK) :
    (word c).toInt = (c : Int) ∧
    (word c + word 1).toInt = ((c+1 : Nat) : Int) := by
  have hd := Nat.div_le_self K BK
  refine ⟨word_toInt (by omega), ?_⟩
  rw [← BitVec.ofNat_add]
  exact word_toInt (by omega)

/-- All output lanes, including K=0: exact signed offset and all four bytes
within the allocated output extent. No reduction iteration is required. -/
theorem all_output_addresses (p : Parameters) (s : StorageLimits p)
    (pm : Fin (gridM p)) (pn : Fin (gridN p)) (i : Fin p.BM) (j : Fin p.BN) :
    (outOffset p pm.val pn.val i.val j.val).toInt =
      (((pm.val*p.BM+i.val)*p.N+(pn.val*p.BN+j.val) : Nat) : Int) ∧
    ((pm.val*p.BM+i.val)*p.N+(pn.val*p.BN+j.val))*4+3 < p.M*p.N*4 := by
  have h := row_major_bound (tiled_coordinate_lt p.M p.BM pm i)
    (tiled_coordinate_lt p.N p.BN pn j)
  constructor
  · rw [outOffset_eq]; apply word_toInt; have := s.output; omega
  · omega

/-- Every A/B load of every actual program, iteration and lane. Arithmetic is
64-bit at each node, and signed pointer extension gives the natural address. -/
theorem all_input_addresses (p : Parameters) (s : StorageLimits p)
    (pm : Fin (gridM p)) (pn : Fin (gridN p)) (c : Fin (p.K/p.BK))
    (i : Fin p.BM) (j : Fin p.BN) (e : Fin p.BK) :
    (aOffset p pm.val c.val i.val e.val).toInt =
      (((pm.val*p.BM+i.val)*p.K+(c.val*p.BK+e.val) : Nat) : Int) ∧
    (bOffset p pm.val pn.val c.val i.val j.val e.val).toInt =
      (((pm.val*p.BM+i.val)*p.N*p.K+(pn.val*p.BN+j.val)*p.K+(c.val*p.BK+e.val) : Nat) : Int) ∧
    ((pm.val*p.BM+i.val)*p.K+(c.val*p.BK+e.val))*4+3 < p.M*p.K*4 ∧
    ((pm.val*p.BM+i.val)*p.N*p.K+(pn.val*p.BN+j.val)*p.K+(c.val*p.BK+e.val))*4+3 < p.M*p.N*p.K*4 := by
  obtain ⟨ha,hb,_⟩ := all_unmasked_addresses_in_declared_extents p pm pn c i j e
  refine ⟨?_,?_,by omega,by omega⟩
  · rw [aOffset_eq]; apply word_toInt; have := s.A; omega
  · rw [bOffset_eq]; apply word_toInt; have := s.B; omega

/-- The complete source integer expression has only additions and products.
Its nonnegative intermediate nodes are bounded by the complete offset whenever
a load/store occurs; the final modular equality above is independent of this
fact. Explicit source coordinate casts and the dynamic K induction are exact. -/
theorem coordinates_exact (p : Parameters) (s : StorageLimits p)
    (pm : Fin (gridM p)) (pn : Fin (gridN p)) (c : Fin (p.K/p.BK))
    (i : Fin p.BM) (j : Fin p.BN) (e : Fin p.BK) :
    (coordinate pm.val p.BM i.val).toInt = ((pm.val*p.BM+i.val : Nat) : Int) ∧
    (coordinate pn.val p.BN j.val).toInt = ((pn.val*p.BN+j.val : Nat) : Int) ∧
    (coordinate c.val p.BK e.val).toInt = ((c.val*p.BK+e.val : Nat) : Int) := by
  have hm := tiled_coordinate_lt p.M p.BM pm i
  have hn := tiled_coordinate_lt p.N p.BN pn j
  have hk := tiled_coordinate_lt p.K p.BK c e
  simp only [coordinate_eq]
  exact ⟨word_toInt (lt_trans hm s.M),word_toInt (lt_trans hn s.N),word_toInt (lt_trans hk s.K)⟩

/-- An allocated address window cannot wrap at pointer addition. This is the
ordinary allocator window fact, independent of shape, tile or numerical data. -/
theorem byte_pointer_exact (base count offset : Nat)
    (hwindow : base+count*4 ≤ 2^64) (h : offset < count) :
    (bytePointer base (word offset)).toNat = base+offset*4 ∧
    base ≤ (bytePointer base (word offset)).toNat ∧
    (bytePointer base (word offset)).toNat+3 < base+count*4 := by
  have he : (bytePointer base (word offset)).toNat = base+offset*4 := by
    unfold bytePointer
    rw [← BitVec.ofNat_mul, ← BitVec.ofNat_add]
    apply word_toNat
    omega
  exact ⟨he,by omega,by omega⟩

/-- Compose every finite-index correspondence for the complete launch domain. -/
theorem complete_address_agreement (p : Parameters) (s : StorageLimits p) :
    AddressAgreement p where
  output pm pn i j := (all_output_addresses p s pm pn i j).1
  inputs pm pn c i j e :=
    ⟨(all_input_addresses p s pm pn c i j e).1,
      (all_input_addresses p s pm pn c i j e).2.1⟩

/-- The original defect witness is now an exact positive 64-bit element offset. -/
theorem original_boundary_repaired :
    (coordinate 2048 16 0 * word 512 * word 128).toInt = 2147483648 := by decide

#axiomsClean program_id_cast
#axiomsClean signed32_cast_word
#axiomsClean arange_cast_word
#axiomsClean unsigned32_cast_word
#axiomsClean loop_bound_exact
#axiomsClean loop_increment_exact
#axiomsClean all_output_addresses
#axiomsClean all_input_addresses
#axiomsClean coordinates_exact
#axiomsClean byte_pointer_exact
#axiomsClean complete_address_agreement
#axiomsClean original_boundary_repaired
end VeriTile.Bench.Repairs.BatchedVecmatMult.Index64
