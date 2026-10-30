import bench.tritonbench_g.matrix_reduction.Proofs.BitEncoding
namespace VeriTile.Bench.TritonBenchG.MatrixReduction.BinaryConversion
set_option maxRecDepth 16000
set_option maxHeartbeats 1000000

theorem widenMagnitude_cases (m : Nat) (hm : m < 32768) :
    (31744 ≤ m ∧ m < 32768 ∧ widenMagnitude m = 2139095040 + (m % 1024) * 8192) ∨
    (1024 ≤ m ∧ m < 31744 ∧ widenMagnitude m = (m + 114688) * 8192) ∨
    (512 ≤ m ∧ m < 1024 ∧ widenMagnitude m = 939524096 + (m - 512) * 16384) ∨
    (256 ≤ m ∧ m < 512 ∧ widenMagnitude m = 931135488 + (m - 256) * 32768) ∨
    (128 ≤ m ∧ m < 256 ∧ widenMagnitude m = 922746880 + (m - 128) * 65536) ∨
    (64 ≤ m ∧ m < 128 ∧ widenMagnitude m = 914358272 + (m - 64) * 131072) ∨
    (32 ≤ m ∧ m < 64 ∧ widenMagnitude m = 905969664 + (m - 32) * 262144) ∨
    (16 ≤ m ∧ m < 32 ∧ widenMagnitude m = 897581056 + (m - 16) * 524288) ∨
    (8 ≤ m ∧ m < 16 ∧ widenMagnitude m = 889192448 + (m - 8) * 1048576) ∨
    (4 ≤ m ∧ m < 8 ∧ widenMagnitude m = 880803840 + (m - 4) * 2097152) ∨
    (2 ≤ m ∧ m < 4 ∧ widenMagnitude m = 872415232 + (m - 2) * 4194304) ∨
    (1 ≤ m ∧ m < 2 ∧ widenMagnitude m = 864026624) ∨
    (0 ≤ m ∧ m < 1 ∧ widenMagnitude m = 0) := by
  by_cases h31744 : 31744 ≤ m
  · left
    refine ⟨h31744, by omega, ?_⟩
    simp only [widenMagnitude, if_pos h31744]
  right
  by_cases h1024 : 1024 ≤ m
  · left
    refine ⟨h1024, by omega, ?_⟩
    simp only [widenMagnitude, if_neg h31744, if_pos h1024]
  right
  by_cases h512 : 512 ≤ m
  · left
    refine ⟨h512, by omega, ?_⟩
    simp only [widenMagnitude, if_neg h31744, if_neg h1024, if_pos h512]
  right
  by_cases h256 : 256 ≤ m
  · left
    refine ⟨h256, by omega, ?_⟩
    simp only [widenMagnitude, if_neg h31744, if_neg h1024, if_neg h512, if_pos h256]
  right
  by_cases h128 : 128 ≤ m
  · left
    refine ⟨h128, by omega, ?_⟩
    simp only [widenMagnitude, if_neg h31744, if_neg h1024, if_neg h512, if_neg h256, if_pos h128]
  right
  by_cases h64 : 64 ≤ m
  · left
    refine ⟨h64, by omega, ?_⟩
    simp only [widenMagnitude, if_neg h31744, if_neg h1024, if_neg h512, if_neg h256, if_neg h128, if_pos h64]
  right
  by_cases h32 : 32 ≤ m
  · left
    refine ⟨h32, by omega, ?_⟩
    simp only [widenMagnitude, if_neg h31744, if_neg h1024, if_neg h512, if_neg h256, if_neg h128, if_neg h64, if_pos h32]
  right
  by_cases h16 : 16 ≤ m
  · left
    refine ⟨h16, by omega, ?_⟩
    simp only [widenMagnitude, if_neg h31744, if_neg h1024, if_neg h512, if_neg h256, if_neg h128, if_neg h64, if_neg h32, if_pos h16]
  right
  by_cases h8 : 8 ≤ m
  · left
    refine ⟨h8, by omega, ?_⟩
    simp only [widenMagnitude, if_neg h31744, if_neg h1024, if_neg h512, if_neg h256, if_neg h128, if_neg h64, if_neg h32, if_neg h16, if_pos h8]
  right
  by_cases h4 : 4 ≤ m
  · left
    refine ⟨h4, by omega, ?_⟩
    simp only [widenMagnitude, if_neg h31744, if_neg h1024, if_neg h512, if_neg h256, if_neg h128, if_neg h64, if_neg h32, if_neg h16, if_neg h8, if_pos h4]
  right
  by_cases h2 : 2 ≤ m
  · left
    refine ⟨h2, by omega, ?_⟩
    simp only [widenMagnitude, if_neg h31744, if_neg h1024, if_neg h512, if_neg h256, if_neg h128, if_neg h64, if_neg h32, if_neg h16, if_neg h8, if_neg h4, if_pos h2]
  right
  by_cases h1 : 1 ≤ m
  · left
    refine ⟨h1, by omega, ?_⟩
    simp only [widenMagnitude, if_neg h31744, if_neg h1024, if_neg h512, if_neg h256, if_neg h128, if_neg h64, if_neg h32, if_neg h16, if_neg h8, if_neg h4, if_neg h2, if_pos h1]
  right
  refine ⟨by omega, by omega, ?_⟩
  simp only [widenMagnitude, if_neg h31744, if_neg h1024, if_neg h512, if_neg h256, if_neg h128, if_neg h64, if_neg h32, if_neg h16, if_neg h8, if_neg h4, if_neg h2, if_neg h1]

theorem widenMagnitude_lt (m : Nat) (hm : m < 32768) :
    widenMagnitude m < 2147483648 := by
  rcases widenMagnitude_cases m hm with ⟨hl, hu, he⟩ | ⟨hl, hu, he⟩ | ⟨hl, hu, he⟩ | ⟨hl, hu, he⟩ | ⟨hl, hu, he⟩ | ⟨hl, hu, he⟩ | ⟨hl, hu, he⟩ | ⟨hl, hu, he⟩ | ⟨hl, hu, he⟩ | ⟨hl, hu, he⟩ | ⟨hl, hu, he⟩ | ⟨hl, hu, he⟩ | ⟨hl, hu, he⟩
  all_goals rw [he]
  all_goals omega

theorem widenMagnitude_zero_iff (m : Nat) (hm : m < 32768) :
    widenMagnitude m = 0 ↔ m = 0 := by
  rcases widenMagnitude_cases m hm with ⟨hl, hu, he⟩ | ⟨hl, hu, he⟩ | ⟨hl, hu, he⟩ | ⟨hl, hu, he⟩ | ⟨hl, hu, he⟩ | ⟨hl, hu, he⟩ | ⟨hl, hu, he⟩ | ⟨hl, hu, he⟩ | ⟨hl, hu, he⟩ | ⟨hl, hu, he⟩ | ⟨hl, hu, he⟩ | ⟨hl, hu, he⟩ | ⟨hl, hu, he⟩
  all_goals rw [he]
  all_goals omega

theorem widenMagnitude_order (m n : Nat) (hm : m < 32768) (hn : n < 32768) :
    widenMagnitude m ≤ widenMagnitude n ↔ m ≤ n := by
  rcases widenMagnitude_cases m hm with ⟨hl, hu, he⟩ | ⟨hl, hu, he⟩ | ⟨hl, hu, he⟩ | ⟨hl, hu, he⟩ | ⟨hl, hu, he⟩ | ⟨hl, hu, he⟩ | ⟨hl, hu, he⟩ | ⟨hl, hu, he⟩ | ⟨hl, hu, he⟩ | ⟨hl, hu, he⟩ | ⟨hl, hu, he⟩ | ⟨hl, hu, he⟩ | ⟨hl, hu, he⟩
  all_goals rw [he]
  all_goals rcases widenMagnitude_cases n hn with ⟨hl2, hu2, he2⟩ | ⟨hl2, hu2, he2⟩ | ⟨hl2, hu2, he2⟩ | ⟨hl2, hu2, he2⟩ | ⟨hl2, hu2, he2⟩ | ⟨hl2, hu2, he2⟩ | ⟨hl2, hu2, he2⟩ | ⟨hl2, hu2, he2⟩ | ⟨hl2, hu2, he2⟩ | ⟨hl2, hu2, he2⟩ | ⟨hl2, hu2, he2⟩ | ⟨hl2, hu2, he2⟩ | ⟨hl2, hu2, he2⟩
  all_goals rw [he2]
  all_goals omega

theorem narrowMagnitude_widenMagnitude (m : Nat) (hm : m < 32768) :
    narrowMagnitude (widenMagnitude m) = m := by
  rcases widenMagnitude_cases m hm with ⟨hl, hu, he⟩ | ⟨hl, hu, he⟩ | ⟨hl, hu, he⟩ | ⟨hl, hu, he⟩ | ⟨hl, hu, he⟩ | ⟨hl, hu, he⟩ | ⟨hl, hu, he⟩ | ⟨hl, hu, he⟩ | ⟨hl, hu, he⟩ | ⟨hl, hu, he⟩ | ⟨hl, hu, he⟩ | ⟨hl, hu, he⟩ | ⟨hl, hu, he⟩
  all_goals rw [he]
  · simp only [narrowMagnitude, if_pos (show 2139095040 ≤ 2139095040 + (m % 1024) * 8192 by omega)]
    omega
  · simp only [narrowMagnitude, if_neg (show ¬ 2139095040 ≤ (m + 114688) * 8192 by omega),
      if_pos (show 947912704 ≤ (m + 114688) * 8192 by omega)]
    omega
  · simp only [narrowMagnitude, if_neg (show ¬ 2139095040 ≤ 939524096 + (m - 512) * 16384 by omega),
      if_neg (show ¬ 947912704 ≤ 939524096 + (m - 512) * 16384 by omega),
      if_pos (show 939524096 ≤ 939524096 + (m - 512) * 16384 by omega)]
    omega
  · simp only [narrowMagnitude, if_neg (show ¬ 2139095040 ≤ 931135488 + (m - 256) * 32768 by omega),
      if_neg (show ¬ 947912704 ≤ 931135488 + (m - 256) * 32768 by omega),
      if_neg (show ¬ 939524096 ≤ 931135488 + (m - 256) * 32768 by omega),
      if_pos (show 931135488 ≤ 931135488 + (m - 256) * 32768 by omega)]
    omega
  · simp only [narrowMagnitude, if_neg (show ¬ 2139095040 ≤ 922746880 + (m - 128) * 65536 by omega),
      if_neg (show ¬ 947912704 ≤ 922746880 + (m - 128) * 65536 by omega),
      if_neg (show ¬ 939524096 ≤ 922746880 + (m - 128) * 65536 by omega),
      if_neg (show ¬ 931135488 ≤ 922746880 + (m - 128) * 65536 by omega),
      if_pos (show 922746880 ≤ 922746880 + (m - 128) * 65536 by omega)]
    omega
  · simp only [narrowMagnitude, if_neg (show ¬ 2139095040 ≤ 914358272 + (m - 64) * 131072 by omega),
      if_neg (show ¬ 947912704 ≤ 914358272 + (m - 64) * 131072 by omega),
      if_neg (show ¬ 939524096 ≤ 914358272 + (m - 64) * 131072 by omega),
      if_neg (show ¬ 931135488 ≤ 914358272 + (m - 64) * 131072 by omega),
      if_neg (show ¬ 922746880 ≤ 914358272 + (m - 64) * 131072 by omega),
      if_pos (show 914358272 ≤ 914358272 + (m - 64) * 131072 by omega)]
    omega
  · simp only [narrowMagnitude, if_neg (show ¬ 2139095040 ≤ 905969664 + (m - 32) * 262144 by omega),
      if_neg (show ¬ 947912704 ≤ 905969664 + (m - 32) * 262144 by omega),
      if_neg (show ¬ 939524096 ≤ 905969664 + (m - 32) * 262144 by omega),
      if_neg (show ¬ 931135488 ≤ 905969664 + (m - 32) * 262144 by omega),
      if_neg (show ¬ 922746880 ≤ 905969664 + (m - 32) * 262144 by omega),
      if_neg (show ¬ 914358272 ≤ 905969664 + (m - 32) * 262144 by omega),
      if_pos (show 905969664 ≤ 905969664 + (m - 32) * 262144 by omega)]
    omega
  · simp only [narrowMagnitude, if_neg (show ¬ 2139095040 ≤ 897581056 + (m - 16) * 524288 by omega),
      if_neg (show ¬ 947912704 ≤ 897581056 + (m - 16) * 524288 by omega),
      if_neg (show ¬ 939524096 ≤ 897581056 + (m - 16) * 524288 by omega),
      if_neg (show ¬ 931135488 ≤ 897581056 + (m - 16) * 524288 by omega),
      if_neg (show ¬ 922746880 ≤ 897581056 + (m - 16) * 524288 by omega),
      if_neg (show ¬ 914358272 ≤ 897581056 + (m - 16) * 524288 by omega),
      if_neg (show ¬ 905969664 ≤ 897581056 + (m - 16) * 524288 by omega),
      if_pos (show 897581056 ≤ 897581056 + (m - 16) * 524288 by omega)]
    omega
  · simp only [narrowMagnitude, if_neg (show ¬ 2139095040 ≤ 889192448 + (m - 8) * 1048576 by omega),
      if_neg (show ¬ 947912704 ≤ 889192448 + (m - 8) * 1048576 by omega),
      if_neg (show ¬ 939524096 ≤ 889192448 + (m - 8) * 1048576 by omega),
      if_neg (show ¬ 931135488 ≤ 889192448 + (m - 8) * 1048576 by omega),
      if_neg (show ¬ 922746880 ≤ 889192448 + (m - 8) * 1048576 by omega),
      if_neg (show ¬ 914358272 ≤ 889192448 + (m - 8) * 1048576 by omega),
      if_neg (show ¬ 905969664 ≤ 889192448 + (m - 8) * 1048576 by omega),
      if_neg (show ¬ 897581056 ≤ 889192448 + (m - 8) * 1048576 by omega),
      if_pos (show 889192448 ≤ 889192448 + (m - 8) * 1048576 by omega)]
    omega
  · simp only [narrowMagnitude, if_neg (show ¬ 2139095040 ≤ 880803840 + (m - 4) * 2097152 by omega),
      if_neg (show ¬ 947912704 ≤ 880803840 + (m - 4) * 2097152 by omega),
      if_neg (show ¬ 939524096 ≤ 880803840 + (m - 4) * 2097152 by omega),
      if_neg (show ¬ 931135488 ≤ 880803840 + (m - 4) * 2097152 by omega),
      if_neg (show ¬ 922746880 ≤ 880803840 + (m - 4) * 2097152 by omega),
      if_neg (show ¬ 914358272 ≤ 880803840 + (m - 4) * 2097152 by omega),
      if_neg (show ¬ 905969664 ≤ 880803840 + (m - 4) * 2097152 by omega),
      if_neg (show ¬ 897581056 ≤ 880803840 + (m - 4) * 2097152 by omega),
      if_neg (show ¬ 889192448 ≤ 880803840 + (m - 4) * 2097152 by omega),
      if_pos (show 880803840 ≤ 880803840 + (m - 4) * 2097152 by omega)]
    omega
  · simp only [narrowMagnitude, if_neg (show ¬ 2139095040 ≤ 872415232 + (m - 2) * 4194304 by omega),
      if_neg (show ¬ 947912704 ≤ 872415232 + (m - 2) * 4194304 by omega),
      if_neg (show ¬ 939524096 ≤ 872415232 + (m - 2) * 4194304 by omega),
      if_neg (show ¬ 931135488 ≤ 872415232 + (m - 2) * 4194304 by omega),
      if_neg (show ¬ 922746880 ≤ 872415232 + (m - 2) * 4194304 by omega),
      if_neg (show ¬ 914358272 ≤ 872415232 + (m - 2) * 4194304 by omega),
      if_neg (show ¬ 905969664 ≤ 872415232 + (m - 2) * 4194304 by omega),
      if_neg (show ¬ 897581056 ≤ 872415232 + (m - 2) * 4194304 by omega),
      if_neg (show ¬ 889192448 ≤ 872415232 + (m - 2) * 4194304 by omega),
      if_neg (show ¬ 880803840 ≤ 872415232 + (m - 2) * 4194304 by omega),
      if_pos (show 872415232 ≤ 872415232 + (m - 2) * 4194304 by omega)]
    omega
  · simp only [narrowMagnitude, if_neg (show ¬ 2139095040 ≤ 864026624 by omega),
      if_neg (show ¬ 947912704 ≤ 864026624 by omega),
      if_neg (show ¬ 939524096 ≤ 864026624 by omega),
      if_neg (show ¬ 931135488 ≤ 864026624 by omega),
      if_neg (show ¬ 922746880 ≤ 864026624 by omega),
      if_neg (show ¬ 914358272 ≤ 864026624 by omega),
      if_neg (show ¬ 905969664 ≤ 864026624 by omega),
      if_neg (show ¬ 897581056 ≤ 864026624 by omega),
      if_neg (show ¬ 889192448 ≤ 864026624 by omega),
      if_neg (show ¬ 880803840 ≤ 864026624 by omega),
      if_neg (show ¬ 872415232 ≤ 864026624 by omega),
      if_pos (show 864026624 ≤ 864026624 by omega)]
    omega
  · simp only [narrowMagnitude, if_neg (show ¬ 2139095040 ≤ 0 by omega),
      if_neg (show ¬ 947912704 ≤ 0 by omega),
      if_neg (show ¬ 939524096 ≤ 0 by omega),
      if_neg (show ¬ 931135488 ≤ 0 by omega),
      if_neg (show ¬ 922746880 ≤ 0 by omega),
      if_neg (show ¬ 914358272 ≤ 0 by omega),
      if_neg (show ¬ 905969664 ≤ 0 by omega),
      if_neg (show ¬ 897581056 ≤ 0 by omega),
      if_neg (show ¬ 889192448 ≤ 0 by omega),
      if_neg (show ¬ 880803840 ≤ 0 by omega),
      if_neg (show ¬ 872415232 ≤ 0 by omega),
      if_neg (show ¬ 864026624 ≤ 0 by omega)]
    omega

theorem widen_toNat (x : BitVec 16) :
    (widen x).toNat = (x.toNat / 32768) * 2147483648 + widenMagnitude (x.toNat % 32768) := by
  have h : x.toNat < 65536 := x.isLt
  have hm := widenMagnitude_lt (x.toNat % 32768) (Nat.mod_lt _ (by decide))
  simp only [widen, BitVec.toNat_ofNat]
  omega

theorem widen_magnitude (x : BitVec 16) :
    (widen x).toNat % 2147483648 = widenMagnitude (x.toNat % 32768) := by
  have h : x.toNat < 65536 := x.isLt
  have hm := widenMagnitude_lt (x.toNat % 32768) (Nat.mod_lt _ (by decide))
  rw [widen_toNat]
  omega

theorem widen_preserves_sign (x : BitVec 16) :
    (widen x).toNat / 2147483648 = x.toNat / 32768 := by
  have h : x.toNat < 65536 := x.isLt
  have hm := widenMagnitude_lt (x.toNat % 32768) (Nat.mod_lt _ (by decide))
  rw [widen_toNat]
  omega

theorem narrow_widen (x : BitVec 16) : narrow (widen x) = x := by
  apply BitVec.eq_of_toNat_eq
  simp only [narrow, BitVec.toNat_ofNat, widen_preserves_sign, widen_magnitude,
    narrowMagnitude_widenMagnitude _ (Nat.mod_lt _ (by decide))]
  have h : x.toNat < 65536 := x.isLt
  omega

theorem widen_order (x y : BitVec 16) :
    key32 (widen x) ≤ key32 (widen y) ↔ key16 x ≤ key16 y := by
  change (key32 (widen x)).toNat ≤ (key32 (widen y)).toNat ↔
    (key16 x).toNat ≤ (key16 y).toNat
  rw [key32_toNat, key32_toNat, key16_toNat, key16_toNat,
    widen_magnitude, widen_magnitude, widen_toNat, widen_toNat]
  have hx : x.toNat < 65536 := x.isLt
  have hy : y.toNat < 65536 := y.isLt
  have hmx := widenMagnitude_lt (x.toNat % 32768) (Nat.mod_lt _ (by decide))
  have hmy := widenMagnitude_lt (y.toNat % 32768) (Nat.mod_lt _ (by decide))
  have hzx := widenMagnitude_zero_iff (x.toNat % 32768) (Nat.mod_lt _ (by decide))
  have hzy := widenMagnitude_zero_iff (y.toNat % 32768) (Nat.mod_lt _ (by decide))
  have hxy := widenMagnitude_order (x.toNat % 32768) (y.toNat % 32768)
    (Nat.mod_lt _ (by decide)) (Nat.mod_lt _ (by decide))
  have hyx := widenMagnitude_order (y.toNat % 32768) (x.toNat % 32768)
    (Nat.mod_lt _ (by decide)) (Nat.mod_lt _ (by decide))
  repeat' first | split | omega

private theorem subnormal_magnitude (m : Nat) (hm : m < 1024) :
    (if widenMagnitude m / 8388608 = 0 then widenMagnitude m % 8388608
      else (8388608 + widenMagnitude m % 8388608) * 2^(widenMagnitude m / 8388608 - 1)) =
    m * 2^125 := by
  by_cases hz : m = 0
  · subst m
    norm_num [widenMagnitude]
  have hh : ¬ 31744 ≤ m := by omega
  have hn : ¬ 1024 ≤ m := by omega
  simp only [widenMagnitude, if_neg hh, if_neg hn]
  by_cases h512 : 512 ≤ m
  · simp only [if_pos h512]
    have he : (939524096 + (m - 512) * 16384) / 8388608 = 112 := by omega
    have hf : 8388608 + (939524096 + (m - 512) * 16384) % 8388608 = m * 16384 := by omega
    rw [he]
    simp only [show (112: Nat) ≠ 0 by decide, if_false, hf]
    norm_num
    omega
  simp only [if_neg h512]
  by_cases h256 : 256 ≤ m
  · simp only [if_pos h256]
    have he : (931135488 + (m - 256) * 32768) / 8388608 = 111 := by omega
    have hf : 8388608 + (931135488 + (m - 256) * 32768) % 8388608 = m * 32768 := by omega
    rw [he]
    simp only [show (111: Nat) ≠ 0 by decide, if_false, hf]
    norm_num
    omega
  simp only [if_neg h256]
  by_cases h128 : 128 ≤ m
  · simp only [if_pos h128]
    have he : (922746880 + (m - 128) * 65536) / 8388608 = 110 := by omega
    have hf : 8388608 + (922746880 + (m - 128) * 65536) % 8388608 = m * 65536 := by omega
    rw [he]
    simp only [show (110: Nat) ≠ 0 by decide, if_false, hf]
    norm_num
    omega
  simp only [if_neg h128]
  by_cases h64 : 64 ≤ m
  · simp only [if_pos h64]
    have he : (914358272 + (m - 64) * 131072) / 8388608 = 109 := by omega
    have hf : 8388608 + (914358272 + (m - 64) * 131072) % 8388608 = m * 131072 := by omega
    rw [he]
    simp only [show (109: Nat) ≠ 0 by decide, if_false, hf]
    norm_num
    omega
  simp only [if_neg h64]
  by_cases h32 : 32 ≤ m
  · simp only [if_pos h32]
    have he : (905969664 + (m - 32) * 262144) / 8388608 = 108 := by omega
    have hf : 8388608 + (905969664 + (m - 32) * 262144) % 8388608 = m * 262144 := by omega
    rw [he]
    simp only [show (108: Nat) ≠ 0 by decide, if_false, hf]
    norm_num
    omega
  simp only [if_neg h32]
  by_cases h16 : 16 ≤ m
  · simp only [if_pos h16]
    have he : (897581056 + (m - 16) * 524288) / 8388608 = 107 := by omega
    have hf : 8388608 + (897581056 + (m - 16) * 524288) % 8388608 = m * 524288 := by omega
    rw [he]
    simp only [show (107: Nat) ≠ 0 by decide, if_false, hf]
    norm_num
    omega
  simp only [if_neg h16]
  by_cases h8 : 8 ≤ m
  · simp only [if_pos h8]
    have he : (889192448 + (m - 8) * 1048576) / 8388608 = 106 := by omega
    have hf : 8388608 + (889192448 + (m - 8) * 1048576) % 8388608 = m * 1048576 := by omega
    rw [he]
    simp only [show (106: Nat) ≠ 0 by decide, if_false, hf]
    norm_num
    omega
  simp only [if_neg h8]
  by_cases h4 : 4 ≤ m
  · simp only [if_pos h4]
    have he : (880803840 + (m - 4) * 2097152) / 8388608 = 105 := by omega
    have hf : 8388608 + (880803840 + (m - 4) * 2097152) % 8388608 = m * 2097152 := by omega
    rw [he]
    simp only [show (105: Nat) ≠ 0 by decide, if_false, hf]
    norm_num
    omega
  simp only [if_neg h4]
  by_cases h2 : 2 ≤ m
  · simp only [if_pos h2]
    have he : (872415232 + (m - 2) * 4194304) / 8388608 = 104 := by omega
    have hf : 8388608 + (872415232 + (m - 2) * 4194304) % 8388608 = m * 4194304 := by omega
    rw [he]
    simp only [show (104: Nat) ≠ 0 by decide, if_false, hf]
    norm_num
    omega
  simp only [if_neg h2]
  by_cases h1 : 1 ≤ m
  · simp only [if_pos h1]
    have he : 864026624 / 8388608 = 103 := by omega
    have hf : 8388608 + 864026624 % 8388608 = m * 8388608 := by omega
    rw [he]
    simp only [show (103: Nat) ≠ 0 by decide, if_false, hf]
    norm_num
    omega
  simp only [if_neg h1]
  omega

theorem widen_preserves_finite_magnitude (x : BitVec 16)
    (hfinite : x.toNat % 32768 / 1024 < 31) :
    magnitude32 (widen x) = magnitude16 x := by
  let m := x.toNat % 32768
  have hm : m < 31744 := by dsimp [m]; omega
  simp only [magnitude16, magnitude32, widen_magnitude]
  change (if widenMagnitude m / 8388608 = 0 then widenMagnitude m % 8388608
    else (8388608 + widenMagnitude m % 8388608) * 2^(widenMagnitude m / 8388608 - 1)) =
    if m / 1024 = 0 then m % 1024 * 2^125 else (1024 + m % 1024) * 2^(m / 1024 + 124)
  by_cases hn : 1024 ≤ m
  · have hw : widenMagnitude m = (m + 114688) * 8192 := by
      simp only [widenMagnitude, if_neg (show ¬ 31744 ≤ m by omega), if_pos hn]
    have he : widenMagnitude m / 8388608 = m / 1024 + 112 := by rw [hw]; omega
    have hf : widenMagnitude m % 8388608 = (m % 1024) * 8192 := by rw [hw]; omega
    rw [he, hf, if_neg (by omega), if_neg (by omega)]
    have he' : m / 1024 + 112 - 1 = m / 1024 + 111 := by omega
    have hp : m / 1024 + 124 = (m / 1024 + 111) + 13 := by omega
    rw [he', hp, Nat.pow_add]
    norm_num
    ring
  · have he : m / 1024 = 0 := by omega
    have hf : m % 1024 = m := by omega
    rw [he, if_pos rfl, hf]
    exact subnormal_magnitude m (by omega)

theorem widen_positive_infinity : widen (0x7c00 : BitVec 16) = (0x7f800000 : BitVec 32) := by decide

theorem widen_negative_infinity : widen (0xfc00 : BitVec 16) = (0xff800000 : BitVec 32) := by decide

end VeriTile.Bench.TritonBenchG.MatrixReduction.BinaryConversion
