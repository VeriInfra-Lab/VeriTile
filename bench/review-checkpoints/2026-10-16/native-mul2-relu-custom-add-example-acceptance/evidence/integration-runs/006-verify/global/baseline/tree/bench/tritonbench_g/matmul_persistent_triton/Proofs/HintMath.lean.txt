import bench.tritonbench_g.matmul_persistent_triton.Proofs.NativeDomain

/-! Explicit mathematical obligations for a single hinted one-dimensional
coordinate tile. They do not define or assume compiler correctness. In
particular, aligned contiguous-group starts do not mean that every lane is a
multiple of the block size. -/
namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.HintMath
open NativeDomain
set_option maxHeartbeats 2000000

def Contiguous (block : Nat) (v : Nat → Int) : Prop :=
  ∀ lane, lane < block → v lane = v 0 + (lane : Int)

def Aligned (block : Nat) (v : Nat → Int) : Prop :=
  v 0 % (block : Int) = 0

def TilePromises (block : Nat) (v : Nat → Int) : Prop :=
  0 < block ∧ Aligned block v ∧ Contiguous block v

def clamped (bound block tile lane : Nat) : Int :=
  (clamp bound (tile*block+lane) : Nat)

theorem blocks_mul (block tiles : Nat) (hb : 0 < block) :
    RealMath.blocks (block*tiles) block = tiles := by
  unfold RealMath.blocks
  have he : block*tiles+block-1 = (block-1)+block*tiles := by omega
  rw [he,Nat.add_mul_div_left _ _ hb]
  rw [Nat.div_eq_of_lt (by omega)]
  omega

theorem padded_eq (bound block : Nat) (hb : 0 < block) (hd : block ∣ bound) :
    RealMath.blocks bound block * block = bound := by
  obtain ⟨tiles,rfl⟩ := hd
  rw [blocks_mul block tiles hb,Nat.mul_comm]

theorem full_tile_lane (bound block tile lane : Nat) (hb : 0 < block)
    (hd : block ∣ bound) (ht : tile < RealMath.blocks bound block)
    (hl : lane < block) : tile*block+lane < bound := by
  have h := NativeDomain.lane_lt ht hl
  rwa [padded_eq bound block hb hd] at h

theorem clamped_full (bound block tile lane : Nat) (hb : 0 < block)
    (hd : block ∣ bound) (ht : tile < RealMath.blocks bound block)
    (hl : lane < block) : clamped bound block tile lane = (tile*block+lane : Nat) := by
  simp only [clamped,clamp,if_pos (full_tile_lane bound block tile lane hb hd ht hl)]

theorem full_tile_promises (bound block tile : Nat) (hb : 0 < block)
    (hd : block ∣ bound) (ht : tile < RealMath.blocks bound block) :
    TilePromises block (clamped bound block tile) := by
  have hz := clamped_full bound block tile 0 hb hd ht hb
  refine ⟨hb,?_,?_⟩
  · unfold Aligned
    rw [hz]
    simp
  · intro lane hl
    rw [clamped_full bound block tile lane hb hd ht hl,hz]
    simp

/-- A partial tile with a positive active prefix fails full-block contiguity:
the first inactive lane becomes zero, although its promised value is bound. -/
theorem partial_tile_not_contiguous (bound block tile tail : Nat)
    (hbound : bound = tile*block+tail) (hpos : 0 < tail) (htail : tail < block) :
    ¬ Contiguous block (clamped bound block tile) := by
  intro hc
  have h0 : tile*block < bound := by omega
  have htailbound : ¬tile*block+tail < bound := by omega
  have hh := hc tail htail
  simp only [clamped,clamp,Nat.add_zero,if_pos h0,if_neg htailbound,
    Nat.cast_zero,Nat.cast_mul] at hh
  have hnonneg : 0 ≤ (tile : Int)*(block : Int) := Int.mul_nonneg (by omega) (by omega)
  omega

/-- Clamping alone preserves the first-lane block alignment, even in a tail.
Thus the failed property in the tail examples is consecutive-lane contiguity. -/
theorem clamped_aligned (bound block tile : Nat) :
    Aligned block (clamped bound block tile) := by
  unfold Aligned clamped clamp
  split_ifs <;> simp

theorem every_lane_multiple_is_wrong :
    TilePromises 128 (clamped 256 128 1) ∧
    (clamped 256 128 1 1) % 128 ≠ 0 := by
  refine ⟨full_tile_promises 256 128 1 (by decide) (by decide) (by decide),?_⟩
  norm_num [clamped,clamp]

theorem promises_congr (block : Nat) (v w : Nat → Int)
    (hw : TilePromises block w) (he : ∀ lane, lane < block → v lane = w lane) :
    TilePromises block v := by
  obtain ⟨hb,ha,hc⟩ := hw
  refine ⟨hb,?_,?_⟩
  · change v 0 % (block : Int) = 0
    rw [he 0 hb]
    exact ha
  · intro lane hl
    rw [he lane hl,he 0 hb]
    exact hc lane hl

end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.HintMath
