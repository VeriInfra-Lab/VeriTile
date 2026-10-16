import bench.tritonbench_g.fused_recurrent_delta.Proofs.FPBackwardKernels
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentDelta.FPStages
open VeriTile.Triton FP.Structural
noncomputable section

/-! Independent ordered scalar/reduction graphs for the backward stages.
Zero padding is supplied to the backend but never removed by a mathematical
zero law. Multiplication order, reduction axes and intermediate casts are
preserved even for NaN/infinity/signed-zero-sensitive backends. -/
def backStateAddress {α : Type} (s : State α) (K V BK BV : Nat) (i : TileIndex [BK,BV]) : Nat :=
  s.pids 2 * K * V + keyIndex s BK i.1 * V + valueIndex s BV i.2.1

def keyOutputAddress {α : Type} (s : State α) (t stride B H K BK : Nat) (i : TileIndex [BK]) : Nat :=
  (s.pids 2 + s.pids 0 * B * H) * stride + s.pids 1 * BK + i.1.val + t * K

def betaOutputAddress {α : Type} (s : State α) (t stride NK B H V BV : Nat) (i : TileIndex [BV]) : Nat :=
  (s.pids 2 + s.pids 1 * B * H + s.pids 0 * B * H * NK) * stride + i.1.val + t * V

def scalarBetaAddress {α : Type} (s : State α) (t B H T : Nat) : Nat :=
  (s.pids 2 + s.pids 0 * B * H) * T + t

def loadKey {α : Type} [Inhabited α] (M : Algebra α) (s : State α) (R : RegionName)
    (t stride K BK : Nat) (i : Fin BK) : α :=
  M.cast (some .fp32) .real .real
    (maskedInput M s R (keyAddress s t stride K BK (i,PUnit.unit)) (keyIndex s BK i < K))

def loadValue {α : Type} [Inhabited α] (M : Algebra α) (s : State α) (R : RegionName)
    (t stride V BV : Nat) (i : Fin BV) : α :=
  M.cast (some .fp32) .real .real
    (maskedInput M s R (valueAddress s t stride V BV (i,PUnit.unit)) (valueIndex s BV i < V))

def loadBeta {α : Type} [Inhabited α] (M : Algebra α) (s : State α) (R : RegionName)
    (headwise : Bool) (t stride T V BV : Nat) (i : Fin BV) : α :=
  if headwise then loadValue M s R t stride V BV i
  else M.cast (some .fp32) .real .real ((s.mem R (s.pids 2 * T + t)).read .real)

def loadBackState {α : Type} [Inhabited α] (M : Algebra α) (s : State α) (R : RegionName)
    (K V BK BV : Nat) (i : TileIndex [BK,BV]) : α :=
  M.fp32Load (maskedInput M s R (backStateAddress s K V BK BV i)
    (keyIndex s BK i.1 < K ∧ valueIndex s BV i.2.1 < V))

def dhValues {α : Type} [Inhabited α] (M : Algebra α) (s : State α) (DHPrev q do_ : RegionName)
    (t s_qk_h s_vo_h K V BK BV : Nat) (scale : ℝ) : TileIndex [BK,BV] → α := fun i =>
  M.binary (some .fp32) .real .add (loadBackState M s DHPrev K V BK BV i)
    (M.binary (some .fp32) .real .mul
      (M.binary (some .fp32) .real .mul (loadKey M s q t s_qk_h K BK i.1)
        (M.literal (some .fp32) .real scale))
      (loadValue M s do_ t s_vo_h V BV i.2.1))

def rawDvValues {α : Type} [Inhabited α] (M : Algebra α) (s : State α) (DHPrev q do_ k : RegionName)
    (t s_qk_h s_vo_h K V BK BV : Nat) (scale : ℝ) : TileIndex [BV] → α :=
  M.reduceSum (some .fp32) (shape := [BK,BV]) ⟨0, by simp⟩ Bool.false (fun i =>
    M.binary (some .fp32) .real .mul (dhValues M s DHPrev q do_ t s_qk_h s_vo_h K V BK BV scale i)
      (loadKey M s k t s_qk_h K BK i.1))

def dkValues {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (DHPrev q do_ v beta : RegionName) (headwise : Bool)
    (t s_qk_h s_vo_h T K V BK BV : Nat) (scale : ℝ) : TileIndex [BK] → α :=
  let sums := M.reduceSum (some .fp32) (shape := [BK,BV]) ⟨1, by simp⟩ Bool.false (fun i =>
    M.binary (some .fp32) .real .mul (dhValues M s DHPrev q do_ t s_qk_h s_vo_h K V BK BV scale i)
      (M.binary (some .fp32) .real .mul (loadValue M s v t s_vo_h V BV i.2.1)
        (loadBeta M s beta headwise t s_vo_h T V BV i.2.1)))
  fun i => M.cast (some .fp32) .real .real (sums i)

def dvValues {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (DHPrev q do_ k beta : RegionName) (headwise : Bool)
    (t s_qk_h s_vo_h T K V BK BV : Nat) (scale : ℝ) : TileIndex [BV] → α := fun i =>
  M.cast (some .fp32) .real .real (M.binary (some .fp32) .real .mul
    (rawDvValues M s DHPrev q do_ k t s_qk_h s_vo_h K V BK BV scale i)
    (loadBeta M s beta headwise t s_vo_h T V BV i.1))

def rawDbetaValues {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (DHPrev q do_ k v : RegionName) (t s_qk_h s_vo_h K V BK BV : Nat) (scale : ℝ) : TileIndex [BV] → α := fun i =>
  M.binary (some .fp32) .real .mul (rawDvValues M s DHPrev q do_ k t s_qk_h s_vo_h K V BK BV scale i)
    (loadValue M s v t s_vo_h V BV i.1)

def dbetaValues {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (DHPrev q do_ k v : RegionName) (t s_qk_h s_vo_h K V BK BV : Nat) (scale : ℝ) : TileIndex [BV] → α := fun i =>
  M.cast (some .fp32) .real .real (rawDbetaValues M s DHPrev q do_ k v t s_qk_h s_vo_h K V BK BV scale i)

def scalarDbetaValue {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (DHPrev q do_ k v : RegionName) (t s_qk_h s_vo_h K V BK BV : Nat) (scale : ℝ) : α :=
  M.cast (some .fp32) .real .real
    (M.reduceSum (some .fp32) (shape := [BV]) ⟨0, by simp⟩ Bool.false
      (rawDbetaValues M s DHPrev q do_ k v t s_qk_h s_vo_h K V BK BV scale) PUnit.unit)

def dqValues {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (HPrev k v beta do_ : RegionName) (headwise : Bool)
    (t s_qk_h s_vo_h T K V BK BV : Nat) (scale : ℝ) : TileIndex [BK] → α :=
  let products : TileIndex [BK,BV] → α := fun i =>
    M.binary (some .fp32) .real .mul
      (M.binary (some .fp32) .real .add (loadBackState M s HPrev K V BK BV i)
        (M.binary (some .fp32) .real .mul (loadKey M s k t s_qk_h K BK i.1)
          (M.binary (some .fp32) .real .mul (loadValue M s v t s_vo_h V BV i.2.1)
            (loadBeta M s beta headwise t s_vo_h T V BV i.2.1))))
      (loadValue M s do_ t s_vo_h V BV i.2.1)
  let sums := M.reduceSum (some .fp32) (shape := [BK,BV]) ⟨1, by simp⟩ Bool.false products
  fun i => M.cast (some .fp32) .real .real (M.binary (some .fp32) .real .mul (sums i)
    (M.literal (some .fp32) .real scale))

def correctionValues {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (HPrev dv dk : RegionName) (t s_qk_h s_vo_h B H K V BK BV : Nat) : TileIndex [BK] → α :=
  let products : TileIndex [BK,BV] → α := fun i =>
    M.binary (some .fp32) .real .mul
      (M.cast (some .fp32) .real .real (maskedInput M s dv
        (outputAddress s t s_vo_h B H V BV i.2) (valueIndex s BV i.2.1 < V)))
      (loadBackState M s HPrev K V BK BV i)
  let sums := M.reduceSum (some .fp32) (shape := [BK,BV]) ⟨1, by simp⟩ Bool.false products
  fun i => M.cast (some .fp32) .real .real (M.binary (some .fp32) .real .sub
    (M.cast (some .fp32) .real .real (maskedInput M s dk
      (keyOutputAddress s t s_qk_h B H K BK i) (keyIndex s BK i.1 < K))) (sums i))

theorem keyOutputAddress_injective {α : Type} (s : State α) (t stride B H K BK : Nat) :
    Function.Injective (keyOutputAddress s t stride B H K BK) := by
  intro i j h
  have he : i.1 = j.1 := by
    apply Fin.ext
    dsimp [keyOutputAddress] at h
    omega
  exact Prod.ext he (Subsingleton.elim _ _)

theorem betaOutputAddress_injective {α : Type} (s : State α) (t stride NK B H V BV : Nat) :
    Function.Injective (betaOutputAddress s t stride NK B H V BV) := by
  intro i j h
  have he : i.1 = j.1 := by
    apply Fin.ext
    dsimp [betaOutputAddress] at h
    omega
  exact Prod.ext he (Subsingleton.elim _ _)
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentDelta.FPStages
