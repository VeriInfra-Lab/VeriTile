import bench.repairs.tritonbench_g.rms_rbe_matmul_launch_guard.Proofs.PublicLogicalModel

namespace VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard.PublicIO
open VeriTile.Bench.Repairs.RmsRbeMatmul
open VeriTile.Triton FP.Structural KernelSource KernelGraph
noncomputable section
set_option maxHeartbeats 3000000
variable {α : Type} [Inhabited α]

structure FiniteAgreement (fp8 : Bool) (B : Nat) (p : Parameters) (u v : Input α) : Prop where
  x : ∀ b m k, b<B → m<p.M → k<p.K → u.x b m k=v.x b m k
  wh : fp8=Bool.false → ∀ n k, n<p.N → k<p.K → u.weightHalf n k=v.weightHalf n k
  rh : fp8=Bool.false → ∀ k, k<p.K → u.rmsHalf k=v.rmsHalf k
  wb : fp8=Bool.true → ∀ n k, n<p.N → k<p.K → u.weightByte n k=v.weightByte n k
  rb : fp8=Bool.true → ∀ k, k<p.K → u.rmsByte k=v.rmsByte k

theorem graph_x_congr (a : Operations α) (fp8 : Bool) (B : Nat) (p : Parameters)
    (u v : Input α) (same : FiniteAgreement fp8 B p u v) (t : Tile)
    (batch : t.batch<B) (rows : 0<p.M) (block : Nat) :
    KernelGraph.x a p u t block=KernelGraph.x a p v t block := by
  funext i
  unfold KernelGraph.x
  split_ifs with active
  · rw [same.x t.batch _ _ batch (Nat.mod_lt _ rows) (by omega)]
  · rfl

theorem graph_rms_congr (a : Operations α) (fp8 : Bool) (B : Nat) (p : Parameters)
    (u v : Input α) (same : FiniteAgreement fp8 B p u v) (block lane : Nat) :
    KernelGraph.rms a fp8 p u block lane=KernelGraph.rms a fp8 p v block lane := by
  unfold KernelGraph.rms
  cases hf : fp8 <;> simp only [Bool.false_eq_true,if_false,if_true]
  · split_ifs with active
    · rw [same.rh hf (lane+block*64) (by omega)]
    · rfl
  · split_ifs with active
    · rw [same.rb hf _ (by omega)]
    · rfl

theorem graph_weight_congr (a : Operations α) (fp8 : Bool) (B : Nat) (p : Parameters)
    (u v : Input α) (same : FiniteAgreement fp8 B p u v) (t : Tile)
    (cols : 0<p.N) (block : Nat) :
    KernelGraph.weight a fp8 p u t block=KernelGraph.weight a fp8 p v t block := by
  funext i
  unfold KernelGraph.weight
  cases hf : fp8 <;> simp only [Bool.false_eq_true,if_false,if_true]
  · split_ifs with active
    · rw [same.wh hf _ _ (Nat.mod_lt _ cols) (by omega)]
    · rfl
  · split_ifs with active
    · rw [same.wb hf _ _ (Nat.mod_lt _ cols) (by omega)]
    · rfl

theorem graph_scaled_congr (a : Operations α) (fp8 : Bool) (B : Nat) (p : Parameters)
    (u v : Input α) (same : FiniteAgreement fp8 B p u v) (t : Tile)
    (batch : t.batch<B) (rows : 0<p.M) (block : Nat) :
    KernelGraph.scaled a fp8 p u t block=KernelGraph.scaled a fp8 p v t block := by
  funext i
  unfold KernelGraph.scaled
  rw [graph_x_congr a fp8 B p u v same t batch rows block,
    graph_rms_congr a fp8 B p u v same block i.2.1.val]

theorem graph_products_congr (a : Operations α) (fp8 : Bool) (B : Nat) (p : Parameters)
    (u v : Input α) (same : FiniteAgreement fp8 B p u v) (t : Tile)
    (batch : t.batch<B) (rows : 0<p.M) (cols : 0<p.N) (count : Nat) :
    products a fp8 p u t count=products a fp8 p v t count := by
  induction count with
  | zero => rfl
  | succ count ih =>
    funext i
    simp only [products,ih,graph_scaled_congr a fp8 B p u v same t batch rows count,
      graph_weight_congr a fp8 B p u v same t cols count]

theorem graph_moments_congr (a : Operations α) (fp8 : Bool) (B : Nat) (p : Parameters)
    (u v : Input α) (same : FiniteAgreement fp8 B p u v) (t : Tile)
    (batch : t.batch<B) (rows : 0<p.M) (count : Nat) :
    moments a p u t count=moments a p v t count := by
  induction count with
  | zero => rfl
  | succ count ih =>
    funext i
    simp only [moments,ih,graph_x_congr a fp8 B p u v same t batch rows count]

theorem graph_projection_congr (a : Operations α) (fp8 : Bool) (B : Nat) (p : Parameters)
    (u v : Input α) (same : FiniteAgreement fp8 B p u v) (t : Tile)
    (batch : t.batch<B) (rows : 0<p.M) (cols : 0<p.N) :
    projection a fp8 p u t=projection a fp8 p v t := by
  funext i
  unfold projection
  rw [graph_products_congr a fp8 B p u v same t batch rows cols,
    graph_moments_congr a fp8 B p u v same t batch rows]

theorem graph_output_congr (a : Operations α) (fp8 : Bool) (B : Nat) (p : Parameters)
    (u v : Input α) (same : FiniteAgreement fp8 B p u v) (t : Tile)
    (batch : t.batch<B) (rows : 0<p.M) (cols : 0<p.N) :
    KernelGraph.output a fp8 p u t=KernelGraph.output a fp8 p v t := by
  funext i
  simp only [KernelGraph.output,graph_projection_congr a fp8 B p u v same t batch rows cols]

#axiomsClean graph_output_congr
end
end VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard.PublicIO
