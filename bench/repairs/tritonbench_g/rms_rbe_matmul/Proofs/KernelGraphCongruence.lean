import bench.repairs.tritonbench_g.rms_rbe_matmul.Proofs.KernelGraph

namespace VeriTile.Bench.Repairs.RmsRbeMatmul.KernelGraph
open VeriTile.Triton FP.Structural
open KernelSource (Operations Parameters)
noncomputable section
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false
variable {α : Type}

/-- Equality only of ordinary logical input elements consumed by this batch.
No masked padding cell or completed output is a premise. -/
structure Agreement (p : Parameters) (batch : Nat) (v w : Input α) : Prop where
  x : ∀ m k, m<p.M → k<p.K → v.x batch m k=w.x batch m k
  rmsHalf : ∀ k, k<p.K → v.rmsHalf k=w.rmsHalf k
  rmsByte : ∀ k, k<p.K → v.rmsByte k=w.rmsByte k
  weightHalf : ∀ n k, n<p.N → k<p.K → v.weightHalf n k=w.weightHalf n k
  weightByte : ∀ n k, n<p.N → k<p.K → v.weightByte n k=w.weightByte n k

theorem x_congr (a : Operations α) (p : Parameters) (v w : Input α) (t : Tile)
    (hm : 0<p.M) (h : Agreement p t.batch v w) (block : Nat) : x a p v t block=x a p w t block := by
  funext i
  by_cases mask : i.2.1.val<p.K-block*64
  · have hk : i.2.1.val+block*64<p.K := by omega
    simp only [x,mask,if_true,h.x _ _ (Nat.mod_lt _ hm) hk]
  · simp only [x,mask,if_false]
theorem rms_congr (a : Operations α) (fp8 : Bool) (p : Parameters) (v w : Input α) (t : Tile)
    (h : Agreement p t.batch v w) (block lane : Nat) : rms a fp8 p v block lane=rms a fp8 p w block lane := by
  by_cases mask : lane<p.K-block*64
  · have hk : lane+block*64<p.K := by omega
    cases fp8 <;> simp only [rms,mask,if_true,if_false,Bool.false_eq_true,
      h.rmsHalf _ hk,h.rmsByte _ hk]
  · cases fp8 <;> simp [rms,mask]
theorem weight_congr (a : Operations α) (fp8 : Bool) (p : Parameters) (v w : Input α) (t : Tile)
    (hn : 0<p.N) (h : Agreement p t.batch v w) (block : Nat) :
    weight a fp8 p v t block=weight a fp8 p w t block := by
  funext i
  by_cases mask : i.1.val<p.K-block*64
  · have hk : i.1.val+block*64<p.K := by omega
    cases fp8 <;> simp only [weight,mask,if_true,if_false,Bool.false_eq_true,
      h.weightHalf _ _ (Nat.mod_lt _ hn) hk,h.weightByte _ _ (Nat.mod_lt _ hn) hk]
  · cases fp8 <;> simp [weight,mask]
theorem scaled_congr (a : Operations α) (fp8 : Bool) (p : Parameters) (v w : Input α) (t : Tile)
    (hm : 0<p.M) (h : Agreement p t.batch v w) (block : Nat) :
    scaled a fp8 p v t block=scaled a fp8 p w t block := by
  funext i
  simp only [scaled,x_congr a p v w t hm h,rms_congr a fp8 p v w t h]
theorem products_congr (a : Operations α) (fp8 : Bool) (p : Parameters) (v w : Input α) (t : Tile)
    (hm : 0<p.M) (hn : 0<p.N) (h : Agreement p t.batch v w) (count : Nat) :
    products a fp8 p v t count=products a fp8 p w t count := by
  induction count with
  | zero => rfl
  | succ count ih =>
    funext i
    simp only [products,ih,scaled_congr a fp8 p v w t hm h,weight_congr a fp8 p v w t hn h]
theorem moments_congr (a : Operations α) (p : Parameters) (v w : Input α) (t : Tile)
    (hm : 0<p.M) (h : Agreement p t.batch v w) (count : Nat) :
    moments a p v t count=moments a p w t count := by
  induction count with
  | zero => rfl
  | succ count ih =>
    funext i
    simp only [moments,ih,x_congr a p v w t hm h]
theorem projection_congr (a : Operations α) (fp8 : Bool) (p : Parameters) (v w : Input α) (t : Tile)
    (hm : 0<p.M) (hn : 0<p.N) (h : Agreement p t.batch v w) :
    projection a fp8 p v t=projection a fp8 p w t := by
  funext i
  simp only [projection,products_congr a fp8 p v w t hm hn h,moments_congr a p v w t hm h]
theorem output_congr (a : Operations α) (fp8 : Bool) (p : Parameters) (v w : Input α) (t : Tile)
    (hm : 0<p.M) (hn : 0<p.N) (h : Agreement p t.batch v w) (i : TileIndex [16,64]) :
    output a fp8 p v t i=output a fp8 p w t i := by
  simp only [output,projection_congr a fp8 p v w t hm hn h]

#axiomsClean output_congr
end
end VeriTile.Bench.Repairs.RmsRbeMatmul.KernelGraph
