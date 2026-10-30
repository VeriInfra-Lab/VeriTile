import bench.tritonbench_g.rms_rbe_matmul.Proofs.KernelGraph
import bench.tritonbench_g.rms_rbe_matmul.Proofs.KernelTarget
import bench.tritonbench_g.rms_matmul_rbe.Proofs.RmsMatmulRbe_Real
import bench.tritonbench_g.triton_linear_activation.Proofs.KernelRealArithmetic

namespace VeriTile.Bench.Repairs.RmsRbeMatmul.KernelReal
open VeriTile.Triton FP.Structural
open KernelSource (Operations Parameters)
open KernelGraph (Input Tile)
open Int8DequantMatmul (rsum rsum_swap rsum_congr)
noncomputable section
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

def reduction {s : TileShape} (ax : Fin s.length) (keep : Bool) (xs : Values ℝ .real s) :
    Values ℝ .real (TileShape.reduceShape s ax keep) :=
  match keep with
  | Bool.false => fun out => ∑ k, xs (TileShape.insertAxisIndex s ax out k)
  | Bool.true => fun out => ∑ k, xs (TileShape.replaceAxisIndex s ax out k)

/-- Exact arithmetic interpretation, distinct from the native word model.
`decode` is the scalar mathematical interpretation of original E5 input bytes. -/
def operations (decode : Float8E5Bits → ℝ) : Operations ℝ where
  scalar := {
    literal := fun _ _ x => x
    negInf := 0
    binary := fun _ _ op x y => match op with
      | .add => x+y | .sub => x-y | .mul => x*y | .div => x/y | .max => max x y | .pow => Real.rpow x y
    unary := fun _ op x => match op with
      | .rsqrt => 1/Real.sqrt x | .sqrt => Real.sqrt x
      | .sin => Real.sin x | .cos => Real.cos x | .neg _ => -x | .abs _ => |x| | _ => x
    cast := fun _ _ _ => id
    fromNat := fun _ n => n
    fromInt := fun _ n => n
    fp32Bits := fun _ => 0
    fp32Load := id
    reduceMax := fun _ _ _ _ _ => 0
    reduceSum := fun _ => reduction }
  power := fun _ x n => x^n
  decode := fun _ => decode
  dot := fun _ _ _ _ _ => TritonLinearActivation.KernelRealArithmetic.laneDot
    (fun x y => ∑ j, x j*y j)

def logical (decode : Float8E5Bits → ℝ) (fp8 : Bool) (v : Input ℝ) (b m : Nat) : KernelMath.Input where
  x := v.x b m
  rms := fun k => if fp8 then decode (Float8E5Bits.ofInt (v.rmsByte k)) else v.rmsHalf k
  weight := fun n k => if fp8 then decode (Float8E5Bits.ofInt (v.weightByte n k)) else v.weightHalf n k

theorem rsum_sum (f : Nat → ℝ) (n : Nat) : rsum f n=∑ k∈Finset.range n,f k := by
  induction n with
  | zero => simp [rsum]
  | succ n ih => simp [rsum,ih,Finset.sum_range_succ]
theorem finSum_rsum (f : Nat → ℝ) (n : Nat) : (∑ k : Fin n,f k.val)=rsum f n := by
  rw [Fin.sum_univ_eq_sum_range,rsum_sum]

theorem partition (K : Nat) (f : Nat → ℝ) :
    rsum (fun t => rsum (fun k => if t*64+k<K then f (t*64+k) else 0) 64)
      ((K+63)/64)=rsum f K := RmsMatmulRbe.k_partition K f

theorem dot_real (decode : Float8E5Bits → ℝ) (zero : decode (Float8E5Bits.ofInt 0)=0)
    (fp8 : Bool) (p : Parameters) (v : Input ℝ) (t : Tile) (block : Nat)
    (i : TileIndex [16,64]) (hm : t.row*16+i.1.val<p.M) (hn : t.column*64+i.2.1.val<p.N) :
    (operations decode).dot (some .fp32) .real .real .fp32 {} (batch:=[])
      (KernelGraph.scaled (operations decode) fp8 p v t block)
      (KernelGraph.weight (operations decode) fp8 p v t block) i=
    rsum (fun k => if block*64+k<p.K then
      (logical decode fp8 v t.batch (t.row*16+i.1.val)).x (block*64+k)*
      (logical decode fp8 v t.batch (t.row*16+i.1.val)).rms (block*64+k)*
      (logical decode fp8 v t.batch (t.row*16+i.1.val)).weight
        (t.column*64+i.2.1.val) (block*64+k) else 0) 64 := by
  simp only [operations,TritonLinearActivation.KernelRealArithmetic.laneDot]
  rw [←finSum_rsum]
  apply Finset.sum_congr rfl
  intro k _
  have mask : k.val<p.K-block*64 ↔ block*64+k.val<p.K := by omega
  cases fp8 <;> by_cases active : block*64+k.val<p.K <;>
    simp [KernelGraph.scaled,KernelGraph.x,KernelGraph.rms,KernelGraph.weight,
      KernelGraph.zero,KernelGraph.halfZero,operations,logical,mask,active,zero,
      Nat.mod_eq_of_lt hm,Nat.mod_eq_of_lt hn]
  all_goals simp only [Nat.add_comm]

theorem products_real (decode : Float8E5Bits → ℝ) (zero : decode (Float8E5Bits.ofInt 0)=0)
    (fp8 : Bool) (p : Parameters) (v : Input ℝ) (t : Tile)
    (i : TileIndex [16,64]) (hm : t.row*16+i.1.val<p.M) (hn : t.column*64+i.2.1.val<p.N)
    (count : Nat) :
    KernelGraph.products (operations decode) fp8 p v t count i=
    rsum (fun block => rsum (fun k => if block*64+k<p.K then
      (logical decode fp8 v t.batch (t.row*16+i.1.val)).x (block*64+k)*
      (logical decode fp8 v t.batch (t.row*16+i.1.val)).rms (block*64+k)*
      (logical decode fp8 v t.batch (t.row*16+i.1.val)).weight
        (t.column*64+i.2.1.val) (block*64+k) else 0) 64) count := by
  induction count with
  | zero => rfl
  | succ count ih =>
    change KernelGraph.products (operations decode) fp8 p v t count i+_=_
    rw [dot_real decode zero fp8 p v t count i hm hn,ih]
    rfl

theorem moments_real (decode : Float8E5Bits → ℝ) (p : Parameters) (v : Input ℝ) (t : Tile)
    (i : TileIndex [16,64]) (hm : t.row*16+i.1.val<p.M) (count : Nat) :
    KernelGraph.moments (operations decode) p v t count i=
      rsum (fun block => if block*64+i.2.1.val<p.K then
        (v.x t.batch (t.row*16+i.1.val) (block*64+i.2.1.val))^2 else 0) count := by
  induction count with
  | zero => rfl
  | succ count ih =>
    have mask : i.2.1.val<p.K-count*64 ↔ count*64+i.2.1.val<p.K := by omega
    change KernelGraph.moments (operations decode) p v t count i+
      (KernelGraph.x (operations decode) p v t count i)^(2:Int)=_
    rw [ih]
    by_cases active : count*64+i.2.1.val<p.K <;>
      simp [rsum,KernelGraph.x,KernelGraph.zero,operations,mask,active,Nat.mod_eq_of_lt hm]
    all_goals try simp only [Nat.add_comm]
    all_goals try rfl
    all_goals norm_num

theorem projection_real (decode : Float8E5Bits → ℝ) (zero : decode (Float8E5Bits.ofInt 0)=0)
    (fp8 : Bool) (p : Parameters) (v : Input ℝ) (t : Tile)
    (i : TileIndex [16,64]) (hm : t.row*16+i.1.val<p.M) (hn : t.column*64+i.2.1.val<p.N) :
    KernelGraph.projection (operations decode) fp8 p v t i=
      KernelMath.projection p.K (logical decode fp8 v t.batch (t.row*16+i.1.val))
        (t.column*64+i.2.1.val) := by
  unfold KernelGraph.projection
  rw [products_real decode zero fp8 p v t i hm hn]
  rw [partition p.K (fun k =>
    (logical decode fp8 v t.batch (t.row*16+i.1.val)).x k*
    (logical decode fp8 v t.batch (t.row*16+i.1.val)).rms k*
    (logical decode fp8 v t.batch (t.row*16+i.1.val)).weight (t.column*64+i.2.1.val) k)]
  have hs : (∑ k : Fin 64, KernelGraph.moments (operations decode) p v t ((p.K+63)/64)
      (i.1,k,PUnit.unit))=rsum (fun k => (v.x t.batch (t.row*16+i.1.val) k)^2) p.K := by
    have hmoment (k : Fin 64) := moments_real decode p v t (i.1,k,PUnit.unit) hm ((p.K+63)/64)
    simp_rw [hmoment]
    rw [finSum_rsum (fun lane => rsum (fun block => if block*64+lane<p.K then
      (v.x t.batch (t.row*16+i.1.val) (block*64+lane))^2 else 0) ((p.K+63)/64)) 64,
      rsum_swap,partition p.K
      (fun k => (v.x t.batch (t.row*16+i.1.val) k)^2)]
  change _*(1/Real.sqrt (((∑ k : Fin 64,
    KernelGraph.moments (operations decode) p v t ((p.K+63)/64) (i.1,k,PUnit.unit))/(p.K:ℝ))+
      (1/1000000)))=_
  rw [hs]
  simp only [KernelMath.projection,KernelMath.inverse,KernelMath.squareMean,
    KernelMath.epsilon,logical,rsum_sum]

#axiomsClean dot_real
#axiomsClean projection_real
end
end VeriTile.Bench.Repairs.RmsRbeMatmul.KernelReal
