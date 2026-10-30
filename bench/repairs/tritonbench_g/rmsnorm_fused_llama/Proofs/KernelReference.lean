import bench.repairs.tritonbench_g.rmsnorm_fused_llama.Proofs.KernelBinding
import bench.repairs.tritonbench_g.rmsnorm_fused_llama.Proofs.Native

namespace VeriTile.Bench.Repairs.RmsnormFusedLlama.KernelReference
open VeriTile Triton FP.Structural
open Native
noncomputable section
set_option maxHeartbeats 4000000

/-- Independent single-row source expression, including masked zero lanes,
initial accumulation, casts and the source FP16 output round trip. -/
def lit (x : ℝ) : Term := .literal (some .fp32) .real x
def cast32 (x : Term) : Term := .cast (some .fp32) .real .real x
def add (x y : Term) : Term := .binary (some .fp32) .real .add x y
def mul (x y : Term) : Term := .binary (some .fp32) .real .mul x y
def div (x y : Term) : Term := .binary (some .fp32) .real .div x y

def squares (xs : Nat → Term) (n B : Nat) : Values Term .real [B] := fun i=>
  let x := cast32 (if i.1.val<n then xs i.1.val else lit 0)
  add (lit 0) (mul x x)

def std (R : Reductions) (xs : Nat → Term) (n B : Nat) (eps : ℝ) : Term :=
  let variance := div (reduction R Bool.false (some .fp32) (s:=[B]) ⟨0,by simp⟩ Bool.false
    (squares xs n B) PUnit.unit) (lit n)
  div (lit 1) (.unary (some .fp32) .sqrt (add variance (lit eps)))

def expression (R : Reductions) (xs ws : Nat → Term) (n B : Nat) (eps : ℝ) (k : Nat) : Term :=
  .cast (some .fp32) .fp16 .real (.cast (some .fp32) .real .fp16
    (mul (mul (cast32 (xs k)) (std R xs n B eps)) (cast32 (ws k))))

/-- Only the supported scalar instructions and compiler's structural choices
interpret the independent graph. No kernel/state/heap result occurs here. -/
def native (I : Instructions) (R : Reductions) (S : Schedule)
    (xs ws : Nat → Term) (n B : Nat) (eps : ℝ) (k : Nat) : Word :=
  eval I S (expression R xs ws n B eps k)

theorem source_value (R : Reductions) (s : State Term) (X W : RegionName)
    (row n B : Nat) (eps : ℝ) (xs ws : Nat → Term)
    (hx : ∀ j,j<n → Row.read s X (row*n+j)=xs j)
    (hw : ∀ j,j<n → Row.read s W j=ws j) (k : Nat) (hk : k<n) :
    Row.value (algebra R) s X W row n B eps k=expression R xs ws n B eps k := by
  have hsq : Row.square (algebra R) s X row n B=squares xs n B := by
    funext i
    simp only [Row.square,Row.xload,squares]
    by_cases hi : i.1.val<n
    · simp [hi,hx i.1.val hi,Row.cast,Row.lit,Row.add,Row.mul,cast32,lit,add,mul,algebra]
    · simp [hi,Row.cast,Row.lit,Row.add,Row.mul,cast32,lit,add,mul,algebra]
  unfold Row.value Row.rstd Row.variance
  rw [hsq]
  simp [Row.xload,Row.wload,hk,hx k hk,hw k hk,
    Row.cast,Row.lit,Row.add,Row.mul,Row.div,expression,std,cast32,lit,add,mul,div,algebra]

theorem mathematical_congr (x y w v : Nat → ℝ) (row n : Nat) (eps : ℝ)
    (hx : ∀ j,j<n → x (row*n+j)=y (row*n+j)) (hw : ∀ j,j<n → w j=v j)
    (k : Nat) (hk : k<n) :
    KernelAPI.mathematical x w row n eps k=KernelAPI.mathematical y v row n eps k := by
  unfold KernelAPI.mathematical
  rw [hx k hk,hw k hk]
  congr 6
  apply Finset.sum_congr rfl
  intro j _
  exact congrArg (fun x : ℝ=>x^2) (hx j.val j.isLt)

theorem denominator_positive (xs : Nat → ℝ) (row n : Nat) (eps : ℝ) (hp : 0<eps) :
    0<Real.sqrt ((∑ j : Fin n,xs (row*n+j.val)^2)/(n : ℝ)+eps) := by
  apply Real.sqrt_pos.2
  have nonneg : 0≤∑ j : Fin n,xs (row*n+j.val)^2 := Finset.sum_nonneg (fun _ _=>sq_nonneg _)
  exact add_pos_of_nonneg_of_pos (div_nonneg nonneg (Nat.cast_nonneg _)) hp

#axiomsClean source_value
#axiomsClean denominator_positive
#guard_msgs (drop info) in
#specNonCircular expression avoiding [kernel,State,Row.value,Grid.run,Grid.graph]
#guard_msgs (drop info) in
#specNonCircular native avoiding [kernel,State,Row.value,Grid.run,Grid.graph]
end
end VeriTile.Bench.Repairs.RmsnormFusedLlama.KernelReference
