import bench.repairs.tritonbench_g.rmsnorm_fused_llama.Proofs.KernelBinding
import bench.repairs.tritonbench_g.rmsnorm_fused_llama.Proofs.Native
import bench.repairs.tritonbench_g.rmsnorm_fused_llama.Proofs.RuntimeGrid

namespace VeriTile.Bench.Repairs.RmsnormFusedLlama.RuntimeReference
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

def std (R : Reductions) (xs : Nat → Term) (n B : Nat) (eps : Term) : Term :=
  let variance := div (reduction R Bool.false (some .fp32) (s:=[B]) ⟨0,by simp⟩ Bool.false
    (squares xs n B) PUnit.unit) (lit n)
  div (lit 1) (.unary (some .fp32) .sqrt (add variance eps))

def expression (R : Reductions) (xs ws : Nat → Term) (n B : Nat) (eps : Term) (k : Nat) : Term :=
  .cast (some .fp32) .fp16 .real (.cast (some .fp32) .real .fp16
    (mul (mul (cast32 (xs k)) (std R xs n B eps)) (cast32 (ws k))))

/-- Only the supported scalar instructions and compiler's structural choices
interpret the independent graph. No kernel/state/heap result occurs here. -/
def native (I : Instructions) (R : Reductions) (S : Schedule)
    (xs ws : Nat → Term) (n B : Nat) (eps : Term) (k : Nat) : Word :=
  eval I S (expression R xs ws n B eps k)

theorem source_value (R : Reductions) (s : State Term) (X W : RegionName)
    (row n B : Nat) (eps : Term) (xs ws : Nat → Term)
    (hx : ∀ j,j<n → RuntimeRow.read s X (row*n+j)=xs j)
    (hw : ∀ j,j<n → RuntimeRow.read s W j=ws j) (k : Nat) (hk : k<n) :
    RuntimeRow.value (algebra R) s X W row n B eps k=expression R xs ws n B eps k := by
  have hsq : RuntimeRow.square (algebra R) s X row n B=squares xs n B := by
    funext i
    simp only [RuntimeRow.square,RuntimeRow.xload,squares]
    by_cases hi : i.1.val<n
    · simp [hi,hx i.1.val hi,RuntimeRow.cast,RuntimeRow.lit,RuntimeRow.add,RuntimeRow.mul,cast32,lit,add,mul,algebra]
    · simp [hi,RuntimeRow.cast,RuntimeRow.lit,RuntimeRow.add,RuntimeRow.mul,cast32,lit,add,mul,algebra]
  unfold RuntimeRow.value RuntimeRow.rstd RuntimeRow.variance
  rw [hsq]
  simp [RuntimeRow.xload,RuntimeRow.wload,hk,hx k hk,hw k hk,
    RuntimeRow.cast,RuntimeRow.lit,RuntimeRow.add,RuntimeRow.mul,RuntimeRow.div,expression,std,cast32,lit,add,mul,div,algebra]

#axiomsClean source_value
#guard_msgs (drop info) in
#specNonCircular native avoiding [runtimeKernel,State,RuntimeRow.value,RuntimeGrid.run,RuntimeGrid.graph]
end
end VeriTile.Bench.Repairs.RmsnormFusedLlama.RuntimeReference
