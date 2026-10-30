import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.Contract

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe
open Direct

/-- Only active GEMM gains a bound. No payload or scalar-scale hypothesis. -/
def LaunchSafe (M N K : Nat) : Prop := M = 0 ∨ N = 0 ∨ K ≤ 131071
def launchSafe (M N K : Nat) : Bool := M == 0 || N == 0 || K ≤ 131071

theorem launchSafe_iff (M N K : Nat) : launchSafe M N K = Bool.true ↔ LaunchSafe M N K := by
  simp [launchSafe, LaunchSafe, or_assoc]

def GemmContract (a : TensorInfo) (as : Option TensorInfo) (b bs : TensorInfo)
    (out : Option TensorInfo) (composed : Bool) : Prop :=
  Direct.GemmContract a as b bs out composed ∧ LaunchSafe (rows a) (cols b) (cols a)

def gemmValid (a : TensorInfo) (as : Option TensorInfo) (b bs : TensorInfo)
    (out : Option TensorInfo) (composed : Bool) : Bool :=
  Direct.gemmValid a as b bs out composed && launchSafe (rows a) (cols b) (cols a)

theorem gemmValid_iff (a : TensorInfo) (as : Option TensorInfo) (b bs : TensorInfo)
    (out : Option TensorInfo) (composed : Bool) :
    gemmValid a as b bs out composed = Bool.true ↔ GemmContract a as b bs out composed := by
  simp only [gemmValid, Bool.and_eq_true, Direct.gemmValid_iff, launchSafe_iff, GemmContract]

theorem active_bound (M N K : Nat) (h : LaunchSafe M N K) (hm : 0 < M) (hn : 0 < N) :
    K ≤ 131071 := by rcases h with h | h | h <;> omega

theorem empty_rows (N K : Nat) : LaunchSafe 0 N K := Or.inl rfl
theorem empty_columns (M K : Nat) : LaunchSafe M 0 K := Or.inr (Or.inl rfl)
theorem zero_inner (M N : Nat) : LaunchSafe M N 0 := Or.inr (Or.inr (by omega))
theorem sharp_boundary : LaunchSafe 1 1 131071 ∧ ¬ LaunchSafe 1 1 131072 := by
  norm_num [LaunchSafe]

#axiomsClean launchSafe_iff
#axiomsClean gemmValid_iff
#axiomsClean active_bound
#axiomsClean sharp_boundary
end VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe
