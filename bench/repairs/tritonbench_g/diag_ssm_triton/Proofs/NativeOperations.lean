import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.NativeBinding

namespace VeriTile.Bench.Repairs.DiagSsmTriton.NativeOperations
open Kernels ByteMemory Execution NativeIndex NativeBinding
set_option autoImplicit false

/-- Static source arithmetic executes even at zero time. A nonempty grid
implies dim>0, so signed remainder agrees with the nonnegative coordinate. -/
theorem static (route : Route) (d : Dimensions) (h : Shape route d)
    (pid lane k : Nat) (hp : pid < grid d.width) (hl : lane < 128) (hk : k < route.components) :
    let width := i64 (i64 d.batch * i64 d.dim)
    let col := i64 (i64 (i64 pid * 128) + lane)
    width = d.width ∧ col = column pid lane ∧
      (col < width ↔ column pid lane < d.width) ∧
      i64 ((col % i64 d.dim) * route.components + k) =
        (route.components * (column pid lane % d.dim) + k : Nat) := by
  have sb := static_bounds route d h pid lane hp hl
  have bp : i64 ((pid : Int) * 128) = (pid * 128 : Nat) := by simpa using exact_mul pid 128 sb.block
  have cc : i64 ((pid * 128 : Nat) + (lane : Int)) = column pid lane := by
    simpa [column] using exact_add (pid * 128) lane sb.columnBound
  dsimp only
  rw [exact_nat _ sb.batch, exact_nat _ sb.dim, exact_mul d.batch d.dim sb.width,
    exact_nat _ sb.pidBound, bp, cc]
  refine ⟨rfl, rfl, ?_, ?_⟩
  · exact_mod_cast Iff.rfl
  · have bound := exact_nat _ (sb.remainder k hk)
    simpa [Int.natCast_mod, Nat.mul_comm, Int.mul_comm] using bound

/-- The intermediate component*batch product and the component width are
both exact before the previous-state pointer subtraction. -/
theorem previous (route : Route) (d : Dimensions) (h : Shape route d)
    (pid lane t k : Nat) (hp : pid < grid d.width) (hl : lane < 128)
    (ht : t < d.length) (ht0 : 0 < t) (hk : k < route.components) :
    let width := i64 (i64 ((route.components : Int) * d.batch) * d.dim)
    let offset := (route.components * (t * d.batch * d.dim + column pid lane) : Int)
    width = (route.components * (d.batch * d.dim) : Nat) ∧
      i64 (i64 (offset - width) + k) =
        (route.components * ((t - 1) * d.batch * d.dim + column pid lane) + k : Nat) := by
  have sb := static_bounds route d h pid lane hp hl
  have tb := time_bounds route d h pid lane t hp hl ht
  have hc : 0 < route.components := by cases route <;> decide
  have scale := exact_mul route.components d.batch sb.componentBatch
  have whole : i64 (((route.components * d.batch : Nat) : Int) * d.dim) =
      (route.components * (d.batch * d.dim) : Nat) := by
    simpa [Nat.mul_assoc] using exact_mul (route.components * d.batch) d.dim
      (by simpa [Nat.mul_assoc] using sb.componentWidth)
  dsimp only
  rw [scale, whole]
  constructor
  · rfl
  · have eq : (route.components : Int) * ((t : Int) * d.batch * d.dim + column pid lane) -
        (route.components * (d.batch * d.dim) : Nat) =
        (route.components * ((t - 1) * d.batch * d.dim + column pid lane) : Nat) := by
      have tt : t = (t - 1) + 1 := by omega
      rw [tt]
      simp only [Nat.add_sub_cancel]
      push_cast
      ring
    rw [eq]
    rw [exact_nat _ (by simpa using tb.previous ht0 0 hc)]
    exact exact_add _ k (tb.previous ht0 k hk)

/-- Actual pointer bytes, including storage_offset, fit unsigned native
addresses. The result is proved from the allocation interval; no zero-offset
or signed-63-bit base-pointer restriction is imposed. -/
theorem pointer_bytes (view : View) (word : Nat) (bound : word < view.words) (byte : Fin 4) :
    ((view.pointer : Int) + 4 * word + byte.val) % (2 ^ 64 : Int) =
      (view.address word + byte.val : Nat) := by
  have native := view.address_native word bound
  have nonnegative : 0 ≤ (view.address word + byte.val : Nat) := Nat.zero_le _
  have hi : (view.address word + byte.val : Nat) < 2 ^ 64 := by omega
  have hiInt : ((view.address word + byte.val : Nat) : Int) < 2 ^ 64 := by exact_mod_cast hi
  have eq : (view.pointer : Int) + 4 * word + byte.val = (view.address word + byte.val : Nat) := by
    simp [View.address]
  rw [eq]
  exact Int.emod_eq_of_lt (by omega) hiInt

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.DiagSsmTriton.NativeOperations
