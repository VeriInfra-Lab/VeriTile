import bench.repairs.tritonbench_g.bmm_chunk_bwd.Kernels
import Mathlib.Data.Nat.ModEq

/-! Local fixed-width address facts. Widened additions and multiplications are
modeled modulo 2^64, including inactive padded pointer lanes. An active byte
address within a real allocation has its natural value. The complete signed-i64
control/address/source/grid composition is in NativeExecution and NativeBoundary. -/
namespace VeriTile.Bench.Repairs.BmmChunkBwd.NativeIndex

def modulus : Nat := 2^64

def wrap (n : Nat) : Nat := n % modulus

inductive Polynomial where
  | input (n : Nat)
  | add (x y : Polynomial)
  | mul (x y : Polynomial)

def ideal : Polynomial → Nat
  | .input n => n
  | .add x y => ideal x + ideal y
  | .mul x y => ideal x * ideal y

def native : Polynomial → Nat
  | .input n => wrap n
  | .add x y => wrap (native x + native y)
  | .mul x y => wrap (native x * native y)

/-- Modular agreement includes arbitrarily large inactive-lane intermediates;
no nonwrapping premise is imposed on unobserved pointers. -/
theorem native_mod (e : Polynomial) : native e = wrap (ideal e) := by
  induction e with
  | input => rfl
  | add x y hx hy =>
    simp only [native,ideal,hx,hy,wrap]
    exact (Nat.add_mod _ _ _).symm
  | mul x y hx hy =>
    simp only [native,ideal,hx,hy,wrap]
    exact (Nat.mul_mod _ _ _).symm

/-- Observable active offsets fit by the tensor allocation's byte capacity. -/
theorem active_exact (e : Polynomial) (bounded : ideal e < modulus) : native e = ideal e := by
  rw [native_mod]
  exact Nat.mod_eq_of_lt bounded

/-- Source tensor offsets retain the original parenthesization, and can denote
A/Res/Out by their own independent strides. `r` may be a padded inactive row. -/
def tensorOffset (b c g r n chunk sb ss sg sk : Nat) : Polynomial :=
  .add (.add (.add (.mul (.input b) (.input sb))
    (.mul (.mul (.input c) (.input chunk)) (.input ss))) (.mul (.input g) (.input sg)))
    (.add (.mul (.input r) (.input ss)) (.mul (.input n) (.input sk)))

def doutOffset (b c g r k sb sc sg sr sk : Nat) : Polynomial :=
  .add (.add (.add (.mul (.input b) (.input sb)) (.mul (.input c) (.input sc)))
    (.mul (.input g) (.input sg)))
    (.add (.mul (.input r) (.input sk)) (.mul (.input k) (.input sr)))

/-- Repeated source pointer advances preserve the same modulo address as the
initial pointer plus the accumulated contraction index. -/
theorem pointer_advance (base k step stride : Nat) :
    wrap (wrap (base+k*stride) + wrap (step*stride)) = wrap (base+(k+step)*stride) := by
  unfold wrap
  rw [← Nat.add_mod]
  congr 1
  simp [Nat.add_mul, Nat.add_assoc]

theorem mask_tail (length block iteration lane : Nat)
    (iterationInside : iteration*block ≤ length) :
    lane < length-iteration*block ↔ iteration*block+lane < length := by omega

/- This states only a local arithmetic bridge. Signed comparisons, cdiv
intermediates, hardware grid limits, and the full native execution adapter
are composed in NativeScalar/NativeMask/NativeExecution; they are not numerical assumptions. -/
#axiomsClean native_mod
#axiomsClean active_exact
#axiomsClean pointer_advance
#axiomsClean mask_tail
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.BmmChunkBwd.NativeIndex
