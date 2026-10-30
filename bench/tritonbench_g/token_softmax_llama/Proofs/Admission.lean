import bench.tritonbench_g.token_softmax_llama.Contract
import Mathlib.Tactic

namespace VeriTile.Bench.Repairs.TokenSoftmaxLlama.Checked

theorem check_iff (c : Call) : check c=.ok () ↔ MetadataContract c := by
  simp only [check]
  by_cases a : tensorChecks c <;> by_cases b : active c <;> by_cases d : launchChecks c <;>
    simp [a,b,d,MetadataContract.mk,show MetadataContract c ↔ tensorChecks c ∧ (active c → launchChecks c) from
      ⟨fun h => ⟨h.tensors,h.launch⟩,fun h => ⟨h.1,h.2⟩⟩]

/-- The checked row/column separation proves actual output-coordinate
injectivity. Input and metadata may have zero or nonunit strides. -/
theorem ordered_injective (H K R C h t k u : Nat)
    (order : (H ≤ 1 ∨ (K-1)*C < R) ∧ (K ≤ 1 ∨ 0 < C))
    (hh : h < H) (ht : t < K) (hk : k < H) (hu : u < K)
    (same : h*R+t*C=k*R+u*C) : h=k ∧ t=u := by
  have ht' : t*C ≤ (K-1)*C := Nat.mul_le_mul_right C (by omega)
  have hu' : u*C ≤ (K-1)*C := Nat.mul_le_mul_right C (by omega)
  have heads : h=k := by
    rcases order.1 with a|a
    · omega
    · by_contra ne
      rcases lt_or_gt_of_ne ne with less|greater
      · have step := Nat.mul_le_mul_right R (show h+1 ≤ k by omega)
        nlinarith
      · have step := Nat.mul_le_mul_right R (show k+1 ≤ h by omega)
        nlinarith
  refine ⟨heads,?_⟩
  subst k
  rcases order.2 with a|a
  · omega
  · nlinarith

theorem nonoverlap_injective (H K R C h t k u : Nat)
    (order : nonoverlap H K R C) (hh : h < H) (ht : t < K) (hk : k < H) (hu : u < K)
    (same : h*R+t*C=k*R+u*C) : h=k ∧ t=u := by
  rcases order with a|a
  · exact ordered_injective H K R C h t k u a hh ht hk hu same
  · have result := ordered_injective K H C R t h u k a ht hh hu hk (by omega)
    exact ⟨result.2,result.1⟩

#axiomsClean check_iff
#axiomsClean nonoverlap_injective
end VeriTile.Bench.Repairs.TokenSoftmaxLlama.Checked

namespace VeriTile.Bench.Repairs.TokenSoftmaxLlama.Checked

def AdmittedArguments : Argument → Argument → Argument → Argument → Maximum → Prop
  | .tensor x,.tensor s,.tensor n,.tensor o,.integer k =>
    0 ≤ k ∧ MetadataContract ⟨x,s,n,o,k.toNat⟩
  | _,_,_,_,_ => False

theorem check_arguments_iff (x s n o : Argument) (k : Maximum) :
    checkArguments x s n o k=.ok () ↔ AdmittedArguments x s n o k := by
  cases x <;> cases s <;> cases n <;> cases o <;> cases k <;>
    simp [checkArguments,AdmittedArguments]
  split_ifs <;> simp_all [check_iff]


#axiomsClean check_arguments_iff
end VeriTile.Bench.Repairs.TokenSoftmaxLlama.Checked
