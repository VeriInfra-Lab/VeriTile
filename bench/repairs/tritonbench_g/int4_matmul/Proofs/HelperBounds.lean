import bench.repairs.tritonbench_g.int4_matmul.Proofs.Helpers
namespace VeriTile.Bench.Repairs.Int4Matmul.HelperBounds
open Helpers
/-- The transposed contiguous copy, group reshape and each lane of both packing loops. -/
theorem coordinates (s : Shape) (hs : s.Valid) (g n i : Nat)
    (hg : g<s.k/s.group) (hn : n<s.n) (hi : i<s.group) :
    g*s.group+i<s.k ∧
    (g*s.group+i)*s.n+n<s.k*s.n ∧
    n*s.k+(g*s.group+i)<s.n*s.k ∧
    (n*(s.k/s.group)+g)*s.group+i<s.n*s.k := by
  have hk : g*s.group+i<s.k := by
    simpa [Nat.div_mul_cancel hs.groups] using rowMajor_bound _ _ _ _ hg hi
  refine ⟨hk,rowMajor_bound _ _ _ _ hk hn,rowMajor_bound _ _ _ _ hn hk,?_⟩
  have he : (n*(s.k/s.group)+g)*s.group+i=n*s.k+(g*s.group+i) := by
    rw [Nat.add_mul, Nat.mul_assoc,Nat.div_mul_cancel hs.groups]
    omega
  rw [he];exact rowMajor_bound _ _ _ _ hn hk

theorem packed_coordinates (s : Shape) (hs : s.Valid) (p n : Nat) (hp : p<s.k/8)
    (hn : n<s.n) (i : Fin 8) :
    p*8+i.val<s.k ∧ n*(s.k/8)+p<s.n*(s.k/8) ∧ p*s.n+n<(s.k/8)*s.n := by
  exact ⟨by simpa [Nat.div_mul_cancel hs.k8] using rowMajor_bound _ _ _ _ hp i.isLt,
    rowMajor_bound _ _ _ _ hn hp,rowMajor_bound _ _ _ _ hp hn⟩

theorem zero_coordinates (s : Shape) (hs : s.Valid) (g p : Nat) (hg : g<s.k/s.group)
    (hp : p<s.n/8) (i : Fin 8) :
    p*8+i.val<s.n ∧ p*(s.k/s.group)+g<(s.n/8)*(s.k/s.group) ∧
    g*(s.n/8)+p<(s.k/s.group)*(s.n/8) := by
  exact ⟨by simpa [Nat.div_mul_cancel hs.n8] using rowMajor_bound _ _ _ _ hp i.isLt,
    rowMajor_bound _ _ _ _ hp hg,rowMajor_bound _ _ _ _ hg hp⟩

theorem half_byte_bound (s : Shape) (guard : s.k*s.n<2^31) (k n : Nat)
    (hk : k<s.k) (hn : n<s.n) (i : Fin 2) : 2*(n*s.k+k)+i.val<2^63 := by
  have h := unpack_address s k n hk hn
  have hg : s.n*s.k<2^31 := by simpa [Nat.mul_comm] using guard
  omega
#axiomsClean coordinates
#axiomsClean packed_coordinates
#axiomsClean zero_coordinates
#axiomsClean half_byte_bound
end VeriTile.Bench.Repairs.Int4Matmul.HelperBounds
