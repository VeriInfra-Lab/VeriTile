import bench.tritonbench_g.relu_triton_kernel.Proofs.FiniteAddress

/-! Byte-scaled finite-address arithmetic, separately from memory semantics.
The explicit mathematical interval is not evidence of an allocated object,
provenance, alignment, byte loading, endianness or native pointer lowering.
Input scaling reflects the nine finite-word widths; output scaling is fp32.
Only active addresses require a no-wrap bound. -/
namespace VeriTile.Bench.TritonBenchG.ReluTritonKernel.FiniteAddress
open VeriTile.Triton GatedGrid IntegerWords

def elementBytes (e : Encoding) : Nat := e.bits / 8

theorem element_bytes_positive (e : Encoding) : 0 < elementBytes e := by
  cases e with
  | signed w => cases w <;> decide
  | unsigned w => cases w <;> decide
  | boolean => decide

theorem element_bytes_width (e : Encoding) : elementBytes e * 8 = e.bits := by
  cases e with
  | signed w => cases w <;> decide
  | unsigned w => cases w <;> decide
  | boolean => decide

/-- A representable base and an exclusive mathematical endpoint. For N=0
this certifies only a base value, not a native empty allocation or launch. -/
def ByteDomain (base bytes N : Nat) : Prop :=
  base < 2^64 ∧ base + N * bytes ≤ 2^64

def pointer64 (base bytes pid lane : Nat) : BitVec 64 :=
  BitVec.ofNat 64 base + (offset32 pid lane).signExtend 64 * BitVec.ofNat 64 bytes

def bytePointer64 (base bytes pid lane byte : Nat) : BitVec 64 :=
  pointer64 base bytes pid lane + BitVec.ofNat 64 byte

theorem relative_byte_bound (bytes N n byte : Nat)
    (active : n < N) (inElement : byte < bytes) :
    n * bytes + byte < N * bytes := by
  calc
    n * bytes + byte < n * bytes + bytes := Nat.add_lt_add_left inElement _
    _ = (n + 1) * bytes := by simp only [Nat.add_mul, Nat.one_mul]
    _ ≤ N * bytes := Nat.mul_le_mul_right bytes (by omega)

/-- The complete footprint, including the last byte of each element, stays
inside [base, base+N*bytes) and the unsigned 64-bit address range. -/
theorem byte_span (base bytes N n byte : Nat)
    (domain : ByteDomain base bytes N) (active : n < N) (inElement : byte < bytes) :
    base ≤ base + n * bytes + byte ∧
    base + n * bytes + byte < base + N * bytes ∧
    base + n * bytes + byte < 2^64 := by
  have h := relative_byte_bound bytes N n byte active inElement
  unfold ByteDomain at domain
  omega

theorem pointer_exact (base bytes N pid lane : Nat)
    (indexDomain : IndexDomain N) (domain : ByteDomain base bytes N)
    (positive : 0 < bytes) (launched : pid < gridCount N blockSize)
    (inBlock : lane < blockSize) (active : offset pid lane < N) :
    (pointer64 base bytes pid lane).toNat = base + offset pid lane * bytes := by
  have bound := (byte_span base bytes N (offset pid lane) 0 domain active positive).2.2
  simp only [Nat.add_zero] at bound
  unfold pointer64
  rw [offset_sign_extend N pid lane indexDomain launched inBlock,
    BitVec.ofNat_mul_ofNat, BitVec.ofNat_add_ofNat, BitVec.toNat_ofNat,
    Nat.mod_eq_of_lt bound]

theorem byte_pointer_exact (base bytes N pid lane byte : Nat)
    (indexDomain : IndexDomain N) (domain : ByteDomain base bytes N)
    (launched : pid < gridCount N blockSize) (inBlock : lane < blockSize)
    (active : offset pid lane < N) (inElement : byte < bytes) :
    (bytePointer64 base bytes pid lane byte).toNat = base + offset pid lane * bytes + byte := by
  have bound := (byte_span base bytes N (offset pid lane) byte domain active inElement).2.2
  unfold bytePointer64 pointer64
  rw [offset_sign_extend N pid lane indexDomain launched inBlock,
    BitVec.ofNat_mul_ofNat, BitVec.ofNat_add_ofNat, BitVec.ofNat_add_ofNat,
    BitVec.toNat_ofNat, Nat.mod_eq_of_lt bound]

/-- Byte coordinates in one positive-width contiguous view cannot alias.
This is an arithmetic statement, with no storage/view representation claim. -/
theorem byte_coordinates_injective (base bytes n m i j : Nat)
    (hi : i < bytes) (hj : j < bytes)
    (same : base + n * bytes + i = base + m * bytes + j) : n = m ∧ i = j := by
  have hn : n = m := by
    rcases lt_trichotomy n m with h | h | h
    · have hb := relative_byte_bound bytes m n i h hi
      omega
    · exact h
    · have hb := relative_byte_bound bytes n m j h hj
      omega
  subst m
  exact ⟨rfl, by omega⟩

/-- Numeric separation implies disjoint byte footprints. Fresh native output
allocation is a separate open obligation and is not inferred here. -/
theorem separate_byte_ranges (a wa na b wb nb i j u v : Nat)
    (separate : a + na * wa ≤ b ∨ b + nb * wb ≤ a)
    (hi : i < na) (hj : j < nb) (hu : u < wa) (hv : v < wb) :
    a + i * wa + u ≠ b + j * wb + v := by
  have ha := relative_byte_bound wa na i u hi hu
  have hb := relative_byte_bound wb nb j v hj hv
  rcases separate with h | h <;> omega

#specNonCircular pointer64 avoiding [TypedValue.source, runGrid, FP.Structural.exec]
#axiomsClean element_bytes_positive
#axiomsClean element_bytes_width
#axiomsClean relative_byte_bound
#axiomsClean byte_span
#axiomsClean pointer_exact
#axiomsClean byte_pointer_exact
#axiomsClean byte_coordinates_injective
#axiomsClean separate_byte_ranges
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ReluTritonKernel.FiniteAddress
