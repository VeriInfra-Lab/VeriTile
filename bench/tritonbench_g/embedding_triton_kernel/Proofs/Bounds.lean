import bench.tritonbench_g.embedding_triton_kernel.Contract
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.EmbeddingTritonKernel.Bounds
open Request
open private Nat.nextPowerOfTwo.go from Init.Data.Nat.Power2.Basic

theorem sizes (c : Request) (h : InputContract c) :
    c.ids.numel = c.n ∧ c.weight.numel = c.rows*c.hidden ∧
    c.output.numel = c.n*c.hidden := by
  obtain ⟨_,_,hs,_⟩ := h
  rcases hs with ⟨hi,hw,ho⟩
  simp [VeriTile.Torch.TensorMetadata.numel,hi,hw,ho]

/-- Host normalization accepts arbitrary Python integer endpoints. -/
theorem interval (c : Request) (id : Int) (lo : -(2^31) ≤ id) (hi : id < 2^31) :
    (c.firstId ≤ id ∧ id < c.lastId) ↔ (c.first ≤ id ∧ id < c.last) := by
  simp only [firstId,lastId,intersects,upper]
  split_ifs <;> omega

theorem first_eq (c : Request) (id : Int) (lo : -(2^31) ≤ id) (hi : id < 2^31)
    (valid : c.first ≤ id ∧ id < c.last) : c.firstId = c.first := by
  have inter : c.intersects := by unfold intersects upper; omega
  exact if_pos inter

theorem selected_row (c : Request) (h : InputContract c) (hp : 0<c.hidden)
    (id : Int) (lo : -(2^31) ≤ id) (hi : id < 2^31)
    (valid : c.first ≤ id ∧ id < c.last) : (id-c.first).toNat<c.rows := by
  have inter : c.intersects := by unfold intersects upper; omega
  have hv := h.2.2.2.2.2.2.2
  have bound : c.upper-c.first ≤ (c.rows : Int) := by
    rcases hv with hz|hn|hb
    · omega
    · exact False.elim (hn inter)
    · exact hb
  have up : id<c.upper := by unfold upper; omega
  omega

theorem flat_bound (rows cols r d : Nat) (hr : r<rows) (hd : d<cols) :
    r*cols+d<rows*cols := by
  have hm := Nat.mul_le_mul_right cols (show r+1≤rows by omega)
  simp only [Nat.add_mul, Nat.one_mul] at hm
  omega

/-- The actual next-power-of-two algorithm covers the hidden columns. -/
theorem nextPower_covers (n : Nat) : n≤n.nextPowerOfTwo := by
  unfold Nat.nextPowerOfTwo
  exact go 1 (by decide)
where
  go (p : Nat) (hp : 0<p) : n≤Nat.nextPowerOfTwo.go n p hp := by
    rw [Nat.nextPowerOfTwo.go]
    split
    · exact go (p*2) (Nat.mul_pos hp (by decide))
    · omega
  termination_by n-p
  decreasing_by exact Nat.nextPowerOfTwo_dec hp (by assumption)

/-- The wrapper's block bound implies the exact rounded width bound. -/
theorem nextPower_bound (n : Nat) (hn : n≤2^20) : n.nextPowerOfTwo≤2^20 := by
  unfold Nat.nextPowerOfTwo
  exact go 0 (by omega)
where
  go (k : Nat) (hk : k≤20) :
      Nat.nextPowerOfTwo.go n (2^k) (Nat.two_pow_pos k)≤2^20 := by
    rw [Nat.nextPowerOfTwo.go]
    split
    · have less : k<20 := by
        by_contra h
        have : k=20 := by omega
        subst k
        omega
      simpa [Nat.pow_succ] using go (k+1) (by omega)
    · exact Nat.pow_le_pow_right (by omega) hk
  termination_by 20-k

/-- Python's tensor assertions and signed int32 ID storage supply the full
per-program memory contract, including padded last programs. No kernel-output
agreement is a premise. The mathematical payload arrays remain arbitrary. -/
theorem kernel_contract (c : Request) (h : InputContract c) (pid : Nat)
    (ids : Nat → Int) (signed : ∀ r,r < c.n → -(2^31) ≤ ids r ∧ ids r < 2^31) :
    KernelContract c.parameters pid ids c.regionBounds := by
  obtain ⟨hn,hw,ho⟩ := sizes c h
  refine ⟨by change ("ids" : VeriTile.Triton.RegionName) ≠ "out"; decide,
    by change ("weight" : VeriTile.Triton.RegionName) ≠ "out"; decide,
    nextPower_covers c.hidden,le_rfl,signed,?_,?_,?_⟩
  · intro a ha hr
    simpa [parameters,regionBounds,hn] using hr
  · intro a ha hr lo hi d hd
    have hs := signed (pid*64+a) hr
    have valid := (interval c (ids (pid*64+a)) hs.1 hs.2).mp ⟨lo,hi⟩
    have row := selected_row c h (by change d < c.hidden at hd; omega)
      (ids (pid*64+a)) hs.1 hs.2 valid
    have first := first_eq c (ids (pid*64+a)) hs.1 hs.2 valid
    simpa [parameters,regionBounds,hw,first] using flat_bound c.rows c.hidden
      (ids (pid*64+a)-c.first).toNat d row hd
  · intro a ha hr d hd
    simpa [parameters,regionBounds,ho] using flat_bound c.n c.hidden (pid*64+a) d hr hd

#axiomsClean kernel_contract
#axiomsClean interval
#axiomsClean selected_row
#axiomsClean nextPower_covers
#axiomsClean nextPower_bound
end VeriTile.Bench.Repairs.EmbeddingTritonKernel.Bounds
