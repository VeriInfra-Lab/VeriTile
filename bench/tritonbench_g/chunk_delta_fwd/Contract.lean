import bench.tritonbench_g.chunk_delta_fwd.Proofs.Contract
import bench.tritonbench_g.chunk_delta_fwd.Kernels

namespace VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot
open VeriTile Triton ChunkDeltaFwd
open VeriTile.Bench.Repairs.ChunkGlaFwd.Host (nextPowerOfTwo nextPowerOfTwo_ge)
abbrev Request := Checked.Request
abbrev Tensor := Checked.Tensor

/-- The wrapper skips launch exactly when B*H*V=0. A nonempty grid still
typechecks the two dot bodies when T=0. No restriction on FP32 payloads. -/
def StaticDot (r : Request) : Prop :=
  r.B*r.H*r.V=0 ∨ (16≤r.dims.BK ∧ 16≤r.dims.BC)
instance (r : Request) : Decidable (StaticDot r) := by
  unfold StaticDot; infer_instance

def InputContract (r : Request) : Prop := r.Admitted ∧ StaticDot r
instance (r : Request) : Decidable (InputContract r) := by
  unfold InputContract; infer_instance

/-- Original guards plus the candidate's sole new assertion. -/
def check (r : Request) : Bool := r.check && decide (StaticDot r)

theorem check_iff (r : Request) : check r=Bool.true ↔ InputContract r := by
  simp [check,InputContract,Checked.Request.check_iff]

theorem base_checked (r : Request) (h : InputContract r) : r.check=Bool.true :=
  r.check_iff.mpr h.1

theorem live_dot_minimum (r : Request) (h : InputContract r) (live : r.B*r.H*r.V≠0) :
    16≤r.dims.BK ∧ 16≤r.dims.BC := h.2.resolve_left live

theorem bk_minimum_iff (K : Nat) : 16≤nextPowerOfTwo K ↔ 9≤K := by
  by_cases large : 16≤K
  · constructor
    · omega
    · intro _; exact large.trans (nextPowerOfTwo_ge K)
  · have small : K≤15 := by omega
    interval_cases K <;> decide

theorem bc_minimum_iff (r : Request) : 16≤r.dims.BC ↔ 16≤r.BT := by
  have bv := Host.bv_legal r.dims
  change 16≤min r.BT r.dims.BV ↔ 16≤r.BT
  rcases bv with bv|bv|bv <;> rw [bv] <;> omega

/-- Exact dimension form: positive source K=9 is admitted. T does not appear
because frontend typechecking precedes elimination of the zero-trip loop. -/
theorem static_dot_iff (r : Request) :
    StaticDot r ↔ r.B*r.H*r.V=0 ∨ (9≤r.K ∧ 16≤r.BT) := by
  unfold StaticDot
  rw [bc_minimum_iff]
  change (_ ∨ (16≤nextPowerOfTwo r.K ∧ _)) ↔ _
  rw [bk_minimum_iff]

theorem empty_admitted (r : Request) (base : r.Admitted) (empty : r.B*r.H*r.V=0) :
    InputContract r := ⟨base,Or.inl empty⟩

#axiomsClean check_iff
#axiomsClean live_dot_minimum
#axiomsClean static_dot_iff
#specNonCircular InputContract avoiding [FP.Structural.run,runGrid,gridMemory,states]
end VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot
