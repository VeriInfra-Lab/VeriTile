import bench.tritonbench_g.lightning_attention.CurrentInterReadback
namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
noncomputable section
set_option maxHeartbeats 6000000
variable {α : Type} [Inhabited α]

theorem fwdCarry_inputs (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (a b : Memory α) (h : Inputs a b) (n : Nat) : fwdCarry M D p pid a n=fwdCarry M D p pid b n := by
  have hk := h "key" (by decide)
  have hv := h "value" (by decide)
  induction n with
  | zero => rfl
  | succ n ih =>
    simp only [fwdCarry,fwdNext,ih]
    unfold fwdK fwdV read
    rw [hk,hv]

theorem fwdOutput_inputs (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (a b : Memory α) (h : Inputs a b) (n : Nat) :
    fwdOutput M D p pid a n (fwdCarry M D p pid a n)=fwdOutput M D p pid b n (fwdCarry M D p pid b n) := by
  rw [fwdCarry_inputs M D p pid a b h n]
  unfold fwdOutput fwdScore fwdQ fwdK fwdV read
  rw [h "query" (by decide),h "key" (by decide),h "value" (by decide)]

theorem intra_values_inputs (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (head block : Nat)
    (a b : Memory α) (h : Inputs a b) :
    intraDQ M D p head block a=intraDQ M D p head block b ∧
    intraDKT M D p head block a=intraDKT M D p head block b ∧
    intraDV M D p head block a=intraDV M D p head block b := by
  unfold intraDQ intraDKT intraDV intraDS intraST intraQ intraK intraV intraG read
  rw [h "query" (by decide),h "key" (by decide),h "value" (by decide),h "incoming" (by decide)]
  exact ⟨rfl,rfl,rfl⟩

theorem queryCarry_inputs (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (head : Nat)
    (a b : Memory α) (h : Inputs a b) (n : Nat) : queryCarry M D p head a n=queryCarry M D p head b n := by
  have hk := h "key" (by decide)
  have hv := h "value" (by decide)
  induction n with
  | zero => rfl
  | succ n ih =>
    simp only [queryCarry,ih]
    unfold queryBlockDelta subVT subK read
    rw [hk,hv]

theorem keyValueCarry_inputs (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (head : Nat)
    (a b : Memory α) (h : Inputs a b) (n : Nat) : keyValueCarry M D p head a n=keyValueCarry M D p head b n := by
  have hq := h "query" (by decide)
  have hg := h "incoming" (by decide)
  induction n with
  | zero => rfl
  | succ n ih =>
    simp only [keyValueCarry,ih]
    unfold keyValueBlockDelta subQT subG read
    rw [hq,hg]
#axiomsClean fwdOutput_inputs
#axiomsClean intra_values_inputs
#axiomsClean queryCarry_inputs
#axiomsClean keyValueCarry_inputs
end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly
