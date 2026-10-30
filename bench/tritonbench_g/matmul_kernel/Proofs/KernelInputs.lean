import bench.tritonbench_g.matmul_kernel.Proofs.KernelReal
import bench.tritonbench_g.matmul_kernel.Proofs.KernelNative

/-! Logical-array binding for the existing complete source-grid proofs.
Only active input matrix coordinates are pinned; no output value or execution
fact is supplied by the binding. -/
namespace VeriTile.Bench.Repairs.MatmulKernel.KernelInputs
open VeriTile.Triton FP.Structural KernelSource KernelScalar
noncomputable section
set_option maxHeartbeats 5000000
local instance : Inhabited KernelNative.Word := ⟨FP.Value.zero .fp32⟩

/-- The ordered scalar reference observes only the selected row/column and
in-range K operands. Padding never creates an input requirement. -/
theorem reference_congr {α : Type} [Inhabited α] (M : Algebra α) (G : LaneDot α)
    (p : Config) (a₀ b₀ a b : Nat → Nat → α) (r n : Nat)
    (ha : ∀ k, k<p.K → a₀ r k=a r k)
    (hb : ∀ k, k<p.K → b₀ k n=b k n) :
    reference M G p a₀ b₀ r n=reference M G p a b r n := by
  have update :
      (fun q acc => G (r%p.BM) (n%p.BN) p.BK
        (fun e => M.fp32Load (if q*p.BK+e.val<p.K then a₀ r (q*p.BK+e.val) else M.literal (some .fp32) .real 0))
        (fun e => M.fp32Load (if q*p.BK+e.val<p.K then b₀ (q*p.BK+e.val) n else M.literal (some .fp32) .real 0)) acc) =
      (fun q acc => G (r%p.BM) (n%p.BN) p.BK
        (fun e => M.fp32Load (if q*p.BK+e.val<p.K then a r (q*p.BK+e.val) else M.literal (some .fp32) .real 0))
        (fun e => M.fp32Load (if q*p.BK+e.val<p.K then b (q*p.BK+e.val) n else M.literal (some .fp32) .real 0)) acc) := by
    funext q acc
    congr 1
    · funext e
      congr 1
      split_ifs with hk
      · exact ha _ hk
      · rfl
    · funext e
      congr 1
      split_ifs with hk
      · exact hb _ hk
      · rfl
  simp only [reference, update]

/-- Complete source grid with independently supplied logical Real matrices. -/
theorem real (p : Config) (hc : p.Valid) (s : State ℝ)
    (a b : Nat → Nat → ℝ)
    (ha : ∀ r k, r<p.M → k<p.K → read (s.mem "A" (p.A.addr r k))=a r k)
    (hb : ∀ k n, k<p.K → n<p.N → read (s.mem "B" (p.B.addr k n))=b k n) :
    ∃ t, KernelSource.launch KernelReal.scalar (dotAcc KernelReal.dot) p s=some t ∧
      (∀ w, p.Active w → t.mem "C" (p.C.addr w.1 w.2)=stored
        (KernelReal.target p.K a b w.1 w.2)) ∧
      (∀ (region : RegionName) offset,
        (region≠"C" ∨ ∀ w, p.Active w → offset≠p.C.addr w.1 w.2) →
        t.mem region offset=s.mem region offset) := by
  obtain ⟨t, ran, output, frame⟩ := KernelReal.launch_correct p hc s
  refine ⟨t, ran, ?_, frame⟩
  intro w hw
  rw [output w hw, ←KernelReal.reference_target p hc.bk,
    reference_congr KernelReal.scalar KernelReal.dot p _ _ a b w.1 w.2
      (fun k hk => ha w.1 k hw.1 hk) (fun k hk => hb k w.2 hk hw.2),
    KernelReal.reference_target p hc.bk]

/-- The same binding for native FP32 words retains the ordered K8 target,
both final conversions, and the complete exterior tagged-cell frame. -/
theorem native (instructions : KernelNative.Instructions) (p : Config) (hc : p.Valid)
    (s : State KernelNative.Word) (a b : Nat → Nat → KernelNative.Word)
    (ha : ∀ r k, r<p.M → k<p.K → read (s.mem "A" (p.A.addr r k))=a r k)
    (hb : ∀ k n, k<p.K → n<p.N → read (s.mem "B" (p.B.addr k n))=b k n) :
    ∃ t, KernelSource.launch KernelNative.scalar
        (dotAcc (KernelNative.dot instructions p.opcode)) p s=some t ∧
      (∀ w, p.Active w → t.mem "C" (p.C.addr w.1 w.2)=stored
        (KernelNative.target instructions p a b w.1 w.2)) ∧
      (∀ (region : RegionName) offset,
        (region≠"C" ∨ ∀ w, p.Active w → offset≠p.C.addr w.1 w.2) →
        t.mem region offset=s.mem region offset) := by
  obtain ⟨t, ran, output, frame⟩ := KernelNative.launch_correct instructions p hc s
  refine ⟨t, ran, ?_, frame⟩
  intro w hw
  rw [output w hw, ←KernelNative.reference_target instructions p,
    reference_congr KernelNative.scalar (KernelNative.dot instructions p.opcode)
      p _ _ a b w.1 w.2
      (fun k hk => ha w.1 k hw.1 hk) (fun k hk => hb k w.2 hk hw.2),
    KernelNative.reference_target instructions p]

#axiomsClean reference_congr
#axiomsClean real
#axiomsClean native
end
end VeriTile.Bench.Repairs.MatmulKernel.KernelInputs
