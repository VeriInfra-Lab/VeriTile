import bench.repairs.tritonbench_g.quant_transpose_kernel.Proofs.KernelModels

namespace VeriTile.Bench.Repairs.QuantTransposeKernel.KernelWhole
open VeriTile.Triton FP.Structural KernelSource KernelResult KernelModels
noncomputable section

/-- Initial typed Tensor values and the scalar actually passed to the kernel.
The host computation producing the inverse is outside this kernel theorem. -/
structure Reads {α : Type} (p : Parameters) (values : Nat → Nat → α)
    (inverse : α) (s : State α) : Prop where
  input : ∀ r c, r < p.M → c < p.N →
    s.mem ⟨p.x⟩ (r * p.sam + c * p.san) = .mk .real (values r c)
  inverse : s.mem ⟨p.scale⟩ 0 = .mk .real inverse

/-- INT8 output observes the low eight bits at the actual pointer store.
The interpreter stores the signed INT8 value; observing it as a byte
preserves the low eight bits of the signed64 rounding result. -/
def byte {α : Type} [Inhabited α] (s : State α) (out : String) (o : Nat) : BitVec 8 :=
  BitVec.ofInt 8 ((s.mem ⟨out⟩ o).read .int)

def Result {α : Type} [Inhabited α] (p : Parameters)
    (expected : Nat → Nat → BitVec 8) (initial final : State α) : Prop :=
  (∀ r c, r < p.M → c < p.N → byte final p.output (c * p.M + r) = expected r c) ∧
  (∀ r o, r ≠ ⟨p.output⟩ ∨ p.M * p.N ≤ o → final.mem r o = initial.mem r o)

theorem kernel_correct {α : Type} [Inhabited α] (M : Algebra α)
    (round : α → BitVec 64) (p : Parameters) (sep : KernelGrid.Separate p)
    (values : Nat → Nat → α) (inverse : α) (s : State α) (reads : Reads p values inverse s) :
    ∃ final, KernelGrid.run M round p Bool.false
      (List.range (QuantTransposeFinite.gridM p.M * QuantTransposeFinite.gridN p.N)) s = some final ∧
      Result p (fun r c => quantized M round (values r c) inverse) s final := by
  obtain ⟨final, he, ho, hf⟩ := KernelGrid.correctness M round p sep s
  refine ⟨final, he, ?_, hf⟩
  intro r c hr hc
  have hb := QuantTransposeFinite.active_encode_bound p.M p.N r c hr hc
  have hd := QuantTransposeFinite.encode_decode p.M r c hr
  unfold byte
  rw [ho _ hb]
  unfold KernelGrid.reference
  rw [hd.1, hd.2, reads.input r c hr hc, reads.inverse]
  simp only [Cell.read_same, BitVec.ofInt_toInt, quantized]

theorem real_correct (rest : Algebra ℝ) (p : Parameters) (sep : KernelGrid.Separate p)
    (values : Nat → Nat → ℝ) (inverse : ℝ) (s : State ℝ) (reads : Reads p values inverse s) :
    ∃ final, KernelGrid.run (realAlgebra rest) realRound p Bool.false
      (List.range (QuantTransposeFinite.gridM p.M * QuantTransposeFinite.gridN p.N)) s = some final ∧
      Result p (fun r c => realByte (values r c) inverse) s final := by
  simpa only [real_formula] using kernel_correct (realAlgebra rest) realRound p sep values inverse s reads

theorem native_correct (multiply : Word → Word → Word) (rest : Algebra Word)
    (p : Parameters) (sep : KernelGrid.Separate p) (values : Nat → Nat → Word)
    (inverse : Word) (s : State Word) (reads : Reads p values inverse s) :
    ∃ final, KernelGrid.run (nativeAlgebra multiply rest) nativeRound p Bool.false
      (List.range (QuantTransposeFinite.gridM p.M * QuantTransposeFinite.gridN p.N)) s = some final ∧
      Result p (fun r c => nativeByte multiply (values r c) inverse) s final := by
  simpa only [native_formula] using
    kernel_correct (nativeAlgebra multiply rest) nativeRound p sep values inverse s reads

theorem ownership (p : Parameters) (hn : 0 < p.N) (pid pid' : Nat)
    (hp : pid < QuantTransposeFinite.gridM p.M * QuantTransposeFinite.gridN p.N)
    (hp' : pid' < QuantTransposeFinite.gridM p.M * QuantTransposeFinite.gridN p.N)
    (i j : Lane) (hi : active p pid i) (hj : active p pid' j)
    (he : outputOffset p pid i = outputOffset p pid' j) : pid = pid' ∧ i = j := by
  have h := QuantTransposeFinite.output_ownership p .first pid (flat i) pid' (flat j) hn hp hp'
    (flat_bound i) (flat_bound j) ((flat_active p pid i).2 hi) ((flat_active p pid' j).2 hj)
    (by simpa only [flat_output] using he)
  exact ⟨h.1, flat_injective h.2⟩

#axiomsClean kernel_correct
#axiomsClean real_correct
#axiomsClean native_correct
#axiomsClean ownership
#specNonCircular Reads avoiding [KernelSource.kernel, KernelExecution.exec, KernelGrid.run]
#specNonCircular Result avoiding [KernelSource.kernel, KernelExecution.exec, KernelGrid.run]
end
end VeriTile.Bench.Repairs.QuantTransposeKernel.KernelWhole
