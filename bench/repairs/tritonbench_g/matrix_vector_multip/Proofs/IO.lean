import bench.repairs.tritonbench_g.matrix_vector_multip.IO
import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.SourceSafety
import VeriTile.Triton.Float.ProgramIO

namespace VeriTile.Bench.Repairs.MatrixVectorMultip.IO
open VeriTile.Triton FP.Structural KernelSource
open VeriTile.Bench.TritonBenchG.MatrixVectorMultip
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000

theorem matrix_binding {α β : Type} [Inhabited α] [Inhabited β]
    (A : Algebra β) (N M BN BM pid : Nat)
    (rowShape : Random.ArangeSizeDomain BN) (columnShape : Random.ArangeSizeDomain BM) (encode : α → β) (observe : β → α)
    (vals : Inputs α N M) (s : State β)
    (pins : (io A N M BN BM pid rowShape columnShape encode observe).InputBinding vals s)
    (i j : Nat) (hi : i < N) (hj : j < M) :
    (s.mem "A" (i * M + j)).read .fp32 = encode (matrixValues vals i j) := by
  simpa [ProgramIO.InputBinding, io, matrixValues, hi, hj, mapValue] using
    pins (0 : Fin 2) (⟨i, hi⟩, ⟨j, hj⟩, PUnit.unit) trivial

theorem vector_binding {α β : Type} [Inhabited α] [Inhabited β]
    (A : Algebra β) (N M BN BM pid : Nat)
    (rowShape : Random.ArangeSizeDomain BN) (columnShape : Random.ArangeSizeDomain BM) (encode : α → β) (observe : β → α)
    (vals : Inputs α N M) (s : State β)
    (pins : (io A N M BN BM pid rowShape columnShape encode observe).InputBinding vals s)
    (j : Nat) (hj : j < M) :
    (s.mem "B" j).read .fp32 = encode (vectorValues vals j) := by
  simpa [ProgramIO.InputBinding, io, vectorValues, hj, mapValue] using
    pins (1 : Fin 2) (⟨j, hj⟩, PUnit.unit) trivial

theorem safe {α β : Type} [Inhabited β] (A : Algebra β) (N M BN BM pid : Nat)
    (rowShape : Random.ArangeSizeDomain BN) (columnShape : Random.ArangeSizeDomain BM)
    (width : 0 < BM) (domain : SourceDomain N M BN BM pid) (encode : α → β) (observe : β → α)
    (vals : Inputs α N M) (bounds : RegionName → Nat)
    (covered : (io A N M BN BM pid rowShape columnShape encode observe).InBounds bounds vals) (s : State β) :
    ScheduleSafe bounds ((io A N M BN BM pid rowShape columnShape encode observe).schedule vals) s := by
  let entry := (invocation A N M BN BM pid rowShape columnShape).initial s
  have actual : FP.Structural.TraceSafe A bounds (Kernels.source N M BN BM rowShape columnShape) entry := by
    apply SourceSafety.source_safe A N M BN BM rowShape columnShape entry
      (by simpa [entry,invocation,Invocation.initial,bindRegisters] using domain) bounds
    · intro i j hi hj
      simpa [parameters] using covered.2.1 (0 : Fin 2) (⟨i, hi⟩, ⟨j, hj⟩, PUnit.unit) trivial
    · intro j hj
      simpa [parameters] using covered.2.1 (1 : Fin 2) (⟨j, hj⟩, PUnit.unit) trivial
    · intro i active
      have act : pid * BN + i.val < N := by
        simpa [parameters, WholeSource.row, SourcePrelude.row, entry, invocation, Invocation.initial,
          bindRegisters] using active
      simpa [parameters, WholeSource.row, SourcePrelude.row, entry, invocation, Invocation.initial,
        bindRegisters] using covered.2.2.1 (0 : Fin 1) (i, PUnit.unit) act
  change (invocation A N M BN BM pid rowShape columnShape).Safe bounds s ∧ _
  refine ⟨actual, ?_⟩
  intro t _
  trivial

/-- Task proof assembly fixes the actual source runner. `result` is a proven
execution/output/frame lemma, never a field or assumption of the public IO. -/
theorem implements {α β : Type} [Inhabited α] [Inhabited β]
    (A : Algebra β) (N M BN BM pid : Nat)
    (rowShape : Random.ArangeSizeDomain BN) (columnShape : Random.ArangeSizeDomain BM) (width : 0 < BM) (domain : SourceDomain N M BN BM pid)
    (encode : α → β) (observe : β → α)
    (f : (Nat → Nat → α) → (Nat → α) → Fin BN → α)
    (result : ∀ (s : State β) (matrix : Nat → Nat → α) (vector : Nat → α),
      (∀ i j, i < N → j < M →
        (s.mem "A" (i * M + j)).read .fp32 = encode (matrix i j)) →
      (∀ j, j < M → (s.mem "B" j).read .fp32 = encode (vector j)) →
      s.pids 0 = pid →
      ∃ t, FP.Structural.exec A (Kernels.source N M BN BM rowShape columnShape) s = some t ∧
        (∀ i : Fin BN, pid * BN + i.val < N →
          mapCell observe (t.mem "C" (pid * BN + i.val)) = .mk .fp32 (f matrix vector i)) ∧
        (∀ r o, Outside N BN pid r o → t.mem r o = s.mem r o)) :
    (io A N M BN BM pid rowShape columnShape encode observe).Implements
      (fun vals => outputValues (f (matrixValues vals) (vectorValues vals))) := by
  apply ProgramIO.Implements.intro
  · intro vals bounds covered initial _
    exact safe A N M BN BM pid rowShape columnShape width domain encode observe vals bounds covered initial
  · intro vals initial pins
    let entry := (invocation A N M BN BM pid rowShape columnShape).initial initial
    obtain ⟨final, ran, outputs, frame⟩ := result entry (matrixValues vals) (vectorValues vals)
      (matrix_binding A N M BN BM pid rowShape columnShape encode observe vals initial pins)
      (vector_binding A N M BN BM pid rowShape columnShape encode observe vals initial pins) rfl
    refine ⟨final, ?_, ?_, ?_⟩
    · change ((invocation A N M BN BM pid rowShape columnShape).run initial).bind (fun s => some s) = some final
      change (FP.Structural.exec A (Kernels.source N M BN BM rowShape columnShape) entry).bind (fun s => some s) = some final
      rw [ran]
      rfl
    · intro o
      refine Fin.cases ?_ (fun i => i.elim0) o
      intro i active
      exact outputs i.1 active
    · intro r o outside
      apply frame r o
      by_cases region : r = "C"
      · right
        intro i active
        exact (outside.1 (0 : Fin 1) (i, PUnit.unit) active).resolve_left (not_not.mpr region)
      · exact Or.inl region

end
end VeriTile.Bench.Repairs.MatrixVectorMultip.IO
