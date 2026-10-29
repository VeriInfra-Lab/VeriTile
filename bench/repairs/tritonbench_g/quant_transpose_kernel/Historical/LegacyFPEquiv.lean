import bench.repairs.tritonbench_g.quant_transpose_kernel.Proofs.Contract
import bench.repairs.tritonbench_g.quant_transpose_kernel.Proofs.Indices
import bench.tritonbench_g.quant_transpose_kernel.FPEquiv

namespace VeriTile.Bench.Repairs.QuantTransposeKernel.FPEquiv
open QuantTransposeFinite
noncomputable section

def runChecked (b : Numerics) (p : Metadata) (alloc : Allocations) (m : Memory)
    (x : String) (trials : List Config) (last : Config) :
    Except InputError (Except Diagnostic Returned) := do
  checkInputs p
  return execute b (fun _ => none) .fp32 p.input alloc m x trials last

/-- Full scalar-precision logical host, returned int8 transpose and FP32
absmax, all completed trials, exact frame and physical byte representation.
Only the local scalar definedness/numerics ports remain conditional. -/
specification checked_finite (b : Numerics) (p : Metadata)
    (alloc : Allocations) (m : Memory) (x : String)
    (logical : List (Word .fp32)) (xs : Nat → Word .fp32) (physical : PhysicalLayout)
    (admitted : InputContract p) (D : Domain .fp32 p.input alloc m x logical xs)
    (ordinary : OrdinaryPhysicalBinding p.input alloc m x physical)
    (pointer : physical.inputPointer.toNat=p.dataPointer)
    (maximum : Word .fp32) (hm : produced b .fp32 logical = some maximum)
    (defined : Defined b .fp32 (parameters p.input alloc x) xs (b.reciprocal .fp32 maximum))
    (trials : List Config) (last : Config) :
    ∃ ret, runChecked b p alloc m x trials last = .ok (.ok ret) ∧
      Result b .fp32 p.input alloc m xs maximum ret ∧
      PhysicalResult b .fp32 p.input alloc physical xs maximum ret
        (externalBytes .fp32 p.input alloc physical ret) ∧
      ArithmeticRefines p.input .fp32 physical := by
  let P := ordinary.admitted admitted pointer
  obtain ⟨cuda,rank,stride⟩ := admitted_metadata p admitted
  obtain ⟨ret,he,hr⟩ := quant_transpose_finite_host_correctness b .fp32 p.input alloc m x
    logical xs maximum D hm rank stride cuda defined trials last
  exact ⟨ret,by simp [runChecked,(checkInputs_iff p).2 admitted,he],hr,
    quant_transpose_external_result b .fp32 p.input alloc m x physical xs maximum ret D.matrixCount P hr,
    arithmetic_refines p.input admitted.indices .fp32 alloc m x logical xs physical D P⟩

/-- Native finite outcomes retain undefined numerical observations explicitly.
Every successful output now has the logical transpose formula without an
assumed ArithmeticRefines premise: the checked source envelope proves it. -/
specification checked_native_outcomes (b : Numerics) (p : Metadata)
    (alloc : Allocations) (m : Memory) (x : String)
    (logical : List (Word .fp32)) (xs : Nat → Word .fp32) (physical : PhysicalLayout)
    (admitted : InputContract p) (D : Domain .fp32 p.input alloc m x logical xs)
    (ordinary : OrdinaryPhysicalBinding p.input alloc m x physical)
    (pointer : physical.inputPointer.toNat=p.dataPointer)
    (trials : List Config) (last : Config) :
    ∃ maximum, produced b .fp32 logical = some maximum ∧
      ((∃ issue, nativeTrialsPlan b .fp32 p.input alloc physical
          (hostPhysical .fp32 physical maximum (b.reciprocal .fp32 maximum)) (trials++[last]) = .error issue ∧
        executeNativeAPI b (fun _ => none) .fp32 p.input alloc m x physical trials last =
          .attempted .fp32 (.error (.observation issue))) ∨
       ∃ ret, executeNativeAPI b (fun _ => none) .fp32 p.input alloc m x physical trials last =
          .attempted .fp32 (.ok ret) ∧
        NativeResult b .fp32 p.input alloc physical maximum trials last ret ∧
        (∀ i, i < p.input.numel → ret.memory (physical.output.base+i) =
          result b .fp32 (xs i) (b.reciprocal .fp32 maximum))) := by
  let P := ordinary.admitted admitted pointer
  obtain ⟨cuda,rank,stride⟩ := admitted_metadata p admitted
  have nonempty : 0 < p.input.numel := by
    rw [D.matrixCount]
    exact Nat.mul_pos admitted.indices.rows admitted.indices.columns
  obtain ⟨maximum,hm,outcome⟩ := native_host_all_outcomes b .fp32 p.input alloc m x logical xs
    physical D P nonempty rank stride cuda trials last
  refine ⟨maximum,hm,?_⟩
  rcases outcome with rejected | observed | success
  · exact (rejected.1 admitted.indices.grid).elim
  · exact Or.inl observed.2
  · obtain ⟨ret,he,hr⟩ := success.2
    exact Or.inr ⟨ret,he,hr,native_logical_transpose_refinement b .fp32 p.input alloc m x logical xs physical D P
      (arithmetic_refines p.input admitted.indices .fp32 alloc m x logical xs physical D P)
      maximum trials last ret hr⟩

#axiomsClean checked_finite
#axiomsClean checked_native_outcomes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.QuantTransposeKernel.FPEquiv
