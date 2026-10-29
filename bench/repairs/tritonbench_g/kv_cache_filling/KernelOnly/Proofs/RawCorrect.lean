import bench.repairs.tritonbench_g.kv_cache_filling.Proofs.SharedReadback

namespace VeriTile.Bench.Repairs.KvCacheFilling.RawReal
open VeriTile.Triton FP.Structural
noncomputable section

/-- Both actual shared kernels execute over arbitrary mathematical Real
payloads. Every active row, scale/zero value, packed byte and complementary
logical cell is covered. Inputs are not constrained by a finite byte codec.
The only observation premises are the two input INT64 job-array reads. -/
theorem whole_shared_real (r : Request) (d : Domain r) (initial : Job → State ℝ)
    (src : ∀ j ∈ jobs r,
      ((initial j).mem "src" ((initial j).pids 0)).read .int = (sourceRow r j : Int))
    (dst : ∀ j ∈ jobs r,
      ((initial j).mem "dst" ((initial j).pids 0)).read .int = (destinationRow r j : Int)) :
    ∃ states : Bool → Job → State ℝ,
      (∀ version j, j ∈ jobs r →
        exec (QuantProfile.algebra QuantReal.ops) (Shared.kernel version r) (initial j) =
          some (states version j)) ∧
      (∀ version j, j ∈ jobs r → ∀ address, address ∈ rowAddresses r j →
        SourceBacking.project r (states version j) address = SharedTarget.idealValue r
          (QuantExecution.inputRow (Shared.parameters r) (initial j) (sourceRow r j) Bool.true)
          (QuantExecution.inputRow (Shared.parameters r) (initial j) (sourceRow r j) Bool.false) address) ∧
      (∀ version j, j ∈ jobs r → ∀ reg i,
        SharedReadback.Outside r (destinationRow r j) reg i →
          (states version j).mem reg i = (initial j).mem reg i) := by
  let states : Bool → Job → State ℝ := fun version j =>
    if hj : j ∈ jobs r then
      (SharedExecution.executed QuantReal.ops r d (initial j) (sourceRow r j)
        (destinationRow r j) (src j hj) (dst j hj) version).val
    else initial j
  have success : ∀ version j, j ∈ jobs r →
      exec (QuantProfile.algebra QuantReal.ops) (Shared.kernel version r) (initial j) =
        some (states version j) := by
    intro version j hj
    simp only [states,hj,↓reduceDIte]
    exact (SharedExecution.executed QuantReal.ops r d (initial j) (sourceRow r j)
      (destinationRow r j) (src j hj) (dst j hj) version).property
  refine ⟨states,success,?_,?_⟩
  · intro version j hj address ha
    rw [SharedReadback.project QuantReal.ops r d (initial j) (states version j) j
      (src j hj) (dst j hj) version (success version j hj) address ha]
    exact SharedTarget.real_value_independent r d _ _ address
  · intro version j hj reg i outside
    exact SharedReadback.frame QuantReal.ops r d (initial j) (states version j)
      (sourceRow r j) (destinationRow r j) (src j hj) (dst j hj) version
      (success version j hj) reg i outside

#axiomsClean whole_shared_real
#specNonCircular SharedTarget.idealValue avoiding [FP.Structural.exec,Shared.original,Shared.candidate]
end
end VeriTile.Bench.Repairs.KvCacheFilling.RawReal
