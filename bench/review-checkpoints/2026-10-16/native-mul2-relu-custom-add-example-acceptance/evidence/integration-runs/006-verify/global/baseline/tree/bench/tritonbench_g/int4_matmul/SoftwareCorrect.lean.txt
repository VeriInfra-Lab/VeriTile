import bench.tritonbench_g.int4_matmul.Proofs.FPClosed
namespace VeriTile.Bench.TritonBenchG.Int4Matmul.SoftwareCorrect
open VeriTile.Triton FP.Structural FPTyped FPClosed
open FPSoftware (Word)
noncomputable section

/-- Complete accepted source-loop scope under the explicit separate-dot
software schedule. This is an exposed specialization of the old ∀D theorem,
not evidence that the compiled accumulator-dot graph equals this graph. -/
theorem source_profile_software_correctness
    (p : Parameters) (s : State Word)
    (hTrip : p.inner = p.blockInner * p.steps) (hSplit : s.pids 1 = 0)
    (hInputs : PackedInputs p s)
    (hw : ∀ i j, active p s i → active p s j →
      outputOffset p s i = outputOffset p s j → i = j) :
    ∃ final,
      FP.Structural.exec softwareBackend (source p) s = some final ∧
      Post p s final (outputBits p s) ∧
      ∀ step, step < p.steps → ∀ i : BIndex p, ∃ word zero : BitVec 32,
        s.mem p.weight (bOffset p s step i) = .mk .int word.toInt ∧
        s.mem p.zero (zeroOffset p s step i) = .mk .int zero.toInt ∧
        difference p s step i = packedDifference p s i word zero := by
  exact correctness_for_any_rest unused p s hTrip hSplit hInputs hw

#axiomsClean source_profile_software_correctness
end
end VeriTile.Bench.TritonBenchG.Int4Matmul.SoftwareCorrect
