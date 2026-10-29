import bench.tritonbench_g.int4_matmul.SoftwareCorrect
namespace VeriTile.Bench.TritonBenchG.Int4Matmul.SoftwareCorrect
open VeriTile.Triton FP.Structural FPTyped FPClosed
open FPSoftware (Word)
noncomputable section

/-- Public admission of the previously independently checked complete source
software profile. The separate dot-then-add graph, full original scope, and
independent bit output are preserved. This is not the compiled DotAcc graph
or a claim about native tensor-core accumulation. -/
specification source_profile_software_value_correctness
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
  exact source_profile_software_correctness p s hTrip hSplit hInputs hw

#axiomsClean source_profile_software_value_correctness
end
end VeriTile.Bench.TritonBenchG.Int4Matmul.SoftwareCorrect
