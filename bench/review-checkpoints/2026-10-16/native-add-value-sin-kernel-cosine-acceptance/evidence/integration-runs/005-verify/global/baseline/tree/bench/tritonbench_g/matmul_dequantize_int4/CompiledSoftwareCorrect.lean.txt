import bench.tritonbench_g.matmul_dequantize_int4.Proofs.FPCompiledWitness
namespace VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4
open VeriTile.Triton FP.Structural
open Int4Matmul.FPSoftware (Word)
noncomputable section
/-! Two distinct compiled instruction graphs. These statements prove their
complete software bit profiles, including all original K iterations and both
NO_GROUPS branches. FMAPlan is explicit ordering data, not a law that hardware
MMA/WGMMA is representable by such a serial schedule. No GPU equivalence is
asserted. The historical source-contract declarations remain unchanged. -/

specification matmul_dequantize_int4_compiled_separate_bits_correctness
    (plan : FPCompiled.FMAPlan) (p : FPTyped.Parameters) (s : State Word)
    (hInputs : FPTyped.PackedInputs p s)
    (hw : ∀ i j, FPTyped.active p s i → FPTyped.active p s j →
      FPTyped.outputOffset p s i = FPTyped.outputOffset p s j → i = j) :
    ∃ final, FP.Structural.exec (FPCompiled.backend (FPSoftware.scalar FPCompiled.unused)
        (FPCompiled.plannedDotAcc plan)) (FPCompiled.source p) s = some final ∧
      final.mem = (FPCompiled.plannedReference plan p s).mem ∧
      FPCompiled.Post p s final (FPCompiled.plannedOutput plan p s) ∧
      ∀ step, step < FPTyped.steps p → ∀ i : FPTyped.BIndex p, ∃ word zero : BitVec 32,
        s.mem p.weight (FPTyped.bOffset p s step i) = .mk .int word.toInt ∧
        s.mem p.zero (FPTyped.zeroOffset p s step (i.2.1, PUnit.unit)) = .mk .int zero.toInt ∧
        FPTyped.weightNibble p s step i = ((word >>> (i.1.val % 8 * 4)) &&& (15 : BitVec 32)).toInt ∧
        FPTyped.zeroNibble p s step (i.2.1, PUnit.unit) =
          ((zero >>> (FPTyped.inputColumn p s i.2.1 % 8 * 4)) &&& (15 : BitVec 32)).toInt := by
  obtain ⟨final,he,hp⟩ := FPCompiled.planned_readback_correctness plan FPCompiled.unused p s hw
  obtain ⟨other,ho,hm⟩ := FPCompiled.planned_memory_correctness plan FPCompiled.unused p s
  have same : final=other := Option.some.inj (he.symm.trans ho)
  exact ⟨final,he,same.symm ▸ hm,hp,FPTyped.packed_nibbles_bridge p s hInputs⟩

#axiomsClean matmul_dequantize_int4_compiled_separate_bits_correctness

specification matmul_dequantize_int4_compiled_fused_bits_correctness
    (plan : FPCompiled.FMAPlan) (p : FPTyped.Parameters) (s : State Word)
    (hInputs : FPTyped.PackedInputs p s)
    (hw : ∀ i j, FPTyped.active p s i → FPTyped.active p s j →
      FPTyped.outputOffset p s i = FPTyped.outputOffset p s j → i = j) :
    ∃ final, FPFused.exec (FPSoftware.scalar FPCompiled.unused)
        (FPCompiled.plannedDotAcc plan) p s = some final ∧
      final.mem = (FPFused.plannedReference plan p s).mem ∧
      FPCompiled.Post p s final (FPFused.plannedOutput plan p s) ∧
      ∀ step, step < FPTyped.steps p → ∀ i : FPTyped.BIndex p, ∃ word zero : BitVec 32,
        s.mem p.weight (FPTyped.bOffset p s step i) = .mk .int word.toInt ∧
        s.mem p.zero (FPTyped.zeroOffset p s step (i.2.1, PUnit.unit)) = .mk .int zero.toInt ∧
        FPTyped.weightNibble p s step i = ((word >>> (i.1.val % 8 * 4)) &&& (15 : BitVec 32)).toInt ∧
        FPTyped.zeroNibble p s step (i.2.1, PUnit.unit) =
          ((zero >>> (FPTyped.inputColumn p s i.2.1 % 8 * 4)) &&& (15 : BitVec 32)).toInt := by
  obtain ⟨final,he,hp⟩ := FPFused.planned_readback_correctness plan FPCompiled.unused p s hw
  obtain ⟨other,ho,hm⟩ := FPFused.planned_memory_correctness plan FPCompiled.unused p s
  have same : final=other := Option.some.inj (he.symm.trans ho)
  exact ⟨final,he,same.symm ▸ hm,hp,FPTyped.packed_nibbles_bridge p s hInputs⟩

#axiomsClean matmul_dequantize_int4_compiled_fused_bits_correctness

end
end VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4
