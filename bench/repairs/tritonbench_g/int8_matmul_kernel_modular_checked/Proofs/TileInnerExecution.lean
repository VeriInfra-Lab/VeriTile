import bench.repairs.tritonbench_g.int8_matmul_kernel_modular_checked.Proofs.TileExecution

namespace VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.TileModel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001 (Byte)
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
attribute [local simp 2000] FP.Structural.eval_intToSigned

def innerExecution (cast : Word32 → Byte) (M : Algebra Word32) (p : Parameters)
    (initial current : State Word32) (f k : Nat) (hf : f < 4) (hk : k < p.steps)
    (defined : InputDefined p initial) (h : InnerInvariant cast p initial f k current) :
    {t : State Word32 // run (backend cast M) (innerBody p)
      (current.setReg "j" .nat [] (fun _ => k)) = some t} := by
  have hcast (i : AIndex p) : convert8 cast (aWord p initial f k i) =
      some (aValue cast p initial f k i) := convert8_defined _ _ (defined f hf k hk i)
  have validA := (dot_inputs_valid cast p initial f k).1
  have validB := (dot_inputs_valid cast p initial f k).2
  have collected : collectValues? (fun i => convert8 cast (aWord p initial f k i)) =
      some (aValue cast p initial f k) := by
    simp only [hcast, collectValues?_some]
  simp only [aWord, aValue, Parameters.steps] at hcast
  simp only [aWord, Parameters.steps] at collected
  simp only [FP.IntegerDot.valid, bValue_source, Int.natCast_shiftRight] at validA validB
  refine ⟨?_, ?_⟩
  swap
  simp [innerBody, outerBody, source, compileAssertionProjection, quotedSource, kernel,
    ComputeKernel.surfaceBody, run, step, stepAlg, evalExpr, evalComputeOp, evalOp_unfold,
    evalOffsets, store, numeric, bop, natLt, numericEq, ComputeDType.eraseDType, ofFloat, toFloat,
    FloatDType.numericDType, FloatDType.toTileDType,
    Region.cast, Option.bind, integralFloorDiv, integralMod,
    h.offsk, h.offsn, h.aptrs, h.bptrs, h.acc, h.fieldReg, h.memory, backend,
    FP.Structural.eval_intToSigned, collected, hcast, validA, validB,
    FP.IntegerDot.backend, FP.IntegerDot.supported, Parameters.steps,
    Broadcast.leftIndex, Broadcast.rightIndex]
  simp only [BitVec.ofNat_eq_ofNat] at collected
  rw [collected]
  simp [Option.bind, h.offsk, h.offsn, h.aptrs, h.bptrs, h.acc, h.fieldReg, h.memory,
    bop, Broadcast.leftIndex, Broadcast.rightIndex, validA, validB, FP.IntegerDot.valid]
  have hdot : FP.IntegerDot.supported { outDType := some .int } ∧
      FP.IntegerDot.valid (α := Word32) (aValue cast p initial f k) ∧
      FP.IntegerDot.valid (α := Word32) (fun i : BIndex p =>
        (↑((if i.1.val < p.inner then
          (initial.mem p.weight (bOffset p initial k (i.1,i.2.1,PUnit.unit))).read .nat
          else 0) &&& (3 <<< (2*f))) : Int) >>> (2*f)-1) := by
    refine ⟨by simp [FP.IntegerDot.supported], validA, ?_⟩
    intro i
    rcases i with ⟨r,c,u⟩
    cases u
    exact validB (r,c,PUnit.unit)
  dsimp only [FP.IntegerDot.supported, FP.IntegerDot.valid] at hdot
  erw [if_pos hdot]
  simp [Option.bind, h.offsk, h.offsn, h.aptrs, h.bptrs, h.acc, h.fieldReg, h.memory,
    bop, Broadcast.leftIndex, Broadcast.rightIndex]
  rfl

#axiomsClean innerExecution
end
end VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.TileModel
