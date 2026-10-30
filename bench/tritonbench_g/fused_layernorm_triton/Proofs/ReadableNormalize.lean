import bench.tritonbench_g.fused_layernorm_triton.Proofs.ReadableStatistics
import bench.tritonbench_g.fused_layernorm_triton.Proofs.Mathematical_WholeNormalizeLoop

namespace VeriTile.Bench.Repairs.FusedLayernormTriton.ReadableNormalize
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.FusedLayernormTriton FPStages Whole
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

structure Pins (s : State α) (row : Nat) (mean inverse : α) : Prop where
  row : s.regs .nat [] "xrow" = some (fun _ => row)
  mean : s.regs .real [] "rowMean" = some (fun _ => mean)
  inverse : s.regs .real [] "rowInverse" = some (fun _ => inverse)

theorem projections_run (M : Algebra α) (equal32 : α → α → Bool)
    (s : State α) (X W Bias Y Mean Rstd : RegionName) (S D B row : Nat)
    (tree : Kernels.ReductionTree B) (mean inverse : α)
    (hm : s.regs .real [1,1] "tmp3" = some (fun _ => mean))
    (hr : s.regs .real [1,1] "tmp10" = some (fun _ => inverse))
    (hx : s.regs .nat [1,1] "x0" = some (fun _ => row)) :
    ∃ final, run (withEquality M equal32)
      (SourcePieces.projections X W Bias Y Mean Rstd S D B tree) s = some final ∧
      final.mem = s.mem ∧ NormalizeRegisters s final row mean inverse ∧ Pins final row mean inverse := by
  simp only [SourcePieces.projections, SourcePieces.normalization, SourcePieces.statistics,
    SourcePieces.source, Bool.false_eq_true, ite_false, Kernels.kernel,
    ComputeKernel.surfaceBody, Source.dynamicRange?, Option.map, Option.getD,
    String.reduceEq, beq_self_eq_true, ↓reduceIte, List.take]
  simp [run, step, evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
    ComputeDType.eraseDType, withEquality, hm, hr, hx]
  constructor <;> constructor <;> simp [State.setReg, hm, hr, hx]

theorem candidate_chunk_run (M : Algebra α) (equal32 : α → α → Bool)
    (s : State α) (X W Bias Y Mean Rstd : RegionName) (S D B k row : Nat)
    (tree : Kernels.ReductionTree B) (mean inverse : α)
    (hm : s.regs .real [1,1] "tmp3" = some (fun _ => mean))
    (hr : s.regs .real [1,1] "tmp10" = some (fun _ => inverse))
    (hx : s.regs .nat [1,1] "x0" = some (fun _ => row)) (pins : Pins s row mean inverse) :
    ∃ final, run (withEquality M equal32)
      (SourcePieces.normalization Bool.true X W Bias Y Mean Rstd S D B tree).body
      (s.setReg "roffset" .nat [] (fun _ => k*B)) = some final ∧
      final.mem = (outputChunk M s X W Bias Y D B k row mean inverse).mem ∧
      NormalizeRegisters s final row mean inverse ∧ Pins final row mean inverse := by
  simp only [SourcePieces.normalization, SourcePieces.statistics, SourcePieces.source,
    Kernels.hoisted, ComputeKernel.surfaceBody, Source.dynamicRange?, Option.map,
    Option.getD, String.reduceEq, beq_self_eq_true, ↓reduceIte]
  simp [run, step, evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
    ComputeDType.eraseDType, numeric, bop, natLt, numericLt, withEquality,
    Region.cast, ofFloat, toFloat, hm, hr, hx, pins.row, pins.mean, pins.inverse, store,
    maskedNormalizeValue, outputChunk]
  constructor
  · apply typed_scatter_memory_congr
    simp only [State.setReg_mem]
  · constructor
    · constructor <;> simp [typed_scatter_regs, typed_scatter_undef, hm, hr, hx]
    · constructor <;> simp [typed_scatter_regs, pins.row, pins.mean, pins.inverse]

theorem original_chunk_run (M : Algebra α) (equal32 : α → α → Bool)
    (s : State α) (X W Bias Y Mean Rstd : RegionName) (S D B k row : Nat)
    (tree : Kernels.ReductionTree B) (mean inverse : α)
    (hm : s.regs .real [1,1] "tmp3" = some (fun _ => mean))
    (hr : s.regs .real [1,1] "tmp10" = some (fun _ => inverse))
    (hx : s.regs .nat [1,1] "x0" = some (fun _ => row)) :
    ∃ final, run (withEquality M equal32)
      (SourcePieces.normalization Bool.false X W Bias Y Mean Rstd S D B tree).body
      (s.setReg "roffset" .nat [] (fun _ => k*B)) = some final ∧
      final.mem = (outputChunk M s X W Bias Y D B k row mean inverse).mem ∧
      NormalizeRegisters s final row mean inverse ∧ Pins final row mean inverse := by
  simp only [SourcePieces.normalization, SourcePieces.statistics, SourcePieces.source,
    Bool.false_eq_true, ite_false, Kernels.kernel, ComputeKernel.surfaceBody, Source.dynamicRange?, Option.map,
    Option.getD, String.reduceEq, beq_self_eq_true, ↓reduceIte]
  simp [run, step, evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
    ComputeDType.eraseDType, numeric, bop, natLt, numericLt, withEquality,
    Region.cast, ofFloat, toFloat, hm, hr, hx, store,
    maskedNormalizeValue, outputChunk]
  constructor
  · apply typed_scatter_memory_congr
    simp only [State.setReg_mem]
  · constructor
    · constructor <;> simp [typed_scatter_regs, typed_scatter_undef, hm, hr, hx]
    · constructor <;> simp [typed_scatter_regs]

end
end VeriTile.Bench.Repairs.FusedLayernormTriton.ReadableNormalize
