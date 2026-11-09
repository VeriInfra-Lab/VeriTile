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
  row : s.regs .int [] "xrow" = some (fun _ => (row : Int))
  mean : s.regs .real [] "row_mean" = some (fun _ => mean)
  inverse : s.regs .real [] "row_inverse" = some (fun _ => inverse)

theorem projections_run (M : Algebra α) (equal32 : α → α → Bool)
    (s : State α) (X W Bias Y Mean Rstd : RegionName) (S D B row : Nat)
    (tree : Kernels.ReductionTree B) (mean inverse : α)
    (hm : s.regs .real [1,1] "tmp3" = some (fun _ => mean))
    (hr : s.regs .real [1,1] "tmp10" = some (fun _ => inverse))
    (hx : s.regs .int [1,1] "x0" = some (fun _ => (row : Int)))
    (indexPins : SourceIndexPins B s) :
    ∃ final, run (withEquality M equal32)
      (SourcePieces.projections X W Bias Y Mean Rstd S D B tree) s = some final ∧
      final.mem = s.mem ∧ NormalizeRegisters s final row mean inverse ∧
      Pins final row mean inverse ∧ SourceIndexPins B final := by
  rw [←SourcePieces.projectionBody_eq X W Bias Y Mean Rstd S D B tree]
  simp only [SourcePieces.projectionBody]
  simp [run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    ComputeDType.eraseDType,withEquality,hm,hr,hx]
  constructor <;> constructor <;> simp [State.setReg,hm,hr,hx,indexPins.zero,indexPins.base]
  all_goals constructor <;> simp [State.setReg,hm,hr,hx,indexPins.zero,indexPins.base]

theorem candidate_chunk_run
    (M : Algebra α) (equal32 : α → α → Bool)
    (s : State α) (X W Bias Y Mean Rstd : RegionName) (S D B k row : Nat)
    (tree : Kernels.ReductionTree B) (mean inverse : α)
    (hm : s.regs .real [1,1] "tmp3" = some (fun _ => mean))
    (hr : s.regs .real [1,1] "tmp10" = some (fun _ => inverse))
    (hx : s.regs .int [1,1] "x0" = some (fun _ => (row : Int)))
    (indexPins : SourceIndexPins B s) (pins : Pins s row mean inverse) :
    ∃ final, run (withEquality M equal32)
      (SourcePieces.normalization Bool.true X W Bias Y Mean Rstd S D B tree).body
      (s.setReg "roffset" .int [] (fun _ => (k*B : Nat))) = some final ∧
      final.mem = (outputChunk M s X W Bias Y D B k row mean inverse).mem ∧
      NormalizeRegisters s final row mean inverse ∧ SourceIndexPins B final ∧ Pins final row mean inverse := by
  rw [←SourcePieces.normalizationBody_eq Bool.true X W Bias Y Mean Rstd S D B tree]
  simp only [SourcePieces.normalizationBody]
  simp [run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    ComputeDType.eraseDType,numeric,bop,natLt,numericLt,withEquality,
    Region.cast,ofFloat,toFloat,hm,hr,hx,State.setReg,indexPins.base,store,
    maskedNormalizeValue,outputChunk,source_column_lt,source_address,source_column_address,pins.row,pins.mean,pins.inverse]
  constructor
  · apply typed_scatter_memory_congr
    simp only [State.setReg_mem]
  · constructor
    · constructor <;> simp [typed_scatter_regs,typed_scatter_undef,State.setReg,hm,hr,hx]
    · constructor
      · constructor <;> simp [typed_scatter_regs,State.setReg,indexPins.zero,indexPins.base]
      · constructor <;> simp [typed_scatter_regs,State.setReg,pins.row,pins.mean,pins.inverse]

theorem original_chunk_run
    (M : Algebra α) (equal32 : α → α → Bool)
    (s : State α) (X W Bias Y Mean Rstd : RegionName) (S D B k row : Nat)
    (tree : Kernels.ReductionTree B) (mean inverse : α)
    (hm : s.regs .real [1,1] "tmp3" = some (fun _ => mean))
    (hr : s.regs .real [1,1] "tmp10" = some (fun _ => inverse))
    (hx : s.regs .int [1,1] "x0" = some (fun _ => (row : Int)))
    (indexPins : SourceIndexPins B s) :
    ∃ final, run (withEquality M equal32)
      (SourcePieces.normalization Bool.false X W Bias Y Mean Rstd S D B tree).body
      (s.setReg "roffset" .int [] (fun _ => (k*B : Nat))) = some final ∧
      final.mem = (outputChunk M s X W Bias Y D B k row mean inverse).mem ∧
      NormalizeRegisters s final row mean inverse ∧ SourceIndexPins B final := by
  rw [←SourcePieces.normalizationBody_eq Bool.false X W Bias Y Mean Rstd S D B tree]
  simp only [SourcePieces.normalizationBody]
  simp [run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    ComputeDType.eraseDType,numeric,bop,natLt,numericLt,withEquality,
    Region.cast,ofFloat,toFloat,hm,hr,hx,State.setReg,indexPins.base,store,
    maskedNormalizeValue,outputChunk,source_column_lt,source_address,source_column_address]
  constructor
  · simp [TileShape.allIndices,List.finRange_succ,List.foldl_map]
    apply typed_scatter_memory_congr
    simp only [State.setReg_mem]
  · constructor
    · constructor <;> simp [typed_scatter_regs,typed_scatter_undef,State.setReg,hm,hr,hx]
    · constructor <;> simp [typed_scatter_regs,State.setReg,indexPins.zero,indexPins.base]

end
end VeriTile.Bench.Repairs.FusedLayernormTriton.ReadableNormalize
