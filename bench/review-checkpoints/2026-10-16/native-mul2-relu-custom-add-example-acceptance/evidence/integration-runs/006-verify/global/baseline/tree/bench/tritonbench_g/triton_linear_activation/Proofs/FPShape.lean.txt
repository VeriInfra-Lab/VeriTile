import bench.tritonbench_g.triton_linear_activation.Proofs.FPReference
import bench.tritonbench_g.triton_linear_activation.TritonLinearActivation
namespace VeriTile.Bench.TritonBenchG.TritonLinearActivation.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 35000
set_option linter.unusedSimpArgs false

/-- Erased control/address/operation correspondence to the complete accepted
old body. The two old irrational constant expressions are only source tags;
FPLiterals separately binds their actual host→fp32 values. -/
theorem source_projection (fmt saved : Format) (p : Parameters) :
    (source .source fmt saved p).eraseDType.surfaceBody =
      (kernel_fma_surface p.output p.saved p.input p.weight p.bias p.rows p.columns
        p.outputRowStride p.outputColStride p.savedRowStride p.savedColStride
        p.inputRowStride p.inputInnerStride p.weightColStride p.weightInnerStride
        p.blockRows p.groupRows p.blockColumns p.blockInner p.blocks
        p.hasBias p.saveInputs p.activation).eraseDType.surfaceBody := by
  cases fmt <;> cases saved <;>
    simp [source,typed,dtype,LoraExpandGemv.FPValue.outputType,kernel_fma_surface,
      ComputeKernel.eraseDType,ComputeKernel.toAlgorithm?,ComputeStmt.listToAlgorithm?,
      ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
      Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType,
      MemAccess.eraseDType,MaskOpt.eraseDType,ComputeKernel.fromKernelBody,
      ComputeKernel.surfaceBody,ComputeDType.eraseDType,Except.bind,bind,pure,
      VeriTile.Triton.eraseDType,FloatDType.numericDType,NumericDType.eraseDType]
  all_goals repeat' constructor
  all_goals try conv_lhs => rw [Op.eraseDType.eq_def]
  all_goals try simp [Op.eraseDType,VeriTile.Triton.eraseDType]
  all_goals try simp only [Op.eraseDType.eq_def]
  all_goals try simp only [NumericDType.eraseDType,eq_mp_eq_cast,eq_mpr_eq_cast,cast_eq]
  all_goals repeat' constructor
  all_goals rfl

theorem compiled_dot_node (fmt saved : Format) (p : Parameters) :
    (body .accumulator fmt saved p)[2]? = some (.assign .real [p.blockRows,p.blockColumns] "acc"
      (.compute (.alg .fp32 (.dotAcc (dtype fmt) (dtype fmt) .real (batch := [])
        (.ref (dtype fmt).toTileDType [p.blockRows,p.blockInner] "a")
        (.ref (dtype fmt).toTileDType [p.blockInner,p.blockColumns] "b")
        (.ref .real [p.blockRows,p.blockColumns] "acc") {})))) := by
  cases fmt <;> rfl

def geometryState (s : State Word) : BlockState := { (default : BlockState) with pids := s.pids }

theorem output_injective_of_fit (p : Parameters) (s : State Word)
    (hFitM : blockRow p s*p.blockRows+p.blockRows≤p.rows)
    (hFitN : blockCol p s*p.blockColumns+p.blockColumns≤p.columns)
    (hs : p.outputColStride=1) (hb : p.blockColumns≤p.outputRowStride) :
    Function.Injective (outputOffset p s) := by
  have h := cOffset_injective_of_fit (geometryState s) p.rows p.columns p.blockRows p.blockColumns
    p.groupRows p.outputRowStride hFitM hFitN hb
  unfold outputOffset
  unfold cOffset at h
  simpa only [hs,row,col,blockRow,blockCol,groupIdx,groupSize,width,gridRows,gridCols,
    cOffset,rowIndex,colIndex,rowGlobal,colGlobal,blockMIdx,blockNIdx,kernelMin,geometryState] using h

theorem saved_injective_of_fit (p : Parameters) (s : State Word)
    (hFitM : blockRow p s*p.blockRows+p.blockRows≤p.rows)
    (hFitN : blockCol p s*p.blockColumns+p.blockColumns≤p.columns)
    (hs : p.savedColStride=1) (hb : p.blockColumns≤p.savedRowStride) :
    Function.Injective (savedOffset p s) := by
  have h := cOffset_injective_of_fit (geometryState s) p.rows p.columns p.blockRows p.blockColumns
    p.groupRows p.savedRowStride hFitM hFitN hb
  unfold savedOffset
  unfold cOffset at h
  simpa only [hs,row,col,blockRow,blockCol,groupIdx,groupSize,width,gridRows,gridCols,
    cOffset,rowIndex,colIndex,rowGlobal,colGlobal,blockMIdx,blockNIdx,kernelMin,geometryState] using h

theorem active_of_fit (p : Parameters) (s : State Word)
    (hFitM : blockRow p s*p.blockRows+p.blockRows≤p.rows)
    (hFitN : blockCol p s*p.blockColumns+p.blockColumns≤p.columns)
    (i : TileIndex [p.blockRows,p.blockColumns]) : active p s i=Bool.true := by
  have hr := i.1.isLt
  have hc := i.2.1.isLt
  simp only [active,row,col,Bool.and_eq_true,decide_eq_true_eq]
  exact ⟨Nat.mod_lt _ (by omega),Nat.mod_lt _ (by omega)⟩
#axiomsClean source_projection
#axiomsClean compiled_dot_node
#axiomsClean output_injective_of_fit
#axiomsClean saved_injective_of_fit
#axiomsClean active_of_fit
end
end VeriTile.Bench.TritonBenchG.TritonLinearActivation.FPValue
