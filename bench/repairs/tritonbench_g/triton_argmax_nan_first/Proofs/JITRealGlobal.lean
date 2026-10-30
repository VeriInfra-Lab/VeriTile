import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.JITStage1
import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.TypedRealGlobal

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.JITRealGlobal
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source Reduction
open TypedReal
noncomputable section
set_option maxHeartbeats 1500000
set_option maxRecDepth 3000
set_option linter.unusedSimpArgs false
theorem stage1_run (kind : TypedScratch.Kind) (rest : Algebra Value) (s : State Value)
    (X V : RegionName) (I : Region .int) (size block base : Nat) (index64 : Bool) (positive : 0<block) :
    ∃ t, FP.Structural.exec (backend rest) (JITSource.stage1 X V I (JITStage1.floatingType kind) block index64) (JITStage1.arguments s size base)=some t ∧
      t.mem=(firstWritten kind s X V I size block base positive).mem := by
  cases kind <;> cases index64 <;>
    simp [JITSource.stage1,JITStage1.floatingType,JITStage1.arguments,
      ComputeKernel.surfaceBody,FP.Structural.exec,run,step,evalExpr,evalComputeOp,evalOp_unfold,
      ComputeDType.eraseDType,backend,indexed,
      TileShape.axisDim,positive,Nat.ne_of_gt positive,TileShape.insertAxisIndex,TileShape.eraseAxis,
      ofFloat,toFloat,bop,numeric,numericLt,numericEq,natLt,TileShape.discreteMaximum,
      TileShape.discreteIntArgMax,orderedEq,globalLoaded,firstIndex,firstWritten,Option.bind,
      store,TileShape.allIndices,Region.cast,TypedScratch.channel,-Nat.cast_add,-Nat.cast_mul,← Int.natCast_add,← Int.natCast_mul]
  all_goals simp only [State.write,State.setReg_mem]
  all_goals rfl

theorem stage2_run (kind : TypedScratch.Kind) (rest : Algebra Value) (s : State Value)
    (V : RegionName) (I O : Region .int) (size block : Nat) (positive : 0<block) :
    ∃ t, FP.Structural.exec (backend rest) (JITSource.stage2 V I O (JITStage1.floatingType kind) block) (s.setReg "mid_size" .int [] (fun _ => (size : Int)))=some t ∧
      t.mem=(secondWritten kind s V I O size block positive).mem := by
  cases kind <;>
    simp [JITSource.stage2,JITStage1.floatingType,
      ComputeKernel.surfaceBody,FP.Structural.exec,run,step,evalExpr,evalComputeOp,evalOp_unfold,
      ComputeDType.eraseDType,backend,indexed,
      TileShape.axisDim,positive,Nat.ne_of_gt positive,TileShape.insertAxisIndex,TileShape.eraseAxis,
      ofFloat,toFloat,bop,numeric,numericLt,numericEq,natLt,TileShape.discreteMaximum,
      TileShape.discreteIntArgMax,orderedEq,secondIndex,secondWritten,Option.bind,
      store,TileShape.allIndices,Region.cast,TypedScratch.channel,-Nat.cast_add,-Nat.cast_mul,← Int.natCast_add,← Int.natCast_mul]
  all_goals simp only [State.write,State.setReg_mem]
  all_goals rfl


theorem second_reference (kind : TypedScratch.Kind) (rest : Algebra Value) (s : State Value)
    (V : RegionName) (I O : Region .int) (c : GlobalConfig) (coverage : Math.GlobalCoverage c)
    (input : Nat → ℝ)
    (values : ∀ i,i<c.blocks → toFloat (TypedScratch.channel kind)
      ((s.mem V i).read (TypedScratch.channel kind).toTileDType)=(input (Math.firstIndex c input i) : Value))
    (indices : ∀ i,i<c.blocks → (s.mem (Region.cast I) i).read .int=Int.ofNat (Math.firstIndex c input i)) :
    ∃ t, FP.Structural.exec (backend rest) (JITSource.stage2 V I O (JITStage1.floatingType kind) c.midBlock) (s.setReg "mid_size" .int [] (fun _ => (c.blocks : Int)))=some t ∧
      t.mem=(s.write (Region.cast O) 0 (.mk .int (Int.ofNat
        (Math.argmax c.size coverage.sizePositive (fun i : Fin c.size => input i.val)).val))).mem := by
  let positive := coverage.blocksPositive.trans_le coverage.midFits
  obtain ⟨t,run,mem⟩ := stage2_run kind rest s V I O c.blocks c.midBlock positive
  have loads : (fun lane : Fin c.midBlock => if lane.val<c.blocks then
      toFloat (TypedScratch.channel kind) ((s.mem V lane.val).read (TypedScratch.channel kind).toTileDType)
      else ⊥)=Math.midValues c input := by
    funext lane
    by_cases active : lane.val<c.blocks
    · simp [Math.midValues,active,values _ active]
    · simp [Math.midValues,active]
  have selected : secondIndex kind s V c.blocks c.midBlock positive=(Math.globalSelected c input).val := by
    unfold secondIndex
    rw [loads,best_tree positive _ c.secondTree coverage.secondComplete]
    rfl
  refine ⟨t,run,mem.trans ?_⟩
  unfold secondWritten
  rw [selected,indices _ (Math.mid_valid c coverage input)]
  have result := congrArg Fin.val (least_maximum_unique _ _ _ (Math.global_mathematical c coverage input)
    (Math.argmax_spec c.size coverage.sizePositive (fun i => input i.val)))
  rw [←result]
  rfl


#axiomsClean second_reference
#axiomsClean stage1_run
#axiomsClean stage2_run
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.JITRealGlobal
