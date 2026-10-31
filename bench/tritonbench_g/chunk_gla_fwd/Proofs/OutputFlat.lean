import bench.tritonbench_g.chunk_gla_fwd.Proofs.LocalReal
import bench.tritonbench_g.chunk_gla_fwd.Proofs.LocalRealSafety
import VeriTile.Triton.Memory.FlattenR
import VeriTile.Triton.Math.MatrixRewrite

/-! Flat execution and exact exterior frame for the actual repaired FP32
consumer and its current sum-left candidate. All source statements are extracted
from the canonical kernel and the complete decompositions are checked below. -/
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.OutputFlat
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGlaFwd
open RealOutputIO
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false

def source (p : Layout) (r : Regions) (scale : ℝ) (swapped : Bool) : ComputeKernel :=
  TypedOutput.source .fp32 .fp32 .fp32 swapped .fp32 .fp32
    r.query r.value r.gate r.history r.output r.attention
    p.keyHead p.keyTime p.valueHead p.valueTime p.historyHead p.historyKey scale
    p.T p.K p.V p.BT p.BK p.BV

def program (p : Layout) (r : Regions) (scale : ℝ) (swapped : Bool) :=
  (source p r scale swapped).eraseDType.toAlgKernel

def before (p : Layout) (r : Regions) (scale : ℝ) := (program p r scale Bool.false).body.take 12
def after (p : Layout) (r : Regions) (scale : ℝ) := (program p r scale Bool.false).body.drop 13

def intraDot (p : Layout) : Op .real [p.BT,p.BV] :=
  .dot (batch := []) (.ref .real [p.BT,p.BT] "b_A") (.ref .real [p.BT,p.BV] "b_v")
    {allowTF32:=some Bool.false}

theorem decomposition (p : Layout) (r : Regions) (scale : ℝ) (swapped : Bool) :
    (program p r scale swapped).body = before p r scale ++
      .assign .real [p.BT,p.BV] "b_o"
        (if swapped then .add .real (.consSame (.consSame .nil)) (intraDot p) (.ref .real [p.BT,p.BV] "b_o")
          else .add .real (.consSame (.consSame .nil)) (.ref .real [p.BT,p.BV] "b_o") (intraDot p)) ::
      after p r scale := by
  cases swapped
  all_goals simp only [before,after,program,source,TypedOutput.source,TypedOutput.fp32Original,TypedOutput.fp32SumLeft,
    ComputeKernel.surfaceBody,ComputeKernel.eraseDType,ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,
    ComputeKernel.fromKernelBody,ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,ComputeDType.eraseDType,
    List.map_cons,List.map_nil,Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,NumericDType.eraseDType,
    ComparableDType.eraseDType,IntegralDType.eraseDType,Except.bind,bind,eraseDType_float,
    Bool.false_eq_true,↓reduceIte]
  all_goals simp only [Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,FloatDType.toTileDType]
  all_goals rfl

theorem fragment (p : Layout) (r : Regions) (scale : ℝ) :
    (program p r scale Bool.false).FlattenOk := by
  rw [Kernel.FlattenOk]
  simp only [program,source,TypedOutput.source,TypedOutput.fp32Original,TypedOutput.fp32SumLeft,
    ComputeKernel.surfaceBody,ComputeKernel.eraseDType,ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,
    ComputeKernel.fromKernelBody,ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,ComputeDType.eraseDType,
    List.map_cons,List.map_nil,Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,NumericDType.eraseDType,
    ComparableDType.eraseDType,IntegralDType.eraseDType,Except.bind,bind,eraseDType_float,
    Bool.false_eq_true,↓reduceIte]
  simp only [Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,FloatDType.toTileDType]
  simp [StmtList.FlattenOk,Stmt.FlattenOk,Op.FlattenOk]
  all_goals try simp [Op.FlattenOk.eq_def]

theorem candidate_executes_like (p : Layout) (r : Regions) (scale : ℝ)
    (A : FlatAlloc) (s t : BlockState)
    (ran : exec (A.flattenKernel (program p r scale Bool.false)) s=some t) :
    exec (A.flattenKernel (program p r scale Bool.true)) s=some t := by
  rw [exec,FlatAlloc.flattenKernel,decomposition] at ran ⊢
  simp only [Bool.false_eq_true,↓reduceIte,FlatAlloc.flattenStmts_append,
    FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,FlatAlloc.flattenOp] at ran ⊢
  exact MatrixRewrite.add_commute _ _ _ _ _ s t ran

theorem source_executes (p : Layout) (r : Regions) (scale : ℝ)
    (A : FlatAlloc) (hd : A.Disjoint) (hcov : ∀ rg,rg∉A.regions → A.extent rg=0)
    (swapped : Bool) (s : BlockState) (hu : s.undef=fun _ _=>0)
    (hinj : ∀ i j,active p (launch s) i → active p (launch s) j →
      outputAddress p (launch s) i=outputAddress p (launch s) j → i=j)
    (hb : RealOutputSafety.Bounds s r.query r.value r.gate r.history r.output r.attention
      p.keyHead p.keyTime p.valueHead p.valueTime p.historyHead p.historyKey p.T p.K p.V p.BT p.BK p.BV A.extent) :
    ∃ sf,exec (A.flattenKernel (program p r scale swapped)) (A.flattenState s)=some sf ∧
      (∀ i,active p (launch s) i → sf.readMem A.flat (A.addr r.output (outputAddress p (launch s) i))=
        formula p scale (launch s) (LocalReal.outputInputs r s) i) ∧
      BlockState.WriteWithin (RealOutputFlat.footprint A r.output p.valueHead p.valueTime p.T p.V p.BT p.BV s)
        (A.flattenState s) sf := by
  obtain ⟨t,ran,values,frame⟩ := LocalReal.output_executes p r scale s hu hinj
  have safety := LocalRealSafety.output p r scale s hu A.extent hb
  have he := A.exec_flatten hd hcov (program p r scale Bool.false) s safety (fragment p r scale) hu
  change exec (program p r scale Bool.false) s=some t at ran
  rw [ran,Option.map_some] at he
  refine ⟨A.flattenState t,?_,?_,?_⟩
  · cases swapped
    · exact he
    · exact candidate_executes_like p r scale A _ _ he
  · intro i hi
    have bound : outputAddress p (launch s) i<A.extent r.output := by
      simpa [outputAddress,launch,RealOutput.outOffset,RealOutput.tIndex,RealOutput.vIndex,Nat.mul_one] using hb.post.output i.1 i.2.1 hi
    have member : r.output∈A.regions := by
      by_contra absent
      rw [hcov _ absent] at bound
      exact Nat.not_lt_zero _ bound
    rw [A.flattenState_readMem hd t member bound]
    exact values i hi
  · apply RealOutputFlat.flatten_frame
    intro region off outside
    apply frame region off
    simpa only [LocalReal.outputFootprint,RealOutputIO.outputAddress,RealOutputIO.active,RealOutput.footprint,launch,
      RealOutput.outOffset,RealOutput.tIndex,RealOutput.vIndex,Nat.mul_one] using outside

#axiomsClean decomposition
#axiomsClean source_executes
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.OutputFlat
