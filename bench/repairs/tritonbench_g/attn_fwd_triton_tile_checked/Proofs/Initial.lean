import bench.repairs.tritonbench_g.attn_fwd_triton_tile_checked.Proofs.Iteration

namespace VeriTile.Bench.Repairs.AttnFwdTritonTileChecked.Model
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open AttnFwdTriton.TypedExecution Graph
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def fixed (p : Config) (BD : Nat) (memory : RegionName→Nat→Cell Expr) (pm ph : Nat) : Fixed BD where
  context := p.context
  keyStride := p.keyRow
  valueStride := p.valueRow
  startM := pm
  rows := fun i=>pm*128+i.1.val
  keys := fun i=>i.1.val
  keyBase := fun i=>(p.K,ph/p.heads*p.keyBatch+ph%p.heads*p.keyHead+i.1.val*p.keyDim+i.2.1.val*p.keyRow)
  scaleBase := (p.KS,ph*((p.context+64-1)/64))
  valueBase := fun i=>(p.V,ph/p.heads*p.valueBatch+ph%p.heads*p.valueHead+i.1.val*p.valueRow+i.2.1.val*p.valueDim)
  output := fun i=>(p.Out,outputAddress p BD pm ph i)
  query := fun i=>if decide (pm*128+i.1.val<p.context) && decide (i.2.1.val<p.headDim)
    then reads memory (p.Q,ph/p.heads*p.queryBatch+ph%p.heads*p.queryHead+
      (pm*128+i.1.val)*p.queryRow+i.2.1.val*p.queryDim) else .zero
  queryScale := reads memory (p.QS,ph*((p.context+128-1)/128)+pm)

theorem initial_exact (p : Config) (BD : Nat) (s : State Expr) :
    ∃ t, run (engine backend) ((code p BD).take 27) s=some t ∧
      Bound (fixed p BD s.mem (s.pids 0) (s.pids 1)) (initialCarry BD) t ∧ t.mem=s.mem := by
  have zero : (0.0 : ℝ)=0 := by norm_num
  have one : (1.0 : ℝ)=1 := by norm_num
  simp only [code,source,Bool.false_eq_true,↓reduceIte,
    AttnFwdTriton.TypedSource.repaired,ComputeKernel.surfaceBody,List.take_succ_cons,List.take_zero]
  simp [run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,integralFloorDiv,integralMod,bop,
    natLt,numericLt,ofFloat,toFloat,Region.cast,State.setReg,Bound,
    engine,Algebra.withDefaultPrecision,resolvePrecision,backend,
    fixed,initialCarry,reads,outputAddress,Graph.cast,nativeDType,consume,literal,zero,one]
  all_goals repeat' first | constructor | funext i | rfl
  all_goals try simp [bop,consume,nativeDType,literal,zero,one,funext_iff,reads]

  all_goals rfl

#axiomsClean initial_exact
end
end VeriTile.Bench.Repairs.AttnFwdTritonTileChecked.Model
