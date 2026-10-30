import bench.repairs.tritonbench_g.chunk_linear_attn.GradientModel
namespace VeriTile.Bench.Repairs.ChunkLinearAttn
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory)
noncomputable section
set_option maxHeartbeats 24000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

structure GradientReady (p : Parameters) (pid : PID) (heads : Nat) (input : Memory α) (s : State α) : Prop where
  inputs : GradientInputs s.mem input
  ik : s.regs .nat [] "i_k"=some (fun _ => pid.x)
  it : s.regs .nat [] "i_t"=some (fun _ => pid.y)
  bh : s.regs .nat [] "i_bh"=some (fun _ => pid.z)
  nh : s.regs .nat [] "n_bh"=some (fun _ => heads)
  oi : s.regs .nat [p.BT] "o_i"=some (fun i => i.1.val)
structure GradientFixed (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (input : Memory α) (scale : ℝ) (s : State α) : Prop where
  query : s.regs .real [p.BK,p.BT] "b_q"=some (gradQuery M p pid input)
  key : s.regs .real [p.BT,p.BK] "b_k"=some (gradKey M p pid input)
  score : s.regs .real [p.BT,p.BT] "b_s"=some (gradScore M D p pid input scale)
structure HasGrad (p : Parameters) (a : GradAcc p α) (s : State α) : Prop where
  query : s.regs .real [p.BT,p.BK] "b_dq"=some a.query
  key : s.regs .real [p.BT,p.BK] "b_dk"=some a.key
  score : s.regs .real [p.BT,p.BT] "b_ds"=some a.score

def gradientEntry (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ) (s : State α) :
    {t : State α // run (engine (withDot M D)) (gradientParts p scale).before s=some t} := by
  refine ⟨?_,?_⟩
  swap
  simp [gradientParts,backwardQKV,ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,transposeValues,
    BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList]
  rfl

theorem gradientEntry_values (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ) (s : State α) :
    let pid : PID := ⟨s.pids 0,s.pids 1,s.pids 2⟩
    let t := (gradientEntry M D p scale s).val
    GradientReady p pid (s.numPids 2) s.mem t ∧ GradientFixed M D p pid s.mem scale t ∧
      HasGrad p (gradInitial M p) t ∧ t.mem=s.mem := by
  dsimp only
  refine ⟨?_,?_,?_,?_⟩
  · constructor
    · simpa [gradientEntry] using GradientInputs.refl s.mem
    all_goals simp [gradientEntry] <;> rfl
  · constructor <;>
      simp [gradientEntry,gradQuery,gradKey,gradScore,load,rawLoad,zero,mul,scalar,dot,
        FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    all_goals
      funext i
      try unfold gradScore
      try unfold gradQuery
      try unfold gradKey
      unfold VeriTile.Bench.Repairs.ChunkLinearAttn.load VeriTile.Bench.Repairs.ChunkLinearAttn.rawLoad
      simp [gradScore,zero,mul,scalar,dot,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
  · constructor <;> simp [gradientEntry,gradInitial,zero]
  · simp [gradientEntry]

#axiomsClean gradientEntry_values
end
end VeriTile.Bench.Repairs.ChunkLinearAttn
