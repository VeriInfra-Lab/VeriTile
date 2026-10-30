import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelGradientModel
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelGradient
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory)
open KernelBackward (Parameters PID Matrix)
open VeriTile.Bench.Repairs.ChunkLinearAttn (zero scalar cast add mul load rawLoad dot scatter scatter_frame transpose valueBlocks)
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
  nh : s.regs .int [] "n_bh"=some (fun _ => (heads:Int))

structure GradientFixed (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (input : Memory α) (scale : ℝ) (s : State α) : Prop where
  dq : s.regs .real [p.BT] "d_q"=some (fun i=>queryDecay M p pid scale i.1.val)
  dk : s.regs .real [p.BT] "d_k"=some (fun i=>keyDecay M p pid i.1.val)
  ds : s.regs .real [p.BT,p.BT] "d_s"=some (scoreDecay M p pid scale)
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
  simp [gradientParts,kernel,Source.backwardQKV,ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,Option.map,Option.getD,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,transposeValues,
    BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList]
  rfl

theorem gradientEntry_values (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ) (s : State α) (hBT : p.BT≤2^31) :
    let pid : PID := ⟨s.pids 0,s.pids 1,s.pids 2⟩
    let t := (gradientEntry M D p scale s).val
    GradientReady p pid (s.numPids 2) s.mem t ∧ GradientFixed M D p pid s.mem scale t ∧
      HasGrad p (gradInitial M p) t ∧ t.mem=s.mem := by
  dsimp only
  refine ⟨?_,?_,?_,?_⟩
  · constructor
    · simpa [gradientEntry] using GradientInputs.refl s.mem
    all_goals simp [gradientEntry] <;> rfl
  · have lane : ∀ i : TileIndex [p.BT],
        (if 2 * (((i.1.val:Int)%4294967296).toNat % 4294967296)<4294967296
          then max ((i.1.val:Int)%4294967296) 0 % 4294967296
          else max ((i.1.val:Int)%4294967296) 0 % 4294967296-4294967296)=(i.1.val:Int) :=
        fun i=>KernelBackward.signed_lane i.1.val (lt_of_lt_of_le i.1.isLt hBT)
    have one : (1.0:ℝ)=1 := by norm_num
    have minimum : (if p.BT<p.T-s.pids 1*p.BT then (p.BT:Int) else ((p.T-s.pids 1*p.BT : Nat):Int))=
        min (p.BT:Int) ((p.T-s.pids 1*p.BT : Nat):Int) := by split_ifs <;> omega
    constructor <;>
      simp [gradientEntry,gradQuery,gradKey,gradScore,queryDecay,keyDecay,scoreDecay,
        KernelBackward.weight,KernelBackward.logRate,load,rawLoad,zero,
        VeriTile.Bench.Repairs.ChunkLinearAttn.mul,VeriTile.Bench.Repairs.ChunkLinearAttn.cast,scalar,dot,transpose,
        FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
        TileShape.dropInsertedIndex,Function.comp_def,transposeValues,integralMod,one]
    all_goals
      funext i
      try rw [KernelBackward.signed_lane i.1.val (lt_of_lt_of_le i.1.isLt hBT)]
      try rw [KernelBackward.signed_lane i.2.1.val (lt_of_lt_of_le i.2.1.isLt hBT)]
      try unfold gradScore
      try unfold gradQuery
      try unfold gradKey
      try unfold VeriTile.Bench.Repairs.ChunkLinearAttn.load VeriTile.Bench.Repairs.ChunkLinearAttn.rawLoad
      try simp [gradScore,queryDecay,keyDecay,scoreDecay,KernelBackward.weight,KernelBackward.logRate,
        zero,VeriTile.Bench.Repairs.ChunkLinearAttn.mul,VeriTile.Bench.Repairs.ChunkLinearAttn.cast,scalar,dot,transpose,
        FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,one,minimum]
      try simp only [KernelBackward.signed_lane i.1.val (lt_of_lt_of_le i.1.isLt hBT)]
      try simp only [KernelBackward.signed_lane i.2.1.val (lt_of_lt_of_le i.2.1.isLt hBT)]
      try simp
      try rfl
  · constructor <;> simp [gradientEntry,gradInitial,zero]
  · simp [gradientEntry]

#axiomsClean gradientEntry_values
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelGradient
