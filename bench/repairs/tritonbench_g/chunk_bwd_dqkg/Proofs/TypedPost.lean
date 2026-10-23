import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.TypedAccum
namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.Typed
open VeriTile Triton FP.Structural FP.ProfiledRewrite Whole
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def post (M : Algebra α) (dot : Dot α) (c : Config) (s : State α)
    (ready : Ready c s) :
    {t : State α // run (engine (withDot M dot)) ((parts c).after.take 17) s=some t} := by
  have hkey := Classical.choose_spec ready.key
  generalize hvkey : Classical.choose ready.key = vkey at hkey
  have htime := Classical.choose_spec ready.time
  generalize hvtime : Classical.choose ready.time = vtime at htime
  have hhead := Classical.choose_spec ready.head
  generalize hvhead : Classical.choose ready.head = vhead at hhead
  have hheads := Classical.choose_spec ready.heads
  generalize hvheads : Classical.choose ready.heads = vheads at hheads
  have hrows := Classical.choose_spec ready.rows
  generalize hvrows : Classical.choose ready.rows = vrows at hrows
  have hgate := Classical.choose_spec ready.gate
  generalize hvgate : Classical.choose ready.gate = vgate at hgate
  have hlast := Classical.choose_spec ready.last
  generalize hvlast : Classical.choose ready.last = vlast at hlast
  have hdq := Classical.choose_spec ready.dq
  generalize hvdq : Classical.choose ready.dq = vdq at hdq
  have hdk := Classical.choose_spec ready.dk
  generalize hvdk : Classical.choose ready.dk = vdk at hdk
  have hds := Classical.choose_spec ready.ds
  generalize hvds : Classical.choose ready.ds = vds at hds
  have hdgLast := Classical.choose_spec ready.dgLast
  generalize hvdgLast : Classical.choose ready.dgLast = vdgLast at hdgLast
  have hdg := Classical.choose_spec ready.dg
  generalize hvdg : Classical.choose ready.dg = vdg at hdg
  refine ⟨?_,?_⟩
  swap
  simp only [parts,kernel,TypedSource.original,TypedSource.specialize,TypedSource.Storage.dtype,
    ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,Option.map,Option.getD,
    String.reduceEq,beq_self_eq_true,↓reduceIte,List.take,List.drop]
  simp [run,step.eq_2,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,
    Option.bind,engine,withDot,Algebra.withDefaultPrecision,resolvePrecision,
    ComputeDType.eraseDType,numeric,integralFloorDiv,integralMod,natLe,numericLe,
    numericLt,natLt,bop,ofFloat,toFloat,TileShape.axisDim,TileShape.product,Region.cast,
    State.setReg,hkey,htime,hhead,hheads,hrows,hgate,hlast,hdq,hdk,hds,hdgLast,hdg]
  rfl

structure OutputReady (c : Config) (s : State α) : Prop where
  dq : ∃ v : Values α .real [64,c.BK],s.regs .real [64,c.BK] "b_dq"=some v
  dk : ∃ v : Values α .real [64,c.BK],s.regs .real [64,c.BK] "b_dk"=some v
  dg : ∃ v : Values α .real [64],s.regs .real [64] "b_dg"=some v
  pq : ∃ v : Values α .blockPtr [64,c.BK],s.regs .blockPtr [64,c.BK] "p_dq"=some v
  pk : ∃ v : Values α .blockPtr [64,c.BK],s.regs .blockPtr [64,c.BK] "p_dk"=some v
  pg : ∃ v : Values α .blockPtr [64],s.regs .blockPtr [64] "p_dg"=some v

theorem post_ready (M : Algebra α) (dot : Dot α) (c : Config) (s : State α)
    (ready : Ready c s) : OutputReady c (post M dot c s ready).val := by
  constructor <;> simp [post]

theorem post_memory (M : Algebra α) (dot : Dot α) (c : Config) (s : State α)
    (ready : Ready c s) : (post M dot c s ready).val.mem=s.mem := by simp [post]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.Typed
