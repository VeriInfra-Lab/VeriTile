import bench.tritonbench_g.kldiv_ops.Proofs.HistoricalKernels
import bench.tritonbench_g.vector_addition.Proofs.Mathematical_PointwiseSchedule
import VeriTile.Triton.Float.ProfiledRewrite

namespace VeriTile.Bench.Repairs.KldivOps.Tile
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

@[simp] theorem active_int (start lane count : Nat) :
    ((start : Int)+(lane : Int)<(count : Int)) ↔ start+lane<count := by omega

def zero (M : Algebra α) := M.literal (some .fp32) .real 0

def scalar (M : Algebra α) (flag : Bool) (eps y t : α) : α :=
  let a := M.fp32Load y
  let b := M.fp32Load t
  if flag then M.binary (some .fp32) .real .mul
    (M.unary (some .fp32) .exp b) (M.binary (some .fp32) .real .sub b a)
  else M.binary (some .fp32) .real .mul b (M.binary (some .fp32) .real .sub
    (M.unary (some .fp32) .log (M.binary (some .fp32) .real .max b
      (M.cast (some .fp32) .real .real eps))) a)

structure Registers {α : Type} (p : Parameters) (red row : Nat) (eps carry : α) (s : State α) : Prop where
  y : s.regs .ptr [] "Y"=some (fun _ => (p.Y,row*p.cols))
  t : s.regs .ptr [] "T"=some (fun _ => (p.T,row*p.cols))
  o : s.regs .ptr [] "O"=some (fun _ => (p.O,row*(if red=0 then p.cols else 1)))
  base : s.regs .int [p.block] "base_offsets"=some (fun j => (j.1.val : Int))
  epsilon : s.regs .real [] "eps"=some (fun _ => eps)
  sum : s.regs .real [] "loss_sum"=some (fun _ => carry)

def bindEpsilon {α : Type} (eps : α) (s : State α) := s.setReg "eps" .real [] (fun _ => eps)

def prefixRun {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (flag : Bool)
    (red : Nat) (eps : α) (s : State α) :
    {t : State α // run (engine M) (forwardPrefix p flag red) (bindEpsilon eps s)=some t} := by
  refine ⟨?_,?_⟩
  swap
  cases flag <;>
    simp [forwardPrefix,forwardTemplate,KernelDSL.forward,KernelDSL.forwardDefault,KernelDSL.forwardLog,ComputeKernel.surfaceBody,run,step,evalExpr,
      evalComputeOp,evalOp_unfold,Option.bind,Algebra.withDefaultPrecision,resolvePrecision,
      ComputeDType.eraseDType,State.setReg,numeric,numericLt,natLt,FP.Structural.bop,Region.cast,bindEpsilon]
  all_goals rfl

theorem prefix_registers {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (flag : Bool) (red : Nat) (eps : α) (s : State α) :
    Registers p red (s.pids 0) eps (zero M) (prefixRun M p flag red eps s).val := by
  cases flag <;> constructor <;> simp [prefixRun,State.setReg,zero,Region.cast,FP.Structural.bop,Int.toNat_mul,engine,Algebra.withDefaultPrecision,resolvePrecision]
  all_goals
    funext i
    try dsimp [FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex]
    try split_ifs
    all_goals norm_num [Int.toNat_mul]

theorem prefix_memory {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (flag : Bool) (red : Nat) (eps : α) (s : State α) :
    (prefixRun M p flag red eps s).val.mem=s.mem := by cases flag <;> rfl

def lane {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (flag : Bool)
    (eps : α) (row start : Nat) (s : State α) (j : TileIndex [p.block]) : α :=
  let i := start+j.1.val
  let y := if i<p.cols then (s.mem p.Y (row*p.cols+i)).read .real else zero M
  let t := if i<p.cols then (s.mem p.T (row*p.cols+i)).read .real else zero M
  scalar M flag eps y t

def chunkSum {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (flag : Bool)
    (eps : α) (row start : Nat) (s : State α) : α :=
  M.reduceSum (some .fp32) ⟨0,by simp⟩ Bool.false
    (fun j : TileIndex [p.block] => if start+j.1.val<p.cols then
      lane M p flag eps row start s j else zero M) PUnit.unit

def nextCarry {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (flag : Bool)
    (red : Nat) (eps carry : α) (row start : Nat) (s : State α) : α :=
  if red=0 then carry else M.binary (some .fp32) .real .add carry (chunkSum M p flag eps row start s)

def chunkRun {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (flag : Bool)
    (red row start : Nat) (eps carry : α) (s : State α) (h : Registers p red row eps carry s) :
    {t : State α // run (engine M) (forwardBody p flag red)
      (s.setReg "i" .int [] (fun _ => (start : Int)))=some t} := by
  cases flag <;> by_cases hr : red=0
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp [forwardBody,forwardTemplateBody,forwardTemplate,KernelDSL.forward,KernelDSL.forwardDefault,KernelDSL.forwardLog,epsilonLoss,
    ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,State.setReg,
    numeric,numericLt,numericEq,natLt,FP.Structural.bop,store,Region.cast,hr,
    h.y,h.t,h.o,h.base,h.epsilon,h.sum]
    rfl


open VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs (masked_scatter_memory)

def chunkMemory {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (flag : Bool)
    (red row start : Nat) (eps : α) (s : State α) : RegionName→Nat→Cell α :=
  if red=0 then
    (TileShape.allIndices [p.block]).foldl (fun mem j =>
      if start+j.1.val<p.cols then fun r i =>
        if r=p.O ∧ i=row*p.cols+(start+j.1.val) then
          .mk .real (lane M p flag eps row start s j) else mem r i
      else mem) s.mem
  else s.mem

theorem chunk_memory {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (flag : Bool)
    (red row start : Nat) (eps carry : α) (s : State α) (h : Registers p red row eps carry s) :
    (chunkRun M p flag red row start eps carry s h).val.mem=chunkMemory M p flag red row start eps s := by
  cases flag <;> by_cases hr : red=0
  all_goals
    simp [chunkRun,chunkMemory,lane,scalar,zero,hr,masked_scatter_memory,
      State.setReg,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
      TileShape.dropInsertedIndex,Int.toNat_add,active_int,ofFloat,toFloat,engine,
      Algebra.withDefaultPrecision,resolvePrecision]
  all_goals norm_num

end
end VeriTile.Bench.Repairs.KldivOps.Tile
