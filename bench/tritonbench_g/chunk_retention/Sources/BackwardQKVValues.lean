import bench.tritonbench_g.chunk_retention.Sources.BackwardQKVSource

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.BackwardQKVSource
open VeriTile Triton FP.Structural FP.ProfiledRewrite KernelModel
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory)
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxRecDepth 30000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false

def valueLoad (M : Algebra α) (c : Shape) (s : State α) (r : RegionName)
    (f : Nat) (i : ValueIndex c) : α :=
  M.fp32Load (if row s i.1.val<c.T ∧ column c f i.2.1.val<c.V then
    (s.mem r (s.pids 2*(c.T*c.V)+row s i.1.val*c.V+column c f i.2.1.val)).read .real else zero M)
def historyLoad (M : Algebra α) (c : Shape) (s : State α) (f : Nat) (i : TileIndex [c.BV,c.BK]) : α :=
  M.fp32Load (if column c f i.1.val<c.V ∧ feature c s i.2.1.val<c.K then
    (s.mem "h" (s.pids 2*(c.chunks*c.K*c.V)+s.pids 1*c.K*c.V+
      column c f i.1.val+feature c s i.2.1.val*c.V)).read .real else zero M)
def dhistoryLoad (M : Algebra α) (c : Shape) (s : State α) (f : Nat) (i : TileIndex [c.BK,c.BV]) : α :=
  M.fp32Load (if feature c s i.1.val<c.K ∧ column c f i.2.1.val<c.V then
    (s.mem "dh" (s.pids 2*(c.chunks*c.K*c.V)+s.pids 1*c.K*c.V+
      feature c s i.1.val*c.V+column c f i.2.1.val)).read .real else zero M)
def update (M : Algebra α) (D : DotPrimitive α) (c : Shape) (s : State α) (f : Nat)
    (a : Accum c α) : Accum c α :=
  ⟨fun i => M.binary (some .fp32) .real .add (a.1 i)
      (D.apply (some .fp32) .real .real .real {allowTF32:=some Bool.false} (batch:=[])
        (valueLoad M c s "incoming" f) (historyLoad M c s f) i),
   fun i => M.binary (some .fp32) .real .add (a.2.1 i)
      (D.apply (some .fp32) .real .real .real {allowTF32:=some Bool.false} (batch:=[])
        (valueLoad M c s "v" f) (fun (j : TileIndex [c.BV,c.BK]) => dhistoryLoad (α:=α) M c s f ⟨j.2.1,j.1,()⟩) i),
   fun i => M.binary (some .fp32) .real .add (a.2.2 i)
      (D.apply (some .fp32) .real .real .real {allowTF32:=some Bool.false} (batch:=[])
        (valueLoad M c s "incoming" f) (fun (j : TileIndex [c.BV,64]) => valueLoad (α:=α) M c s "v" f ⟨j.2.1,j.1,()⟩) i)⟩
def partialValue (M : Algebra α) (D : DotPrimitive α) (c : Shape) (s : State α)
    (scale : ℝ) (f : Nat) (i : ValueIndex c) : α :=
  M.cast (some .fp32) .real .real (M.binary (some .fp32) .real .add
    (M.binary (some .fp32) .real .mul
      (D.apply (some .fp32) .real .real .real {allowTF32:=some Bool.false} (batch:=[])
        (keyLoad M c s) (dhistoryLoad M c s f) i) (keyDecay M c s ⟨i.1,()⟩))
    (D.apply (some .fp32) .real .real .real {allowTF32:=some Bool.false} (batch:=[])
      (fun j => M.cast (some .fp32) .real .real (partialScore M D c s scale j))
      (valueLoad M c s "incoming" f) i))
def valueActive (c : Shape) (s : State α) (f : Nat) (i : ValueIndex c) : Prop :=
  row s i.1.val<c.T ∧ column c f i.2.1.val<c.V
instance (c : Shape) (s : State α) (f : Nat) : DecidablePred (valueActive c s f) :=
  fun _ => inferInstanceAs (Decidable (_∧_))
def partialAddress (c : Shape) (s : State α) (f : Nat) (i : ValueIndex c) : Nat :=
  (s.pids 0*s.numPids 2+s.pids 2)*(c.T*c.V)+row s i.1.val*c.V+column c f i.2.1.val
def storedPartial (M : Algebra α) (D : DotPrimitive α) (c : Shape) (initial current : State α)
    (scale : ℝ) (f : Nat) : State α :=
  (TileShape.allIndices [64,c.BV]).foldl (fun t i =>
    if valueActive c initial f i then t.write "dv" (partialAddress c initial f i)
      (.mk .real (partialValue M D c initial scale f i)) else t) current

def bodyExecution (M : Algebra α) (D : DotPrimitive α) (c : Shape) (scale : ℝ)
    (initial current : State α) (f : Nat) (a : Accum c α)
    (ctx : Context M D c scale initial current) (ha : HasAcc c a current) :
    {t : State α // run (engine (withDot M D)) (parts c scale).body
      (current.setReg "i_v" .nat [] (fun _ => f))=some t} := by
  simp only [HasAcc] at ha
  refine ⟨?_,?_⟩
  swap
  simp [parts,kernel,Kernels.backwardQKV,ComputeKernel.surfaceBody,Source.dynamicRange?,
    Source.prefixBefore,Option.map,Option.getD,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    evalOffsets,boundaryZero,Option.bind,engine,withDot,Algebra.withDefaultPrecision,
    resolvePrecision,ComputeDType.eraseDType,numeric,integralFloorDiv,integralMod,numericEq,
    natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
    ctx.ik,ctx.it,ctx.bh,ctx.nh,ctx.dq,ctx.dk,ctx.ds,ctx.query,ctx.key,ctx.score,ha.1,ha.2.1,ha.2.2,
    FP.Structural.store]
  rfl

theorem bodyMemory (M : Algebra α) (D : DotPrimitive α) (c : Shape) (scale : ℝ)
    (initial current : State α) (f : Nat) (a : Accum c α)
    (ctx : Context M D c scale initial current) (ha : HasAcc c a current) :
    (bodyExecution M D c scale initial current f a ctx ha).val.mem=
      (storedPartial M D c initial current scale f).mem := by
  simp [bodyExecution,storedPartial,masked_scatter_memory,valueActive,partialAddress,partialValue,
    valueLoad,dhistoryLoad,column,feature,row,zero,ctx.memory "incoming" (by simp),
    ctx.memory "dh" (by simp),bop,Broadcast.leftIndex,Broadcast.rightIndex,
    BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList]
  rfl

theorem bodyContext (M : Algebra α) (D : DotPrimitive α) (c : Shape) (scale : ℝ)
    (initial current : State α) (f : Nat) (a : Accum c α)
    (ctx : Context M D c scale initial current) (ha : HasAcc c a current) :
    Context M D c scale initial (bodyExecution M D c scale initial current f a ctx ha).val := by
  constructor
  · intro r hr
    rw [bodyMemory]
    have hn : r≠"dv" := by intro he; subst r; simp at hr
    have hm : (storedPartial M D c initial current scale f).mem r=current.mem r := by
      funext off
      apply State.masked_scatter_frame
      exact Or.inl hn
    rw [hm,ctx.memory r hr]
  all_goals simp [bodyExecution,ctx.ik,ctx.it,ctx.bh,ctx.nh,ctx.dq,ctx.dk,ctx.ds,ctx.query,ctx.key,ctx.score]

theorem bodyAcc (M : Algebra α) (D : DotPrimitive α) (c : Shape) (scale : ℝ)
    (initial current : State α) (f : Nat) (a : Accum c α)
    (ctx : Context M D c scale initial current) (ha : HasAcc c a current) :
    HasAcc c (update M D c initial f a) (bodyExecution M D c scale initial current f a ctx ha).val := by
  simp [HasAcc,bodyExecution,update,valueLoad,historyLoad,dhistoryLoad,column,feature,row,zero,
    ctx.memory "incoming" (by simp),ctx.memory "v" (by simp),ctx.memory "h" (by simp),ctx.memory "dh" (by simp),
    bop,Broadcast.leftIndex,Broadcast.rightIndex,Function.comp_def,
    BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList]
  constructor
  · rfl
  · constructor <;> rfl

#specNonCircular update avoiding [run,FP.Structural.exec]
#specNonCircular partialValue avoiding [run,FP.Structural.exec]
#axiomsClean bodyMemory
#axiomsClean bodyAcc
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.BackwardQKVSource
