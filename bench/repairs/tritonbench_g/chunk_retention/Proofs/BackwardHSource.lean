import bench.repairs.tritonbench_g.chunk_retention.Proofs.ForwardHReadback

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.BackwardHSource
open VeriTile Triton FP.Structural FP.ProfiledRewrite KernelModel
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory)
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange range_values)
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxRecDepth 30000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false

abbrev Index := ForwardHSource.Index
abbrev feature := @ForwardHSource.feature
abbrev column := @ForwardHSource.column
abbrev active := @ForwardHSource.active
abbrev stateAddress := @ForwardHSource.stateAddress
abbrev historyAddress := @ForwardHSource.historyAddress
abbrev blockDecay := @ForwardHSource.blockDecay
abbrev actualLength := ForwardHSource.actualLength

def kernel (c : Shape) (scale : ℝ) (terminal : Bool) :=
  Kernels.backwardH "q" "incoming" "dh" "terminal"
    (c.T*c.K) c.K 1 (c.T*c.V) c.V 1 (c.chunks*c.K*c.V) c.V
    scale c.H c.T c.K c.V 64 c.BK c.BV c.chunks terminal
def parts (c : Shape) (scale : ℝ) (terminal : Bool) : Source.DynamicRange :=
  (Source.dynamicRange? (kernel c scale terminal).surfaceBody "j").getD
    ⟨[],.constNat 0,.constNat 0,.constNat 0,[],[]⟩
def chunk (c : Shape) (n : Nat) := c.chunks-1-n
def timeDecay (M : Algebra α) (c : Shape) (s : State α) : TileIndex [64] → α :=
  fun i => KernelModel.decay M c (s.pids 2) ((i.1.val:Int)+1)
def initValue (M : Algebra α) (c : Shape) (terminal : Bool) (s : State α) (i : Index c) : α :=
  if terminal then M.cast (some .fp32) .real .real
    (if active c s i then (s.mem "terminal" (stateAddress c s i)).read .real else zero M)
  else zero M
def HasCarry (c : Shape) (carry : Index c → α) (s : State α) : Prop :=
  s.regs .real [c.BK,c.BV] "b_dh"=some carry
structure Context (M : Algebra α) (c : Shape) (initial current : State α) : Prop where
  query : current.mem "q"=initial.mem "q"
  incoming : current.mem "incoming"=initial.mem "incoming"
  ik : current.regs .nat [] "i_k"=some (fun _ => initial.pids 0)
  iv : current.regs .nat [] "i_v"=some (fun _ => initial.pids 1)
  bh : current.regs .nat [] "i_bh"=some (fun _ => initial.pids 2)
  bb : current.regs .real [] "b_b"=some (fun _ => rate M c (initial.pids 2))
  di : current.regs .real [64] "d_i"=some (timeDecay M c initial)

def entryExecution (M : Algebra α) (D : DotPrimitive α) (c : Shape)
    (scale : ℝ) (terminal : Bool) (s : State α) :
    {t : State α // run (engine (withDot M D)) (parts c scale terminal).before s=some t} := by
  cases terminal
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp [parts,kernel,Kernels.backwardH,ComputeKernel.surfaceBody,Source.dynamicRange?,
      Source.prefixBefore,Option.map,Option.getD,run,step,evalExpr,evalComputeOp,evalOp_unfold,
      evalOffsets,boundaryZero,Option.bind,engine,withDot,Algebra.withDefaultPrecision,
      resolvePrecision,ComputeDType.eraseDType,numeric,integralFloorDiv,integralMod,
      natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg]
    rfl

theorem entryContext (M : Algebra α) (D : DotPrimitive α) (c : Shape)
    (scale : ℝ) (terminal : Bool) (s : State α) :
    Context M c s (entryExecution M D c scale terminal s).val ∧
      HasCarry c (initValue M c terminal s) (entryExecution M D c scale terminal s).val := by
  cases terminal
  all_goals
    constructor
    · constructor <;>
        simp [entryExecution,timeDecay,KernelModel.decay,rate,zero,
          bop,Broadcast.leftIndex,Broadcast.rightIndex,TileShape.dropInsertedIndex,Function.comp_def]
      all_goals norm_num [timeDecay,KernelModel.decay,rate,zero]
      all_goals rfl
    · simp [HasCarry,entryExecution,initValue,active,stateAddress,feature,column,zero,
        ForwardHSource.active,ForwardHSource.stateAddress,ForwardHSource.feature,ForwardHSource.column,
        BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList]
      try rfl

def stored (M : Algebra α) (c : Shape) (initial current : State α)
    (n : Nat) (carry : Index c → α) : State α :=
  (TileShape.allIndices [c.BK,c.BV]).foldl (fun t i =>
    if active c initial i then t.write "dh" (historyAddress c initial (chunk c n) i)
      (.mk .real (M.cast (some .fp32) .real .real (carry i))) else t) current

def queryLoad (M : Algebra α) (c : Shape) (s : State α) (n : Nat) :
    TileIndex [c.BK,64] → α := fun i =>
  M.fp32Load (if feature c s i.1.val<c.K ∧ chunk c n*64+i.2.1.val<c.T then
    (s.mem "q" (s.pids 2*(c.T*c.K)+feature c s i.1.val+(chunk c n*64+i.2.1.val)*c.K)).read .real else zero M)
def incomingLoad (M : Algebra α) (c : Shape) (s : State α) (n : Nat) :
    TileIndex [64,c.BV] → α := fun i =>
  M.fp32Load (if chunk c n*64+i.1.val<c.T ∧ column c s i.2.1.val<c.V then
    (s.mem "incoming" (s.pids 2*(c.T*c.V)+(chunk c n*64+i.1.val)*c.V+column c s i.2.1.val)).read .real else zero M)
def gatedIncoming (M : Algebra α) (c : Shape) (s : State α) (scale : ℝ) (n : Nat) :
    TileIndex [64,c.BV] → α := fun i =>
  M.cast (some .fp32) .real .real (M.binary (some .fp32) .real .mul
    (M.binary (some .fp32) .real .mul (incomingLoad M c s n i) (timeDecay M c s ⟨i.1,()⟩))
    (M.literal (some .fp32) .real scale))
def update (M : Algebra α) (D : DotPrimitive α) (c : Shape) (s : State α)
    (scale : ℝ) (n : Nat) (carry : Index c → α) (i : Index c) : α :=
  M.binary (some .fp32) .real .add
    (M.binary (some .fp32) .real .mul (blockDecay M c s (actualLength c (chunk c n))) (carry i))
    (D.apply (some .fp32) .real .real .real {allowTF32:=some Bool.false} (batch:=[])
      (queryLoad M c s n) (gatedIncoming M c s scale n) i)

def bodyExecution (M : Algebra α) (D : DotPrimitive α) (c : Shape) (scale : ℝ) (terminal : Bool)
    (initial current : State α) (n : Nat) (carry : Index c → α)
    (ctx : Context M c initial current) (ha : HasCarry c carry current) :
    {t : State α // run (engine (withDot M D)) (parts c scale terminal).body
      (current.setReg "j" .nat [] (fun _ => n))=some t} := by
  simp only [HasCarry] at ha
  by_cases atEnd : c.chunks-1-n=c.chunks-1 <;> by_cases rem : c.T%64=0
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp [parts,kernel,Kernels.backwardH,ComputeKernel.surfaceBody,Source.dynamicRange?,
      Source.prefixBefore,Option.map,Option.getD,run,step,evalExpr,evalComputeOp,evalOp_unfold,
      evalOffsets,boundaryZero,Option.bind,engine,withDot,Algebra.withDefaultPrecision,
      resolvePrecision,ComputeDType.eraseDType,numeric,integralFloorDiv,integralMod,numericEq,
      natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
      ctx.ik,ctx.iv,ctx.bh,ctx.bb,ctx.di,ha,atEnd,rem,chunk,FP.Structural.store]
    rfl

theorem bodyMemory (M : Algebra α) (D : DotPrimitive α) (c : Shape) (scale : ℝ) (terminal : Bool)
    (initial current : State α) (n : Nat) (carry : Index c → α)
    (ctx : Context M c initial current) (ha : HasCarry c carry current) :
    (bodyExecution M D c scale terminal initial current n carry ctx ha).val.mem=
      (stored M c initial current n carry).mem := by
  by_cases atEnd : c.chunks-1-n=c.chunks-1 <;> by_cases rem : c.T%64=0 <;>
    simp [bodyExecution,atEnd,rem,stored,masked_scatter_memory,active,historyAddress,feature,column,
      ForwardHSource.active,ForwardHSource.historyAddress,ForwardHSource.feature,ForwardHSource.column,
      bop,BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList,FP.Structural.store,chunk]

theorem bodyContext (M : Algebra α) (D : DotPrimitive α) (c : Shape) (scale : ℝ) (terminal : Bool)
    (initial current : State α) (n : Nat) (carry : Index c → α)
    (ctx : Context M c initial current) (ha : HasCarry c carry current) :
    Context M c initial (bodyExecution M D c scale terminal initial current n carry ctx ha).val := by
  constructor
  · rw [bodyMemory]
    have hh : (stored M c initial current n carry).mem "q"=current.mem "q" := by
      funext off
      apply State.masked_scatter_frame
      exact Or.inl (by decide)
    rw [hh,ctx.query]
  · rw [bodyMemory]
    have hh : (stored M c initial current n carry).mem "incoming"=current.mem "incoming" := by
      funext off
      apply State.masked_scatter_frame
      exact Or.inl (by decide)
    rw [hh,ctx.incoming]
  all_goals by_cases atEnd : c.chunks-1-n=c.chunks-1 <;> by_cases rem : c.T%64=0 <;>
    simp [bodyExecution,atEnd,rem,ctx.ik,ctx.iv,ctx.bh,ctx.bb,ctx.di]

theorem bodyCarry (M : Algebra α) (D : DotPrimitive α) (c : Shape) (scale : ℝ) (terminal : Bool)
    (initial current : State α) (n : Nat) (carry : Index c → α)
    (ctx : Context M c initial current) (ha : HasCarry c carry current) :
    HasCarry c (update M D c initial scale n carry)
      (bodyExecution M D c scale terminal initial current n carry ctx ha).val := by
  have other {ι : Type} (l : List ι) (off : ι → Nat) (value : ι → Cell α)
      (enabled : ι → Prop) [DecidablePred enabled] (s : State α) (r : RegionName) (o : Nat)
      (hr : r≠"dh") :
      (l.foldl (fun t i => if enabled i then t.write "dh" (off i) (value i) else t) s).mem r o=
        s.mem r o := State.masked_scatter_frame _ _ _ _ _ _ _ (Or.inl hr) _
  dsimp only [HasCarry,BackwardHSource.update]
  by_cases atEnd : c.chunks-1-n=c.chunks-1 <;> by_cases rem : c.T%64=0 <;>
    simp [bodyExecution,atEnd,rem,queryLoad,incomingLoad,gatedIncoming,actualLength,other,
      blockDecay,ForwardHSource.actualLength,ForwardHSource.blockDecay,ctx.query,ctx.incoming,
      feature,column,ForwardHSource.feature,ForwardHSource.column,zero,chunk,
      bop,Broadcast.leftIndex,Broadcast.rightIndex,Function.comp_def,
      BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList]
  all_goals funext i
  all_goals
    rcases i with ⟨ei,ji,⟨⟩⟩
    dsimp [BackwardHSource.update,queryLoad,incomingLoad,gatedIncoming,actualLength,
      blockDecay,ForwardHSource.actualLength,ForwardHSource.blockDecay,
      feature,column,ForwardHSource.feature,ForwardHSource.column,zero,chunk,
      bop,Broadcast.leftIndex,Broadcast.rightIndex]
    simp [atEnd,rem,ForwardHSource.remainder_ne_width]
    congr 2
    all_goals
      try funext j
      simp_all [queryLoad,gatedIncoming,incomingLoad,feature,column,
        ForwardHSource.feature,ForwardHSource.column,zero,chunk]

#specNonCircular update avoiding [run,FP.Structural.exec]
#axiomsClean bodyMemory
#axiomsClean bodyCarry
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.BackwardHSource
