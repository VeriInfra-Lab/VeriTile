import bench.repairs.tritonbench_g.chunk_retention.Proofs.ForwardHSource

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.ForwardHSource
open VeriTile Triton FP.Structural FP.ProfiledRewrite KernelModel
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory masked_readback_active)
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def keyPointer (c : Shape) (s : State α) (n : Nat) : BlockPtr :=
  ⟨"k",s.pids 2*(c.T*c.K),[c.K,c.T],[c.BK,64],[1,c.K],[s.pids 0*c.BK,n*64]⟩
def valuePointer (c : Shape) (s : State α) (n : Nat) : BlockPtr :=
  ⟨"v",s.pids 2*(c.T*c.V),[c.T,c.V],[64,c.BV],[c.V,1],[n*64,s.pids 1*c.BV]⟩
structure Prepared (M : Algebra α) (c : Shape) (initial current : State α) (n : Nat)
    (carry : Index c → α) : Prop where
  context : Context M c initial current n
  carry : HasCarry c carry current
  pk : current.regs .blockPtr [c.BK,64] "p_k"=some (fun _ => keyPointer c initial n)
  pv : current.regs .blockPtr [64,c.BV] "p_v"=some (fun _ => valuePointer c initial n)
  it : current.regs .nat [] "i_t"=some (fun _ => n)

theorem preludePrepared (M : Algebra α) (D : DotPrimitive α) (c : Shape) (seed final : Bool)
    (initial current : State α) (n : Nat) (carry : Index c → α)
    (ctx : Context M c initial current n) (ha : HasCarry c carry current) :
    Prepared M c initial (preludeExecution M D c seed final initial current n carry ctx ha).val n carry := by
  constructor
  · constructor
    · rw [preludeMemory,stored_other M c initial current n carry "k" (by decide),ctx.key]
    · rw [preludeMemory,stored_other M c initial current n carry "v" (by decide),ctx.val]
    · simp [preludeExecution,ctx.ik]
    · simp [preludeExecution,ctx.iv]
    · simp [preludeExecution,ctx.bh]
    · simp [preludeExecution,ctx.bb]
    · simp [preludeExecution,ctx.oi]
    · intro hn; simp [preludeExecution,ctx.db hn]
    · intro hn; simp [preludeExecution,ctx.di hn]
  · simpa [HasCarry,preludeExecution] using ha
  · simp [preludeExecution,keyPointer]
  · simp [preludeExecution,valuePointer]
  · simp [preludeExecution]

def keyLoad (M : Algebra α) (c : Shape) (s : State α) (n : Nat) :
    TileIndex [c.BK,64] → α := fun i =>
  M.fp32Load (if feature c s i.1.val<c.K ∧ n*64+i.2.1.val<c.T then
    (s.mem "k" (s.pids 2*(c.T*c.K)+feature c s i.1.val+(n*64+i.2.1.val)*c.K)).read .real else zero M)
def valueLoad (M : Algebra α) (c : Shape) (s : State α) (n : Nat) :
    TileIndex [64,c.BV] → α := fun i =>
  M.fp32Load (if n*64+i.1.val<c.T ∧ column c s i.2.1.val<c.V then
    (s.mem "v" (s.pids 2*(c.T*c.V)+(n*64+i.1.val)*c.V+column c s i.2.1.val)).read .real else zero M)
def actualLength (c : Shape) (n : Nat) := if n=c.chunks-1 ∧ c.T%64≠0 then c.T%64 else 64
def gatedValue (M : Algebra α) (c : Shape) (s : State α) (n : Nat) :
    TileIndex [64,c.BV] → α := fun i =>
  M.cast (some .fp32) .real .real (M.binary (some .fp32) .real .mul
    (valueLoad M c s n i) (timeDecay M c s (actualLength c n) ⟨i.1,()⟩))
def update (M : Algebra α) (D : DotPrimitive α) (c : Shape) (s : State α)
    (n : Nat) (carry : Index c → α) (i : Index c) : α :=
  M.binary (some .fp32) .real .add
    (M.binary (some .fp32) .real .mul (blockDecay M c s (actualLength c n)) (carry i))
    (D.apply (some .fp32) .real .real .real {allowTF32:=some Bool.false} (batch:=[])
      (keyLoad M c s n) (gatedValue M c s n) i)

def suffixExecution (M : Algebra α) (D : DotPrimitive α) (c : Shape) (seed final : Bool)
    (initial current : State α) (n : Nat) (carry : Index c → α)
    (hn : n<c.chunks) (ctx : Prepared M c initial current n carry) :
    {t : State α // run (engine (withDot M D)) (suffix c seed final) current=some t} := by
  have ha := ctx.carry
  simp only [HasCarry] at ha
  by_cases last : n=c.chunks-1 ∧ c.T%64≠0
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp [suffix,parts,kernel,Kernels.forwardH,ComputeKernel.surfaceBody,Source.dynamicRange?,
      Source.prefixBefore,Option.map,Option.getD,run,step,evalExpr,evalComputeOp,evalOp_unfold,
      evalOffsets,boundaryZero,Option.bind,engine,withDot,Algebra.withDefaultPrecision,
      resolvePrecision,ComputeDType.eraseDType,numeric,integralFloorDiv,integralMod,numericEq,
      natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
      ctx.pk,ctx.pv,ctx.it,ctx.context.bb,ctx.context.oi,ctx.context.db hn,ctx.context.di hn,ha,last]
    rfl

theorem suffixMemory (M : Algebra α) (D : DotPrimitive α) (c : Shape) (seed final : Bool)
    (initial current : State α) (n : Nat) (carry : Index c → α)
    (hn : n<c.chunks) (ctx : Prepared M c initial current n carry) :
    (suffixExecution M D c seed final initial current n carry hn ctx).val.mem=current.mem := by
  by_cases last : n=c.chunks-1 ∧ c.T%64≠0 <;> simp [suffixExecution,last]

theorem suffixCarry (M : Algebra α) (D : DotPrimitive α) (c : Shape) (seed final : Bool)
    (initial current : State α) (n : Nat) (carry : Index c → α)
    (hn : n<c.chunks) (ctx : Prepared M c initial current n carry) :
    HasCarry c (update M D c initial n carry)
      (suffixExecution M D c seed final initial current n carry hn ctx).val := by
  dsimp only [HasCarry,ForwardHSource.update]
  by_cases last : n=c.chunks-1 ∧ c.T%64≠0 <;>
    simp [HasCarry,suffixExecution,last,update,gatedValue,keyLoad,valueLoad,actualLength,
      keyPointer,valuePointer,feature,column,blockDecay,timeDecay,KernelModel.decay,
      ctx.context.key,ctx.context.val,zero,bop,Broadcast.leftIndex,Broadcast.rightIndex,
      TileShape.dropInsertedIndex,BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,
      TileShape.indexToList,Function.comp_def]
  all_goals funext i
  all_goals
    rcases i with ⟨ei,ji,⟨⟩⟩
    dsimp [ForwardHSource.update,ForwardHSource.keyLoad,ForwardHSource.gatedValue,
      ForwardHSource.valueLoad,feature,column,blockDecay,timeDecay,KernelModel.decay,zero,
      bop,Broadcast.leftIndex,Broadcast.rightIndex]
    simp_all [actualLength,Int.natCast_mod]
    congr 2
    all_goals
      try funext j
      simp_all [keyLoad,gatedValue,valueLoad,feature,column,timeDecay,KernelModel.decay,
        actualLength,zero,Int.natCast_mod]
      all_goals split_ifs <;> simp_all

theorem suffixContext (M : Algebra α) (D : DotPrimitive α) (c : Shape) (seed final : Bool)
    (initial current : State α) (n : Nat) (carry : Index c → α)
    (hn : n<c.chunks) (ctx : Prepared M c initial current n carry) :
    Context M c initial (suffixExecution M D c seed final initial current n carry hn ctx).val (n+1) := by
  constructor
  · rw [suffixMemory,ctx.context.key]
  · rw [suffixMemory,ctx.context.val]
  · by_cases last : n=c.chunks-1 ∧ c.T%64≠0 <;> simp [suffixExecution,last,ctx.context.ik]
  · by_cases last : n=c.chunks-1 ∧ c.T%64≠0 <;> simp [suffixExecution,last,ctx.context.iv]
  · by_cases last : n=c.chunks-1 ∧ c.T%64≠0 <;> simp [suffixExecution,last,ctx.context.bh]
  · by_cases last : n=c.chunks-1 ∧ c.T%64≠0 <;> simp [suffixExecution,last,ctx.context.bb]
  · by_cases last : n=c.chunks-1 ∧ c.T%64≠0 <;> simp [suffixExecution,last,ctx.context.oi]
  · intro hs
    have last : ¬(n=c.chunks-1 ∧ c.T%64≠0) := by omega
    simp [suffixExecution,last,ctx.context.db hn]
  · intro hs
    have last : ¬(n=c.chunks-1 ∧ c.T%64≠0) := by omega
    simp [suffixExecution,last,ctx.context.di hn]

#specNonCircular update avoiding [run,FP.Structural.exec]
#axiomsClean suffixCarry
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.ForwardHSource
