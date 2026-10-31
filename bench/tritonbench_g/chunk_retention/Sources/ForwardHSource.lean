import bench.tritonbench_g.chunk_retention.Sources.KernelModel

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.ForwardHSource
open VeriTile Triton FP.Structural FP.ProfiledRewrite KernelModel
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory masked_readback_active)
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def kernel (c : Shape) (seed final : Bool) :=
  Kernels.forwardH "k" "v" "h" "h0" "ht"
    (c.T*c.K) c.K 1 (c.T*c.V) c.V 1 (c.chunks*c.K*c.V) c.V
    c.H c.T c.K c.V 64 c.BK c.BV c.chunks seed final
def parts (c : Shape) (seed final : Bool) : Source.DynamicRange :=
  (Source.dynamicRange? (kernel c seed final).surfaceBody "i_t").getD
    ⟨[],.constNat 0,.constNat 0,.constNat 0,[],[]⟩
def prelude (c : Shape) (seed final : Bool) := (parts c seed final).body.take 4
def suffix (c : Shape) (seed final : Bool) := (parts c seed final).body.drop 4
abbrev Index (c : Shape) := TileIndex [c.BK,c.BV]
def feature (c : Shape) (s : State α) (e : Nat) := s.pids 0*c.BK+e
def column (c : Shape) (s : State α) (j : Nat) := s.pids 1*c.BV+j
def active (c : Shape) (s : State α) (i : Index c) : Prop :=
  feature c s i.1.val<c.K ∧ column c s i.2.1.val<c.V
instance (c : Shape) (s : State α) : DecidablePred (active c s) :=
  fun _ => inferInstanceAs (Decidable (_∧_))
def stateAddress (c : Shape) (s : State α) (i : Index c) :=
  s.pids 2*c.K*c.V+feature c s i.1.val*c.V+column c s i.2.1.val
def historyAddress (c : Shape) (s : State α) (n : Nat) (i : Index c) :=
  s.pids 2*(c.chunks*c.K*c.V)+n*c.K*c.V+feature c s i.1.val*c.V+column c s i.2.1.val
def initValue (M : Algebra α) (c : Shape) (seed : Bool) (s : State α) (i : Index c) : α :=
  if seed then M.cast (some .fp32) .real .real
    (if active c s i then (s.mem "h0" (stateAddress c s i)).read .real else zero M)
  else zero M
def blockDecay (M : Algebra α) (c : Shape) (s : State α) (n : Nat) : α :=
  M.unary (some .fp32) .exp2
    (M.binary (some .fp32) .real .mul
      (if n=64 then M.literal (some .fp32) .real 64 else M.fromNat (some .fp32) n)
      (rate M c (s.pids 2)))
@[simp] theorem remainder_ne_width (c : Shape) : c.T%64≠64 := by omega
def timeDecay (M : Algebra α) (c : Shape) (s : State α) (n : Nat) : TileIndex [64] → α :=
  fun i => KernelModel.decay M c (s.pids 2) ((n:Int)-i.1.val-1)
def HasCarry (c : Shape) (carry : Index c → α) (s : State α) : Prop :=
  s.regs .real [c.BK,c.BV] "b_h"=some carry
structure Context (M : Algebra α) (c : Shape) (initial current : State α) (n : Nat) : Prop where
  key : current.mem "k"=initial.mem "k"
  val : current.mem "v"=initial.mem "v"
  ik : current.regs .nat [] "i_k"=some (fun _ => initial.pids 0)
  iv : current.regs .nat [] "i_v"=some (fun _ => initial.pids 1)
  bh : current.regs .int [] "i_bh"=some (fun _ => (initial.pids 2 : Int))
  bb : current.regs .real [] "b_b"=some (fun _ => rate M c (initial.pids 2))
  oi : current.regs .int [64] "o_i"=some (fun i => (i.1.val:Int))
  db : n<c.chunks → current.regs .real [] "d_b"=some (fun _ => blockDecay M c initial 64)
  di : n<c.chunks → current.regs .real [64] "d_i"=some (timeDecay M c initial 64)

def entryExecution (M : Algebra α) (D : DotPrimitive α) (c : Shape) (seed final : Bool) (s : State α) :
    {t : State α // run (engine (withDot M D)) (parts c seed final).before s=some t} := by
  cases seed
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp [parts,kernel,Kernels.forwardH,ComputeKernel.surfaceBody,Source.dynamicRange?,
      Source.prefixBefore,Option.map,Option.getD,run,step,evalExpr,evalComputeOp,evalOp_unfold,
      evalOffsets,boundaryZero,Option.bind,engine,withDot,Algebra.withDefaultPrecision,
      resolvePrecision,ComputeDType.eraseDType,numeric,integralFloorDiv,integralMod,KernelModel.signed_head_mod,Int.toNat_mul,Int.toNat_add,Int.toNat_natCast,
      natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg]
    rfl

theorem entryContext (M : Algebra α) (D : DotPrimitive α) (c : Shape) (seed final : Bool) (s : State α) :
    Context M c s (entryExecution M D c seed final s).val 0 ∧
      HasCarry c (initValue M c seed s) (entryExecution M D c seed final s).val := by
  cases seed
  all_goals
    constructor
    · constructor
      all_goals
        try intro hn
        simp [entryExecution,blockDecay,timeDecay,KernelModel.decay,rate,zero,
          bop,Broadcast.leftIndex,Broadcast.rightIndex,TileShape.dropInsertedIndex,Function.comp_def]
        all_goals norm_num [blockDecay,timeDecay,KernelModel.decay,rate,zero]
        all_goals rfl
    · simp [HasCarry,entryExecution,initValue,active,stateAddress,feature,column,zero,
        BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList]
      try rfl

def stored (M : Algebra α) (c : Shape) (initial current : State α) (n : Nat) (carry : Index c → α) : State α :=
  (TileShape.allIndices [c.BK,c.BV]).foldl (fun t i =>
    if active c initial i then t.write "h" (historyAddress c initial n i)
      (.mk .real (M.cast (some .fp32) .real .real (carry i))) else t) current

theorem stored_other (M : Algebra α) (c : Shape) (initial current : State α) (n : Nat)
    (carry : Index c → α) (r : RegionName) (hr : r≠"h") :
    (stored M c initial current n carry).mem r=current.mem r := by
  funext off
  unfold stored
  apply State.masked_scatter_frame
  exact Or.inl hr

def preludeExecution (M : Algebra α) (D : DotPrimitive α) (c : Shape) (seed final : Bool)
    (initial current : State α) (n : Nat) (carry : Index c → α)
    (ctx : Context M c initial current n) (ha : HasCarry c carry current) :
    {t : State α // run (engine (withDot M D)) (prelude c seed final)
      (current.setReg "i_t" .nat [] (fun _ => n))=some t} := by
  simp only [HasCarry] at ha
  refine ⟨?_,?_⟩
  swap
  simp [prelude,parts,kernel,Kernels.forwardH,ComputeKernel.surfaceBody,Source.dynamicRange?,
    Source.prefixBefore,Option.map,Option.getD,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    evalOffsets,boundaryZero,Option.bind,engine,withDot,Algebra.withDefaultPrecision,
    resolvePrecision,ComputeDType.eraseDType,numeric,integralFloorDiv,integralMod,KernelModel.signed_head_mod,Int.toNat_mul,Int.toNat_add,Int.toNat_natCast,
    natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
    ctx.ik,ctx.iv,ctx.bh,KernelModel.signed_head_mod,Int.toNat_mul,Int.toNat_add,Int.toNat_natCast,ha,FP.Structural.store]
  rfl

theorem preludeMemory (M : Algebra α) (D : DotPrimitive α) (c : Shape) (seed final : Bool)
    (initial current : State α) (n : Nat) (carry : Index c → α)
    (ctx : Context M c initial current n) (ha : HasCarry c carry current) :
    (preludeExecution M D c seed final initial current n carry ctx ha).val.mem=
      (stored M c initial current n carry).mem := by
  simp [preludeExecution,stored,masked_scatter_memory,active,historyAddress,feature,column,
    bop,BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList]
  try rfl

#axiomsClean preludeMemory
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.ForwardHSource
