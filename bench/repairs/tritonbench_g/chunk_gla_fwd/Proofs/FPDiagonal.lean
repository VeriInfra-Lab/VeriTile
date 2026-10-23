import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.IntraGrid
import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.SplitGrid
import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.FPMerge
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.FPDiagonal
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange range_values)
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory masked_readback_active masked_scatter_regs)
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false

structure Layout where
  T : Nat
  K : Nat
  BT : Nat
  BC : Nat
  BK : Nat
  NC : Nat

def kernel (split : Bool) (c : Layout) (scale : ℝ) :=
  if split then TypedSplit.original .fp32 .fp32 .fp32 "q" "k" "g" "scratch"
    (c.T*c.K) c.K scale c.T c.K c.BT c.BC c.BK c.NC
  else TypedIntra.original .fp32 .fp32 .fp32 "q" "k" "g" "attention"
    (c.T*c.K) c.K scale c.T c.K c.BT c.BC c.BK

def branch (split : Bool) (c : Layout) (scale : ℝ) : Source.Conditional :=
  (Source.conditional? (kernel split c scale).surfaceBody (fun _ => Bool.true)).getD
    ⟨[],.constBool Bool.false,[],[],[]⟩
def parts (split : Bool) (c : Layout) (scale : ℝ) : Source.DynamicRange :=
  (Source.dynamicRange? (branch split c scale).yes "j").getD
    ⟨[],.constNat 0,.constNat 0,.constNat 0,[],[]⟩
def out (split : Bool) : RegionName := if split then "scratch" else "attention"
abbrev Index (c : Layout) := TileIndex [c.BC]
variable {α : Type} [Inhabited α]

def chunk (split : Bool) (c : Layout) (s : State α) := if split then s.pids 1/c.NC else s.pids 0
def block (split : Bool) (c : Layout) (s : State α) := if split then s.pids 1%c.NC else s.pids 1
def feature (split : Bool) (c : Layout) (s : State α) (e : Nat) := (if split then s.pids 0*c.BK else 0)+e
def rowStart (split : Bool) (c : Layout) (s : State α) := chunk split c s*c.BT+block split c s*c.BC
def row (split : Bool) (c : Layout) (s : State α) (i : Nat) := rowStart split c s+i
def columns (split : Bool) (c : Layout) (s : State α) := min c.BC (c.T-chunk split c s*c.BT-block split c s*c.BC)
def outputBase (split : Bool) (c : Layout) (s : State α) (i : Index c) :=
  if split then (s.pids 2+s.pids 0*s.numPids 2)*c.T*c.BC+row split c s i.1.val*c.BC
  else s.pids 2*c.T*c.BT+row split c s i.1.val*c.BT+block split c s*c.BC
def outputAddress (split : Bool) (c : Layout) (s : State α) (j : Nat) (i : Index c) := outputBase split c s i+j
def keyAddress (split : Bool) (c : Layout) (s : State α) (j e : Nat) :=
  s.pids 2*(c.T*c.K)+rowStart split c s*c.K+feature split c s e+j*c.K
def zero (M : Algebra α) := M.literal (some .fp32) .real 0
def queryLoad (M : Algebra α) (split : Bool) (c : Layout) (s : State α) (r : RegionName)
    (i : TileIndex [c.BC,c.BK]) : α :=
  M.fp32Load (if row split c s i.1.val<c.T ∧ feature split c s i.2.1.val<c.K then
    (s.mem r (s.pids 2*(c.T*c.K)+row split c s i.1.val*c.K+feature split c s i.2.1.val)).read .real else zero M)
def keyLoad (M : Algebra α) (split : Bool) (c : Layout) (s : State α) (r : RegionName) (j e : Nat) : α :=
  M.cast (some .fp32) .real .real
    (if feature split c s e<c.K then (s.mem r (keyAddress split c s j e)).read .real else zero M)
def product (M : Algebra α) (split : Bool) (c : Layout) (s : State α) (j : Nat) (i : TileIndex [c.BC,c.BK]) : α :=
  M.binary (some .fp32) .real .mul
    (M.binary (some .fp32) .real .mul (queryLoad M split c s "q" i) (keyLoad M split c s "k" j i.2.1.val))
    (M.unary (some .fp32) .exp
      (M.binary (some .fp32) .real .sub (queryLoad M split c s "g" i) (keyLoad M split c s "g" j i.2.1.val)))
def value (M : Algebra α) (split : Bool) (c : Layout) (scale : ℝ) (s : State α) (j : Nat) (i : Index c) : α :=
  if j ≤ i.1.val then
    let reduced := M.reduceSum (some .fp32) (shape:=[c.BC,c.BK]) ⟨1,by simp⟩ Bool.false (product M split c s j) i
    let accumulated := if split then M.binary (some .fp32) .real .add (zero M) reduced else reduced
    M.binary (some .fp32) .real .mul accumulated (M.literal (some .fp32) .real scale)
  else zero M

structure Context (M : Algebra α) (split : Bool) (c : Layout) (initial current : State α) (j : Nat) : Prop where
  keyMemory : current.mem "k"=initial.mem "k"
  gateMemory : current.mem "g"=initial.mem "g"
  query : current.regs .real [c.BC,c.BK] "b_q"=some (queryLoad M split c initial "q")
  gate : current.regs .real [c.BC,c.BK] "b_g"=some (queryLoad M split c initial "g")
  key : current.regs .ptr [c.BK] "p_k"=some (fun (i : TileIndex [c.BK]) => (("k" : RegionName),keyAddress split c initial j i.1.val))
  gateKey : current.regs .ptr [c.BK] "p_gk"=some (fun (i : TileIndex [c.BK]) => (("g" : RegionName),keyAddress split c initial j i.1.val))
  featureMask : current.regs .bool [c.BK] "m_k"=some (fun i => decide (feature split c initial i.1.val<c.K))
  rowMask : current.regs .bool [c.BC] "m_A"=some (fun i => decide (row split c initial i.1.val<c.T))
  offsets : current.regs .nat [c.BC] "o_A"=some (outputBase split c initial)
  rowIndex : current.regs .nat [c.BC] "o_i"=some (fun i => i.1.val)

def entryExecution (M : Algebra α) (split : Bool) (c : Layout) (scale : ℝ) (s : State α) :
    {t : State α // run (engine M) ((branch split c scale).before++(parts split c scale).before) s=some t} := by
  cases split with
  | «false» =>
    refine ⟨?_,?_⟩
    swap
    simp [parts,branch,kernel,SourceBridge.intraNew,SourceBridge.splitNew,TypedIntra.original,TypedIntra.source,
      TypedSplit.original,TypedSplit.source,IntraGrid.layout,IntraGrid.regions,SplitGrid.layout,SplitGrid.regions,
      ComputeKernel.surfaceBody,Source.conditional?,Source.dynamicRange?,Source.prefixBefore,Option.map,Option.getD,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,
      Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg]
    rfl
  | «true» =>
    refine ⟨?_,?_⟩
    swap
    simp [parts,branch,kernel,SourceBridge.intraNew,SourceBridge.splitNew,TypedIntra.original,TypedIntra.source,
      TypedSplit.original,TypedSplit.source,IntraGrid.layout,IntraGrid.regions,SplitGrid.layout,SplitGrid.regions,
      ComputeKernel.surfaceBody,Source.conditional?,Source.dynamicRange?,Source.prefixBefore,Option.map,Option.getD,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,
      Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg]
    rfl

theorem entryContext (M : Algebra α) (split : Bool) (c : Layout) (scale : ℝ) (s : State α) :
    Context M split c s (entryExecution M split c scale s).val 0 := by
  cases split <;> constructor
  all_goals simp [entryExecution,queryLoad,feature,row,rowStart,chunk,block,outputBase,keyAddress,zero,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,BlockPtr.inBounds_2d_offsets,
    BlockPtr.address_2d_offsets,TileShape.indexToList]
  all_goals
    funext i
    simp [queryLoad,feature,row,rowStart,chunk,block,outputBase,keyAddress,zero,
      FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]

def bodyExecution (M : Algebra α) (split : Bool) (c : Layout) (scale : ℝ)
    (initial current : State α) (j : Nat) (ctx : Context M split c initial current j) :
    {t : State α // run (engine M) (parts split c scale).body
      (current.setReg "j" .nat [] (fun _ => j))=some t} := by
  cases split with
  | «false» =>
    refine ⟨?_,?_⟩
    swap
    simp [parts,branch,kernel,SourceBridge.intraNew,SourceBridge.splitNew,TypedIntra.original,TypedIntra.source,
      TypedSplit.original,TypedSplit.source,IntraGrid.layout,IntraGrid.regions,SplitGrid.layout,SplitGrid.regions,
      ComputeKernel.surfaceBody,Source.conditional?,Source.dynamicRange?,Source.prefixBefore,Option.map,Option.getD,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,
      Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
      ctx.keyMemory,ctx.gateMemory,ctx.query,ctx.gate,ctx.key,ctx.gateKey,ctx.featureMask,ctx.rowMask,ctx.offsets,ctx.rowIndex,
      FP.Structural.store]
    rfl
  | «true» =>
    refine ⟨?_,?_⟩
    swap
    simp [parts,branch,kernel,SourceBridge.intraNew,SourceBridge.splitNew,TypedIntra.original,TypedIntra.source,
      TypedSplit.original,TypedSplit.source,IntraGrid.layout,IntraGrid.regions,SplitGrid.layout,SplitGrid.regions,
      ComputeKernel.surfaceBody,Source.conditional?,Source.dynamicRange?,Source.prefixBefore,Option.map,Option.getD,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,
      Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
      ctx.keyMemory,ctx.gateMemory,ctx.query,ctx.gate,ctx.key,ctx.gateKey,ctx.featureMask,ctx.rowMask,ctx.offsets,ctx.rowIndex,
      FP.Structural.store]
    rfl

#specNonCircular value avoiding [run,FP.Structural.exec]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.FPDiagonal
