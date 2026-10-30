import bench.repairs.tritonbench_g.chunk_retention.Proofs.ForwardOResult
import bench.repairs.tritonbench_g.chunk_retention.Proofs.ForwardHReadback

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.BackwardQKVSource
open VeriTile Triton FP.Structural FP.ProfiledRewrite KernelModel
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory)
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxRecDepth 30000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false

def kernel (c : Shape) (scale : ℝ) :=
  Kernels.backwardQKV "q" "k" "v" "h" "incoming" "dh" "dq" "dk" "dv"
    (c.T*c.K) c.K 1 (c.T*c.V) c.V 1 (c.chunks*c.K*c.V) c.V
    scale c.H c.T c.K c.V 64 c.BK c.BV c.chunks
def parts (c : Shape) (scale : ℝ) : Source.DynamicRange :=
  (Source.dynamicRange? (kernel c scale).surfaceBody "i_v").getD
    ⟨[],.constNat 0,.constNat 0,.constNat 0,[],[]⟩
abbrev KeyIndex (c : Shape) := TileIndex [64,c.BK]
abbrev ValueIndex (c : Shape) := TileIndex [64,c.BV]
abbrev ScoreIndex := TileIndex [64,64]
def feature (c : Shape) (s : State α) (e : Nat) := s.pids 0*c.BK+e
def column (c : Shape) (f j : Nat) := f*c.BV+j
def row (s : State α) (i : Nat) := s.pids 1*64+i
def queryDecay (M : Algebra α) (c : Shape) (s : State α) (scale : ℝ) (i : TileIndex [64]) : α :=
  M.cast (some .fp32) .real .real (M.binary (some .fp32) .real .mul
    (KernelModel.decay M c (s.pids 2) ((i.1.val:Int)+1)) (M.literal (some .fp32) .real scale))
def keyDecay (M : Algebra α) (c : Shape) (s : State α) (i : TileIndex [64]) : α :=
  KernelModel.decay M c (s.pids 2) ((ForwardHSource.actualLength c (s.pids 1):Int)-i.1.val-1)
def scoreDecay (M : Algebra α) (c : Shape) (s : State α) (scale : ℝ) (i : ScoreIndex) : α :=
  M.binary (some .fp32) .real .mul
    (if i.2.1.val ≤ i.1.val then KernelModel.decay M c (s.pids 2) ((i.1.val:Int)-i.2.1.val) else zero M)
    (M.literal (some .fp32) .real scale)
def keyLoad (M : Algebra α) (c : Shape) (s : State α) (i : KeyIndex c) : α :=
  M.fp32Load (if row s i.1.val<c.T ∧ feature c s i.2.1.val<c.K then
    (s.mem "k" (s.pids 2*(c.T*c.K)+row s i.1.val*c.K+feature c s i.2.1.val)).read .real else zero M)
def queryLoad (M : Algebra α) (c : Shape) (s : State α) (i : TileIndex [c.BK,64]) : α :=
  M.fp32Load (if feature c s i.1.val<c.K ∧ row s i.2.1.val<c.T then
    (s.mem "q" (s.pids 2*(c.T*c.K)+feature c s i.1.val+row s i.2.1.val*c.K)).read .real else zero M)
def partialScore (M : Algebra α) (D : DotPrimitive α) (c : Shape) (s : State α)
    (scale : ℝ) (i : ScoreIndex) : α :=
  M.binary (some .fp32) .real .mul
    (D.apply (some .fp32) .real .real .real {allowTF32:=some Bool.false} (batch:=[])
      (keyLoad M c s) (queryLoad M c s) i) (scoreDecay M c s scale ⟨i.2.1,i.1,()⟩)
def InputEq (a b : State α) : Prop :=
  ∀ (r : RegionName), r∈(["q","k","v","h","incoming","dh"] : List RegionName) → a.mem r=b.mem r
structure Context (M : Algebra α) (D : DotPrimitive α) (c : Shape) (scale : ℝ)
    (initial current : State α) : Prop where
  memory : InputEq current initial
  ik : current.regs .nat [] "i_k"=some (fun _ => initial.pids 0)
  it : current.regs .nat [] "i_t"=some (fun _ => initial.pids 1)
  bh : current.regs .nat [] "i_bh"=some (fun _ => initial.pids 2)
  nh : current.regs .int [] "n_bh"=some (fun _ => (initial.numPids 2:Int))
  dq : current.regs .real [64] "d_q"=some (queryDecay M c initial scale)
  dk : current.regs .real [64] "d_k"=some (keyDecay M c initial)
  ds : current.regs .real [64,64] "d_s"=some (scoreDecay M c initial scale)
  query : current.regs .real [c.BK,64] "b_q"=some (queryLoad M c initial)
  key : current.regs .real [64,c.BK] "b_k"=some (keyLoad M c initial)
  score : current.regs .real [64,64] "b_s"=some (partialScore M D c initial scale)
abbrev Accum (c : Shape) (α : Type) := (KeyIndex c → α) × (KeyIndex c → α) × (ScoreIndex → α)
def HasAcc (c : Shape) (a : Accum c α) (s : State α) : Prop :=
  s.regs .real [64,c.BK] "b_dq"=some a.1 ∧
  s.regs .real [64,c.BK] "b_dk"=some a.2.1 ∧ s.regs .real [64,64] "b_ds"=some a.2.2

def entryExecution (M : Algebra α) (D : DotPrimitive α) (c : Shape) (scale : ℝ) (s : State α) :
    {t : State α // run (engine (withDot M D)) (parts c scale).before s=some t} := by
  by_cases last : s.pids 1=c.chunks-1 ∧ c.T%64≠0
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp [parts,kernel,Kernels.backwardQKV,ComputeKernel.surfaceBody,Source.dynamicRange?,
      Source.prefixBefore,Option.map,Option.getD,run,step,evalExpr,evalComputeOp,evalOp_unfold,
      evalOffsets,boundaryZero,Option.bind,engine,withDot,Algebra.withDefaultPrecision,
      resolvePrecision,ComputeDType.eraseDType,numeric,integralFloorDiv,integralMod,numericEq,
      natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,last]
    rfl

theorem entryMemory (M : Algebra α) (D : DotPrimitive α) (c : Shape) (scale : ℝ) (s : State α) :
    (entryExecution M D c scale s).val.mem=s.mem := by
  by_cases last : s.pids 1=c.chunks-1 ∧ c.T%64≠0 <;> simp [entryExecution,last]

theorem entryContext (M : Algebra α) (D : DotPrimitive α) (c : Shape) (scale : ℝ) (s : State α) :
    Context M D c scale s (entryExecution M D c scale s).val ∧
      HasAcc c ⟨fun _=>zero M,fun _=>zero M,fun _=>zero M⟩ (entryExecution M D c scale s).val := by
  have one : (1.0:ℝ)=1 := by norm_num
  constructor
  · constructor
    · intro r hr
      rw [entryMemory]
    all_goals by_cases last : s.pids 1=c.chunks-1 ∧ c.T%64≠0 <;>
      simp [entryExecution,last,queryDecay,keyDecay,scoreDecay,partialScore,keyLoad,queryLoad,
        ForwardHSource.actualLength,feature,column,row,KernelModel.decay,rate,zero,
        bop,Broadcast.leftIndex,Broadcast.rightIndex,TileShape.dropInsertedIndex,Function.comp_def,one,
        BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList]
    all_goals try rfl
    all_goals funext i
    all_goals simp_all [queryDecay,keyDecay,scoreDecay,partialScore,keyLoad,queryLoad,
      ForwardHSource.actualLength,feature,row,KernelModel.decay,rate,zero,one,Int.natCast_mod,
      bop,Broadcast.leftIndex,Broadcast.rightIndex,transposeValues]
    all_goals try rfl
    all_goals congr 2
    all_goals try funext j
    all_goals try simp_all [queryLoad,keyLoad,feature,row,zero]
    all_goals try (split_ifs <;> simp_all)
    all_goals rfl
  · by_cases last : s.pids 1=c.chunks-1 ∧ c.T%64≠0 <;>
      simp [HasAcc,entryExecution,last,zero]
    all_goals rfl

#axiomsClean entryMemory
#axiomsClean entryContext
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.BackwardQKVSource
