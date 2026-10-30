import bench.repairs.tritonbench_g.chunk_gla_fwd.Kernels
import bench.tritonbench_g.parallel_attention.Proofs.FPRange
import bench.tritonbench_g.kcache_copy_triton.Proofs.TypedCopy
import VeriTile.Triton.Float.ProfiledSoundness

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.FPMerge
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange range_values)
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory masked_readback_active)
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false

structure Layout where
  T : Nat
  BT : Nat
  BC : Nat
  NK : Nat

def kernel (p : Layout) := TypedMerge.original .fp32 "scratch" "attention" p.T p.BT p.BC p.NK
def branch (p : Layout) : Source.Conditional :=
  (Source.conditional? (kernel p).surfaceBody (fun _ => Bool.true)).getD
    ⟨[],.constBool Bool.false,[],[],[]⟩
def parts (p : Layout) : Source.DynamicRange :=
  (Source.dynamicRange? (branch p).yes "i_k").getD
    ⟨[],.constNat 0,.constNat 0,.constNat 0,[],[]⟩
abbrev Index (p : Layout) := TileIndex [p.BC,p.BC]
variable {α : Type} [Inhabited α]

def row (p : Layout) (s : State α) (i : Nat) := s.pids 0*p.BT+s.pids 1*p.BC+i
def column (p : Layout) (s : State α) (j : Nat) := s.pids 1*p.BC+j
def inputAddress (p : Layout) (s : State α) (f : Nat) (i : Index p) :=
  (s.pids 2+f*s.numPids 2)*p.T*p.BC+row p s i.1.val*p.BC+i.2.1.val
def outputAddress (p : Layout) (s : State α) (i : Index p) :=
  s.pids 2*p.T*p.BT+row p s i.1.val*p.BT+column p s i.2.1.val
def enabled (p : Layout) (s : State α) : Prop := row p s 0<p.T
def active (p : Layout) (s : State α) (i : Index p) : Prop :=
  row p s i.1.val<p.T ∧ column p s i.2.1.val<p.BT
instance (p : Layout) (s : State α) : Decidable (enabled p s) := inferInstanceAs (Decidable (_<_))
instance (p : Layout) (s : State α) : DecidablePred (active p s) := fun _ => inferInstanceAs (Decidable (_∧_))
def zero (M : Algebra α) := M.literal (some .fp32) .real 0

def loaded (M : Algebra α) (p : Layout) (s : State α) (f : Nat) (i : Index p) : α :=
  if row p s i.1.val<p.T then (s.mem "scratch" (inputAddress p s f i)).read .real else zero M
def update (M : Algebra α) (p : Layout) (s : State α) (f : Nat) (a : Index p → α) (i : Index p) : α :=
  M.binary (some .fp32) .real .add (a i) (loaded M p s f i)
def accumulated (M : Algebra α) (p : Layout) (s : State α) : Index p → α :=
  foldRange (update M p s) 0 p.NK 1 (fun _ => zero M)
def value (M : Algebra α) (p : Layout) (s : State α) (i : Index p) : α :=
  M.cast (some .fp32) .real .fp32 (accumulated M p s i)

structure Context (p : Layout) (initial current : State α) : Prop where
  memory : current.mem=initial.mem
  it : current.regs .nat [] "i_t"=some (fun _ => initial.pids 0)
  ic : current.regs .nat [] "i_c"=some (fun _ => initial.pids 1)
  bh : current.regs .nat [] "i_bh"=some (fun _ => initial.pids 2)
  nh : current.regs .nat [] "n_bh"=some (fun _ => initial.numPids 2)
def HasAcc (p : Layout) (a : Index p → α) (s : State α) : Prop :=
  s.regs .real [p.BC,p.BC] "b_A"=some a

def entryExecution (M : Algebra α) (p : Layout) (s : State α) :
    {t : State α // run (engine M) ((branch p).before ++ (parts p).before) s=some t} := by
  refine ⟨?_,?_⟩
  swap
  simp [parts,branch,kernel,TypedMerge.original,ComputeKernel.surfaceBody,
    Source.conditional?,Source.dynamicRange?,Source.prefixBefore,Option.map,Option.getD,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg]
  rfl

theorem entryContext (M : Algebra α) (p : Layout) (s : State α) :
    Context p s (entryExecution M p s).val ∧
      HasAcc p (fun _ => zero M) (entryExecution M p s).val := by
  constructor
  · constructor <;> simp [entryExecution] <;> rfl
  · simp [HasAcc,entryExecution,zero]

def bodyExecution (M : Algebra α) (p : Layout) (initial current : State α)
    (f : Nat) (a : Index p → α) (ctx : Context p initial current) (ha : HasAcc p a current) :
    {t : State α // run (engine M) (parts p).body
      (current.setReg "i_k" .nat [] (fun _ => f))=some t} := by
  simp only [HasAcc] at ha
  refine ⟨?_,?_⟩
  swap
  simp [parts,branch,kernel,TypedMerge.original,ComputeKernel.surfaceBody,
    Source.conditional?,Source.dynamicRange?,Source.prefixBefore,Option.map,Option.getD,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,ctx.memory,ctx.it,ctx.ic,ctx.bh,ctx.nh,ha]
  rfl

theorem bodyContext (M : Algebra α) (p : Layout) (initial current : State α)
    (f : Nat) (a : Index p → α) (ctx : Context p initial current) (ha : HasAcc p a current) :
    let t := (bodyExecution M p initial current f a ctx ha).val
    Context p initial t ∧ HasAcc p (update M p initial f a) t := by
  dsimp only
  constructor
  · constructor <;> simp [bodyExecution,ctx.memory,ctx.it,ctx.ic,ctx.bh,ctx.nh]
  · simp [HasAcc,bodyExecution,update,loaded,inputAddress,row,zero,
      FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.FPMerge
