import bench.repairs.tritonbench_g.llama_ff_triton_scalar_cast.Proofs.Output
import bench.tritonbench_g.kcache_copy_triton.Proofs.TypedCopy

namespace VeriTile.Bench.Repairs.LlamaFFScalarCast.Execution
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory masked_readback_active)
open scoped Classical
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 12000000
set_option linter.unusedSimpArgs false

def epilogueExecution {α : Type} [Inhabited α] (a : Arithmetic α)
    (p : Parameters) (initial current : State α)
    (h : Invariant a Bool.false p initial (steps p) current) :
    {t : State α // run a.model (epilogue p) current = some t} := by
  have hep : epilogue p=epilogue {p with byte:=Bool.false} := by
    rcases p with ⟨M,N,K,fp8,eps⟩
    cases fp8 <;> rfl
  refine ⟨?_, ?_⟩
  swap
  rw [hep]
  simp [epilogue,source,Source.kernel,Source.half,ComputeKernel.surfaceBody,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,numeric,
    FP.Structural.bop,natLt,numericLt,ComputeDType.eraseDType,ofFloat,toFloat,
    Region.cast,Option.bind,integralFloorDiv,integralMod,
    h.pidm,h.pidn,h.acc1,h.acc2,h.sum,h.memory,h.m,h.n,h.k,h.outm,h.outn,
    Arithmetic.model]
  rfl

def written {α : Type} [Inhabited α] (a : Arithmetic α)
    (p : Parameters) (initial : State α) : State α :=
  (TileShape.allIndices [16,16]).foldl (fun t i =>
    if active p initial i then t.write "OUT" (outputOffset p initial i)
      (.mk .fp16 (sourceOutput a p initial i)) else t) initial

theorem epilogueMemory {α : Type} [Inhabited α] (a : Arithmetic α)
    (p : Parameters) (initial current : State α)
    (h : Invariant a Bool.false p initial (steps p) current) :
    (epilogueExecution a p initial current h).val.mem=(written a p initial).mem := by
  simp +contextual [epilogueExecution,written,masked_scatter_memory,h.memory,
    active,outputOffset,outputRow,outputColumn,sourceOutput,projection,normalization,
    Arithmetic.model,reduction,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
    TileShape.insertAxisIndex,TileShape.axisDim,TileShape.eraseAxis,
    and_assoc,and_comm,and_left_comm]
  congr 1
  funext mem i
  rcases i with ⟨i,j,⟨⟩⟩
  have castTile (x y : Nat) : (x : Int)*16+(y : Int)=((x*16+y : Nat) : Int) := by
    simp
  simp only [castTile,Int.ofNat_lt]
  have castOffset (n x y : Nat) : ((n : Int)*(x : Int)+(y : Int)).toNat=n*x+y := by
    norm_cast
  simp only [castOffset]

def Post {α : Type} (p : Parameters) (initial final : State α)
    (values : TileIndex [16,16] → α) : Prop :=
  (∀ i,active p initial i → final.mem "OUT" (outputOffset p initial i)=.mk .fp16 (values i)) ∧
  (∀ r o,(r≠("OUT" : RegionName) ∨ ∀ i,active p initial i → o≠outputOffset p initial i) →
    final.mem r o=initial.mem r o)

theorem epiloguePost {α : Type} [Inhabited α] (a : Arithmetic α)
    (p : Parameters) (initial current : State α)
    (h : Invariant a Bool.false p initial (steps p) current) :
    Post p initial (epilogueExecution a p initial current h).val (sourceOutput a p initial) := by
  constructor
  · intro i hi
    rw [epilogueMemory]
    exact masked_readback_active _ _ _ _ _ i hi (active_injective p initial)
  · intro r o outside
    rw [epilogueMemory]
    exact State.masked_scatter_frame _ _ _ _ _ _ _ (outside.imp id (fun h i _ => h i)) _

#axiomsClean epilogueMemory
#axiomsClean epiloguePost
end
end VeriTile.Bench.Repairs.LlamaFFScalarCast.Execution
