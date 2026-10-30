import bench.tritonbench_g.llama_ff_triton.Proofs.Epilogue

namespace VeriTile.Bench.Repairs.LlamaFFScalarCast.Execution
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory masked_readback_active)
open scoped Classical
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 10000000
set_option linter.unusedSimpArgs false

structure StoreContext {α : Type} (p : Parameters) (initial current : State α) : Prop where
  memory : current.mem=initial.mem
  m : current.regs .int [] "M64"=some (fun _ => (p.rows : Int))
  n : current.regs .int [] "N64"=some (fun _ => (p.columns : Int))
  pidm : current.regs .int [] "pid_m"=some (fun _ => (rowBlock p initial : Int))
  pidn : current.regs .int [] "pid_n"=some (fun _ => (columnBlock p initial : Int))
  outm : current.regs .int [] "soutm64"=some (fun _ => (p.columns : Int))
  outn : current.regs .int [] "soutn64"=some (fun _ => (1 : Int))

def Indices.storeContext {α : Type} {p : Parameters} {initial current : State α} {k : Nat}
    (h : Indices p initial k current) : StoreContext p initial current :=
  ⟨h.memory,h.m,h.n,h.pidm,h.pidn,h.outm,h.outn⟩

def storeExecution {α : Type} [Inhabited α] (a : Arithmetic α)
    (p : Parameters) (initial current : State α) (values : TileIndex [16,16] → α)
    (h : StoreContext p initial current)
    (hv : current.regs .real [16,16] "accumulator"=some values) :
    {t : State α // run a.model ((epilogue p).drop 5) current=some t} := by
  have hep : (epilogue p).drop 5=(epilogue {p with byte:=Bool.false}).drop 5 := by
    rcases p with ⟨M,N,K,fp8,eps⟩
    cases fp8 <;> rfl
  refine ⟨?_,?_⟩
  swap
  rw [hep]
  simp [epilogue,source,Source.kernel,Source.half,ComputeKernel.surfaceBody,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,numeric,
    FP.Structural.bop,natLt,numericLt,ComputeDType.eraseDType,ofFloat,toFloat,
    Region.cast,Option.bind,integralFloorDiv,integralMod,
    h.pidm,h.pidn,h.memory,h.m,h.n,h.outm,h.outn,hv,Arithmetic.model]
  rfl

def writtenValues {α : Type} [Inhabited α] (a : Arithmetic α)
    (p : Parameters) (initial : State α) (values : TileIndex [16,16] → α) : State α :=
  (TileShape.allIndices [16,16]).foldl (fun t i =>
    if active p initial i then t.write "OUT" (outputOffset p initial i)
      (.mk .fp16 (a.cast none .real .fp16 (values i))) else t) initial

theorem storeMemory {α : Type} [Inhabited α] (a : Arithmetic α)
    (p : Parameters) (initial current : State α) (values : TileIndex [16,16] → α)
    (h : StoreContext p initial current)
    (hv : current.regs .real [16,16] "accumulator"=some values) :
    (storeExecution a p initial current values h hv).val.mem=(writtenValues a p initial values).mem := by
  simp +contextual [storeExecution,writtenValues,masked_scatter_memory,h.memory,
    active,outputOffset,outputRow,outputColumn,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
    and_assoc,and_comm,and_left_comm]
  congr 1
  funext mem i
  rcases i with ⟨i,j,⟨⟩⟩
  have castTile (x y : Nat) : (x : Int)*16+(y : Int)=((x*16+y : Nat) : Int) := by simp
  simp only [castTile,Int.ofNat_lt]
  have castOffset (n x y : Nat) : ((n : Int)*(x : Int)+(y : Int)).toNat=n*x+y := by norm_cast
  simp only [castOffset]

theorem storePost {α : Type} [Inhabited α] (a : Arithmetic α)
    (p : Parameters) (initial current : State α) (values : TileIndex [16,16] → α)
    (h : StoreContext p initial current)
    (hv : current.regs .real [16,16] "accumulator"=some values) :
    Post p initial (storeExecution a p initial current values h hv).val
      (fun i => a.cast none .real .fp16 (values i)) := by
  constructor
  · intro i hi
    rw [storeMemory]
    exact masked_readback_active _ _ _ _ _ i hi (active_injective p initial)
  · intro r o outside
    rw [storeMemory]
    exact State.masked_scatter_frame _ _ _ _ _ _ _ (outside.imp id (fun h i _ => h i)) _

#axiomsClean storeMemory
#axiomsClean storePost
end
end VeriTile.Bench.Repairs.LlamaFFScalarCast.Execution
