import bench.tritonbench_g.destindex_copy_kv1.Kernels
import bench.tritonbench_g.kcache_copy_triton.Sources.Mathematical_TypedCopy

namespace VeriTile.Bench.Repairs.DestindexCopyKv1.Tile
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def inputOffset {α : Type} (p : Layout) (s : State α)
    (i : TileIndex [p.BLOCK_HEAD,p.BLOCK_DMODEL]) : Nat :=
  s.pids 0*p.stride_k_bs+p.stride_k_h*i.1.val+p.stride_k_d*i.2.1.val

def destination {α : Type} [Inhabited α] (p : Layout) (s : State α) : Int :=
  (s.mem p.Dest_loc.cast (s.pids 0)).read .int

def outputOffset {α : Type} [Inhabited α] (p : Layout) (s : State α)
    (i : TileIndex [p.BLOCK_HEAD,p.BLOCK_DMODEL]) : Nat :=
  (destination p s*(p.stride_o_bs : Int)+
    (p.stride_o_h*i.1.val : Nat)+(p.stride_o_d*i.2.1.val : Nat)).toNat

abbrev active (p : Layout) (i : TileIndex [p.BLOCK_HEAD,p.BLOCK_DMODEL]) : Prop :=
  i.1.val < p.head_num ∧ i.2.1.val < p.head_dim

def loaded {α : Type} [Inhabited α] (p : Layout) (s : State α)
    (i : TileIndex [p.BLOCK_HEAD,p.BLOCK_DMODEL]) : Cell α :=
  .mk .real ((s.mem p.K (inputOffset p s i)).read .real)

/-- FP32 homogeneous memory copy is identity on raw payloads. The original
typed source's explicit same-storage cast is not a rounding operation. -/
def rawBackend {α : Type} (rest : Algebra α) : Algebra α :=
  { copyBackend rest with
    cast := fun precision src dst value =>
      if precision = some ComputeDType.fp32 ∧ src = .real ∧ dst = .real
      then value else rest.cast precision src dst value }

def copied {α : Type} [Inhabited α] (p : Layout) (s : State α) : State α :=
  (TileShape.allIndices [p.BLOCK_HEAD,p.BLOCK_DMODEL]).foldl
    (fun t i => if active p i then t.write p.Out (outputOffset p s i) (loaded p s i) else t) s

/-- The complete selected original source executes for arbitrary opaque words,
registers and masked input values; no output/execution callback is assumed. -/
def execution {α : Type} [Inhabited α] (rest : Algebra α) (p : Layout) (s : State α) :
    {t : State α // FP.Structural.exec (rawBackend rest) (kernel p) s=some t} := by
  refine ⟨?_,?_⟩
  swap
  simp [kernel,
    ComputeKernel.surfaceBody,FP.Structural.exec,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,evalOffsets,evalDeltas,Option.bind,copyBackend,rawBackend,natLt,
    ComputeDType.eraseDType,numeric,bop,ofFloat,toFloat,Region.cast,
    FP.Structural.store,TileShape.dropInsertedIndex]
  rfl

theorem execution_memory {α : Type} [Inhabited α] (rest : Algebra α)
    (p : Layout) (s : State α) : (execution rest p s).val.mem=(copied p s).mem := by
  simp (config := {contextual := Bool.true}) [execution,copied,masked_scatter_memory,
    loaded,active,inputOffset,outputOffset,destination,bop,
    Broadcast.leftIndex,Broadcast.rightIndex,Region.cast]

theorem readback {α : Type} [Inhabited α] (rest : Algebra α) (p : Layout) (s : State α)
    (i : TileIndex [p.BLOCK_HEAD,p.BLOCK_DMODEL]) (hi : active p i)
    (hinj : ∀ a b,active p a → active p b → outputOffset p s a=outputOffset p s b → a=b) :
    (execution rest p s).val.mem p.Out (outputOffset p s i)=loaded p s i := by
  rw [execution_memory]
  exact masked_readback_active _ _ _ _ _ i hi hinj

theorem frame {α : Type} [Inhabited α] (rest : Algebra α) (p : Layout) (s : State α)
    (r : RegionName) (o : Nat)
    (miss : r≠p.Out ∨ ∀ i,active p i → o≠outputOffset p s i) :
    (execution rest p s).val.mem r o=s.mem r o := by
  rw [execution_memory]
  unfold copied
  apply State.masked_scatter_frame
  exact miss.imp id (fun h i _ hi => h i hi)

#axiomsClean execution
#axiomsClean execution_memory
#axiomsClean readback
#axiomsClean frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DestindexCopyKv1.Tile
