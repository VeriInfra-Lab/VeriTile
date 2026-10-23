import bench.repairs.tritonbench_g.index_select_cat.Kernels
import bench.tritonbench_g.kcache_copy_triton.Proofs.TypedCopy

namespace VeriTile.Bench.Repairs.IndexSelectCat.TypedGather
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def indexOffset {α : Type} (p : Layout) (s : State α)
    (i : TileIndex [p.BI,p.BC]) : Nat := s.pids 0*p.BI+i.1.val

def column {α : Type} (p : Layout) (s : State α)
    (i : TileIndex [p.BI,p.BC]) : Nat := s.pids 1*p.BC+i.2.1.val

def row {α : Type} [Inhabited α] (p : Layout) (s : State α)
    (i : TileIndex [p.BI,p.BC]) : Int :=
  (s.mem p.index (indexOffset p s i)).read .int

def inputOffset {α : Type} [Inhabited α] (p : Layout) (s : State α)
    (i : TileIndex [p.BI,p.BC]) : Nat :=
  (row p s i*(p.rowStride : Int)+(column p s i*p.colStride : Nat)).toNat

def outputOffset {α : Type} (p : Layout) (s : State α)
    (i : TileIndex [p.BI,p.BC]) : Nat :=
  indexOffset p s i*p.rowStride+column p s i*p.colStride

abbrev active {α : Type} (p : Layout) (s : State α)
    (i : TileIndex [p.BI,p.BC]) : Prop :=
  indexOffset p s i<p.rows ∧ column p s i<p.cols

def loaded {α : Type} [Inhabited α] (p : Layout) (s : State α)
    (i : TileIndex [p.BI,p.BC]) : Cell α :=
  .mk .real ((s.mem p.source (inputOffset p s i)).read .real)

def copied {α : Type} [Inhabited α] (p : Layout) (s : State α) : State α :=
  (TileShape.allIndices [p.BI,p.BC]).foldl
    (fun t i => if active p s i then t.write p.output (outputOffset p s i) (loaded p s i)
      else t) s

/-- Every original typed source statement executes for an arbitrary raw payload
carrier, arbitrary initial registers, and arbitrary unused numeric callbacks. -/
@[irreducible] def execution {α : Type} [Inhabited α] (rest : Algebra α)
    (p : Layout) (s : State α) :
    {t : State α // FP.Structural.exec (copyBackend rest) (kernel p) s = some t} := by
  refine ⟨?_,?_⟩
  swap
  simp [kernel,
    ComputeKernel.surfaceBody,FP.Structural.exec,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,
    Option.bind,copyBackend,natLt,ComputeDType.eraseDType,numeric,bop,
    ofFloat,toFloat,Region.cast,FP.Structural.store,TileShape.dropInsertedIndex]
  rfl

theorem execution_memory {α : Type} [Inhabited α] (rest : Algebra α)
    (p : Layout) (s : State α) :
    (execution rest p s).val.mem=(copied p s).mem := by
  simp (config := { contextual := Bool.true }) [execution,copied,masked_scatter_memory,
    loaded,active,inputOffset,outputOffset,row,indexOffset,column,bop,
    Broadcast.leftIndex,Broadcast.rightIndex,Region.cast,Int.natCast_mul,Int.natCast_add]

theorem frame {α : Type} [Inhabited α] (rest : Algebra α)
    (p : Layout) (s : State α) (r : RegionName) (o : Nat)
    (hmiss : r ≠ p.output ∨ ∀ i, active p s i → o ≠ outputOffset p s i) :
    (execution rest p s).val.mem r o=s.mem r o := by
  rw [execution_memory]
  unfold copied
  apply State.masked_scatter_frame
  exact hmiss.imp id (fun h i _ hi => h i hi)

#axiomsClean execution
#axiomsClean execution_memory
#axiomsClean frame
end
end VeriTile.Bench.Repairs.IndexSelectCat.TypedGather
