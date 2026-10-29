import bench.tritonbench_g.destindex_copy.Proofs.Layout
import bench.tritonbench_g.kcache_copy_triton.Proofs.TypedCopy

namespace VeriTile.Bench.TritonBenchG.DestindexCopy.TypedBits
open VeriTile.Triton FP.Structural
open TypedSource (Storage)
open KcacheCopyTriton.TypedCopy (copyBackend scatter_memory)
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def tag : Storage → TileDType
  | .fp32 => .real | .fp16 => .fp16 | .bf16 => .bf16

def inputNope {α : Type} (p : Layout) (s : State α)
    (i : TileIndex [1,p.BLOCK_DMODEL_NOPE]) : Nat :=
  s.pids 0*p.stride_kv_nope_bs+p.stride_kv_nope_d*i.2.1.val

def inputRope {α : Type} (p : Layout) (s : State α)
    (i : TileIndex [1,p.BLOCK_DMODEL_ROPE]) : Nat :=
  s.pids 0*p.stride_kv_rope_bs+p.stride_kv_rope_d*i.2.1.val

def destination {α : Type} [Inhabited α] (p : Layout) (s : State α) : Int :=
  (s.mem p.Dest_loc.cast (s.pids 0)).read .int

def outputNope {α : Type} [Inhabited α] (p : Layout) (s : State α)
    (i : TileIndex [1,p.BLOCK_DMODEL_NOPE]) : Nat :=
  (destination p s*(p.stride_o_nope_bs : Int)+(p.stride_o_nope_d*i.2.1.val : Nat)).toNat

def outputRope {α : Type} [Inhabited α] (p : Layout) (s : State α)
    (i : TileIndex [1,p.BLOCK_DMODEL_ROPE]) : Nat :=
  (destination p s*(p.stride_o_rope_bs : Int)+(p.stride_o_rope_d*i.2.1.val : Nat)).toNat

def loadedNope {α : Type} [Inhabited α] (d : Storage) (p : Layout) (s : State α)
    (i : TileIndex [1,p.BLOCK_DMODEL_NOPE]) : Cell α :=
  .mk (tag d) ((s.mem p.KV_nope (inputNope p s i)).read (tag d))
def loadedRope {α : Type} [Inhabited α] (d : Storage) (p : Layout) (s : State α)
    (i : TileIndex [1,p.BLOCK_DMODEL_ROPE]) : Cell α :=
  .mk (tag d) ((s.mem p.KV_rope (inputRope p s i)).read (tag d))

/-- Both tile loads precede both stores. Source memory and destination metadata
come from the initial state, even when either output aliases an input. -/
def copiedNope {α : Type} [Inhabited α] (d : Storage) (p : Layout) (s : State α) : State α :=
  (TileShape.allIndices [1,p.BLOCK_DMODEL_NOPE]).foldl
    (fun t i => t.write p.O_nope (outputNope p s i) (loadedNope d p s i)) s

def copied {α : Type} [Inhabited α] (nd rd : Storage) (p : Layout) (s : State α) : State α :=
  (TileShape.allIndices [1,p.BLOCK_DMODEL_ROPE]).foldl
    (fun t i => t.write p.O_rope (outputRope p s i) (loadedRope rd p s i)) (copiedNope nd p s)

/-- Complete original typed source, all nine storage pairs, arbitrary layout,
metadata and initial state. Raw fp32 loading is the only overridden callback. -/
def execution {α : Type} [Inhabited α] (rest : Algebra α) (nd rd : Storage)
    (p : Layout) (s : State α) :
    {t : State α // FP.Structural.exec (copyBackend rest) (p.source nd rd Bool.false) s = some t} := by
  cases nd <;> cases rd
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp [Layout.source,TypedSource.kernel,ComputeKernel.surfaceBody,FP.Structural.exec,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,
      Option.bind,copyBackend,ComputeDType.eraseDType,numeric,bop,ofFloat,toFloat,
      Region.cast,FP.Structural.store,TileShape.dropInsertedIndex]
    try rfl

/-- Exact whole-memory contract has no output injectivity or separation premise.
The second scatter wins when outputs overlap; even zero widths are retained. -/
theorem execution_memory {α : Type} [Inhabited α] (rest : Algebra α)
    (nd rd : Storage) (p : Layout) (s : State α) :
    (execution rest nd rd p s).val.mem = (copied nd rd p s).mem := by
  cases nd <;> cases rd
  all_goals simp [execution,copied,copiedNope,scatter_memory,loadedNope,loadedRope,
    tag,inputNope,inputRope,outputNope,outputRope,destination,bop,
    Broadcast.leftIndex,Broadcast.rightIndex,Region.cast]

/-- Readback for the last store requires injectivity only for its own tile. -/
theorem rope_readback {α : Type} [Inhabited α] (rest : Algebra α)
    (nd rd : Storage) (p : Layout) (s : State α)
    (i : TileIndex [1,p.BLOCK_DMODEL_ROPE]) (hinj : Function.Injective (outputRope p s)) :
    (execution rest nd rd p s).val.mem p.O_rope (outputRope p s i) = loadedRope rd p s i := by
  rw [execution_memory]
  exact State.scatter_readback _ _ _ _ i hinj

theorem nope_readback {α : Type} [Inhabited α] (rest : Algebra α)
    (nd rd : Storage) (p : Layout) (s : State α)
    (i : TileIndex [1,p.BLOCK_DMODEL_NOPE]) (hinj : Function.Injective (outputNope p s))
    (hsep : p.O_nope ≠ p.O_rope) :
    (execution rest nd rd p s).val.mem p.O_nope (outputNope p s i) = loadedNope nd p s i := by
  rw [execution_memory]
  change ((TileShape.allIndices [1,p.BLOCK_DMODEL_ROPE]).foldl _ (copiedNope nd p s)).mem _ _ = _
  rw [State.scatter_frame _ _ _ _ _ _ (Or.inl hsep)]
  exact State.scatter_readback _ _ _ _ i hinj

theorem frame {α : Type} [Inhabited α] (rest : Algebra α)
    (nd rd : Storage) (p : Layout) (s : State α) (r : RegionName) (o : Nat)
    (hn : r ≠ p.O_nope ∨ ∀ i : TileIndex [1,p.BLOCK_DMODEL_NOPE], o ≠ outputNope p s i)
    (hr : r ≠ p.O_rope ∨ ∀ i : TileIndex [1,p.BLOCK_DMODEL_ROPE], o ≠ outputRope p s i) :
    (execution rest nd rd p s).val.mem r o = s.mem r o := by
  rw [execution_memory]
  change ((TileShape.allIndices [1,p.BLOCK_DMODEL_ROPE]).foldl _ (copiedNope nd p s)).mem r o = _
  rw [State.scatter_frame _ _ _ _ _ _ (hr.imp id (fun h i _ => h i))]
  exact State.scatter_frame _ _ _ _ _ _ (hn.imp id (fun h i _ => h i)) s

#axiomsClean execution
#axiomsClean execution_memory
#axiomsClean rope_readback
#axiomsClean nope_readback
#axiomsClean frame
end
end VeriTile.Bench.TritonBenchG.DestindexCopy.TypedBits
