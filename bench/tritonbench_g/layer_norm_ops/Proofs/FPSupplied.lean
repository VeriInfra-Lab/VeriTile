import bench.tritonbench_g.layer_norm_ops.Proofs.FPContract
namespace VeriTile.Bench.TritonBenchG.LayerNormOps.FPStages
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits (fold_writes_mem)
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

def vectorReference {α : Type} [Inhabited α] (pre out : RegionName) (stride N B : Nat)
    (s : State α) : State α :=
  (TileShape.allIndices [B]).foldl (fun t i => if i.1.val<N then
    t.write out (offset s stride i) (.mk .fp32 ((s.mem pre (offset s stride i)).read .fp32)) else t) s

theorem mean_store_run {α : Type} [Inhabited α] (M : Algebra α)
    (pre out : RegionName) (s : State α) :
    ∃ t, FP.Structural.exec M (meanStore .fp32 pre out) s=some t ∧
      t.mem=(s.write out (s.pids 0) (.mk .fp32 ((s.mem pre (s.pids 0)).read .fp32))).mem := by
  simp [FP.Structural.exec, meanStore, run, step, evalExpr, evalComputeOp, evalOp_unfold,
    evalOffsets, store, Option.bind, numeric, bop, Region.cast, FloatDType.toTileDType]
  rfl

theorem rstd_store_run {α : Type} [Inhabited α] (M : Algebra α)
    (pre out : RegionName) (s : State α) :
    ∃ t, FP.Structural.exec M (rstdStore .fp32 pre out) s=some t ∧
      t.mem=(s.write out (s.pids 0) (.mk .fp32 ((s.mem pre (s.pids 0)).read .fp32))).mem := by
  simp [FP.Structural.exec, rstdStore, run, step, evalExpr, evalComputeOp, evalOp_unfold,
    evalOffsets, store, Option.bind, numeric, bop, Region.cast, FloatDType.toTileDType]
  rfl

theorem vector_store_run {α : Type} [Inhabited α] (M : Algebra α)
    (pre out : RegionName) (stride N B : Nat) (s : State α) :
    ∃ t, FP.Structural.exec M (vectorStore .fp32 pre out stride N B) s=some t ∧
      t.mem=(vectorReference pre out stride N B s).mem := by
  simp [FP.Structural.exec, vectorStore, run, step, evalExpr, evalComputeOp, evalOp_unfold,
    evalOffsets, store, Option.bind, numeric, bop, numericLt, natLt, Region.cast,
    FloatDType.toTileDType, ofFloat, toFloat]
  simp only [vectorReference, fold_writes_mem, State.setReg_mem]
  simp (config := { contextual := Bool.true }) [offset]

theorem vector_value {α : Type} [Inhabited α] (pre out : RegionName) (stride N B : Nat)
    (s : State α) (i : TileIndex [B]) (hi : i.1.val<N) :
    (vectorReference pre out stride N B s).mem out (offset s stride i)=
      .mk .fp32 ((s.mem pre (offset s stride i)).read .fp32) := by
  apply State.masked_scatter_readback
  · exact hi
  · intro a b h
    have hh : a.1.val=b.1.val := Nat.add_left_cancel h
    rcases a with ⟨a,⟨⟩⟩
    rcases b with ⟨b,⟨⟩⟩
    simpa [Fin.ext_iff] using hh

theorem vector_frame {α : Type} [Inhabited α] (pre out : RegionName) (stride N B : Nat)
    (s : State α) (r : RegionName) (o : Nat)
    (h : r≠out ∨ ∀ i : TileIndex [B],i.1.val<N → o≠offset s stride i) :
    (vectorReference pre out stride N B s).mem r o=s.mem r o := by
  apply State.masked_scatter_frame
  rcases h with h | h
  · exact Or.inl h
  · exact Or.inr (fun i _ hi => h i hi)

#axiomsClean mean_store_run
#axiomsClean rstd_store_run
#axiomsClean vector_store_run
#axiomsClean vector_value
#axiomsClean vector_frame
end
end VeriTile.Bench.TritonBenchG.LayerNormOps.FPStages
