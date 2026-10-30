import bench.tritonbench_g.parallel_attention.NativeReference

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.ParallelAttention
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000

def SameInputs (s t : State α) : Prop :=
  ∀ (R : RegionName),R∈(["q","k","v","do","dz"] : List RegionName) → s.mem R=t.mem R

theorem sameReaders {α : Type} [Inhabited α] (shape : Shape) (s t : State α) (h : SameInputs s t) :
    inputReaders shape s=inputReaders shape t := by
  have hq:=h "q" (by simp)
  have hk:=h "k" (by simp)
  have hv:=h "v" (by simp)
  have hd:=h "do" (by simp)
  have hz:=h "dz" (by simp)
  simp only [inputReaders,hq,hk,hv,hd,hz]

theorem forward_values_inputs {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (shape : Shape) (σ : ℝ) (s t : State α) (h : SameInputs s t) (pids : s.pids=t.pids) :
    forwardValues M D (forwardConfig shape σ) s=forwardValues M D (forwardConfig shape σ) t := by
  have hq:=h "q" (by simp)
  have hk:=h "k" (by simp)
  have hv:=h "v" (by simp)
  have update : forwardUpdate M D (forwardConfig shape σ) s =
      forwardUpdate M D (forwardConfig shape σ) t := by
    funext diagonal start previous
    unfold forwardUpdate forwardQuery matrixRead queryChunk keyChunk valueChunk
    simp only [forwardConfig,pids,hq,hk,hv]
  simp only [forwardValues,update,queryChunk,pids]

theorem query_values_inputs {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (shape : Shape) (σ : ℝ) (tile : Tile) (s t : State α) (h : SameInputs s t) :
    queryValues M D (dqConfig shape σ tile) s=queryValues M D (dqConfig shape σ tile) t := by
  have hq:=h "q" (by simp)
  have hk:=h "k" (by simp)
  have hv:=h "v" (by simp)
  have hd:=h "do" (by simp)
  have hz:=h "dz" (by simp)
  have update : queryUpdate M D (dqConfig shape σ tile) s =
      queryUpdate M D (dqConfig shape σ tile) t := by
    funext diagonal start previous i
    unfold queryUpdate queryInput queryOutGrad queryNormGrad matrixRead rawMatrixRead
    simp only [dqConfig,backwardConfig,forwardConfig,hq,hk,hv,hd,hz]
  simp only [queryValues,update]

theorem keyValue_values_inputs {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (shape : Shape) (σ : ℝ) (tile : Tile) (s t : State α) (h : SameInputs s t) :
    backwardValues M D (backwardConfig shape σ tile) s=backwardValues M D (backwardConfig shape σ tile) t := by
  have hq:=h "q" (by simp)
  have hk:=h "k" (by simp)
  have hv:=h "v" (by simp)
  have hd:=h "do" (by simp)
  have hz:=h "dz" (by simp)
  have update : backwardUpdate M D (backwardConfig shape σ tile) s =
      backwardUpdate M D (backwardConfig shape σ tile) t := by
    funext diagonal start previous
    unfold backwardUpdate backwardKey backwardValue matrixRead rawMatrixRead
    simp only [backwardConfig,forwardConfig,hq,hk,hv,hd,hz]
  simp only [backwardValues,update]

theorem entry_sameInputs (s t : State NativeStage.Word) (h : SameInputs s t) :
    SameInputs (NativeStage.entry s) (NativeStage.entry t) := by
  intro R hR
  funext off
  change NativeStage.mapCell _ (s.mem R off)=NativeStage.mapCell _ (t.mem R off)
  rw [h R hR]

#axiomsClean sameReaders
#axiomsClean forward_values_inputs
#axiomsClean query_values_inputs
#axiomsClean keyValue_values_inputs
#axiomsClean entry_sameInputs
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel
