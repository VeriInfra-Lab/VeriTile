import bench.repairs.tritonbench_g.cross_entropy1.Proofs.LocalReduction
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.Frames

namespace VeriTile.Bench.Repairs.CrossEntropy1.Host
open VeriTile.Triton Source
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 4000000

/-- Per-collective input buffers, indexed in rank-major order, and a pending
work handle. The callbacks are transmitted peer inputs, never expected task
outputs. A whole group certificate must bind them to the peers' actual local
reduction traces. The sum result is unavailable until the matching wait. -/
structure CollectiveLibrary {α σ : Type} [Inhabited α] (obs : Observations α σ)
    (A : TensorArithmetic α) (rt : Runtime α σ) (groupSize groupRank : Nat → Nat) where
  gatherInputs : Nat → Tensor → World α σ → Nat → α
  reduceInputs : Nat → Tensor → World α σ → Nat → α
  pending : Nat → Tensor → Nat → Nat → (Nat → α) → World α σ → Prop
  gatherOwn : ∀ group src before rows row, src.shape=[rows] → row<rows →
    gatherInputs group src before (groupRank group*rows+row)=value obs before src row
  reduceOwn : ∀ group src before rows row, src.shape=[rows] → row<rows →
    reduceInputs group src before (groupRank group*rows+row)=value obs before src row
  gather : ∀ dst src group rows before after, before.live dst.region → before.live src.region →
    dst.shape=[groupSize group,rows] → src.shape=[rows] →
    dst.dtype="torch.float32" → src.dtype="torch.float32" →
    rt.allGather dst src group before=.ok () after →
      Mutation obs before after dst ∧
      FloatArray obs after dst [groupSize group,rows] (gatherInputs group src before)
  start : ∀ src group rows before handle after, before.live src.region → src.shape=[rows] →
    src.dtype="torch.float32" → rt.allReduceAsync src group before=.ok handle after →
      Mutation obs before after src ∧ pending handle src (groupSize group) rows (reduceInputs group src before) after
  independentLogsumexp : ∀ handle dst world rows sends before input output after,
    pending handle dst world rows sends before → input.region≠dst.region →
    rt.logsumexp0 input before=.ok output after → pending handle dst world rows sends after
  wait : ∀ handle dst world rows sends before after, pending handle dst world rows sends before →
    rt.wait handle before=.ok () after → Mutation obs before after dst ∧
      FloatArray obs after dst [rows]
        (fun row => A.rankSum ((List.range world).map (fun rank => sends (rank*rows+row))))

def rankResult {α : Type} (reduce : List α → α) (world rows : Nat)
    (own peers : Nat → α) (row : Nat) : α :=
  if world>1 then reduce ((List.range world).map (fun rank => peers (rank*rows+row))) else own row

/-- Bind one collective's transmitted inputs to supplied peer buffers on
exactly their visible coordinates. The final group theorem supplies those
buffers from each peer's source-linked LocalTrace, not a mathematical answer. -/
def RankInputs {α σ : Type} [Inhabited α] {obs : Observations α σ} {A : TensorArithmetic α}
    {rt : Runtime α σ} {gs gr : Nat → Nat} (comms : CollectiveLibrary obs A rt gs gr)
    {o : Options} {world rows : Nat} {localPair : Tensor × Tensor} {initial : World α σ}
    {pair : Tensor × Tensor} {final : World α σ}
    (trace : RankTrace rt o world rows localPair initial pair final) (lossPeers lsePeers : Nat → α) : Prop :=
  match trace with
  | .direct _ => True
  | .reduced _ _ _ _ allocated collected _ _ _ _ _ _ _ _ =>
    (∀ i, i<world*rows → comms.gatherInputs (o.processGroup.getD 0) localPair.2 allocated i=lsePeers i) ∧
    (∀ i, i<world*rows → comms.reduceInputs (o.processGroup.getD 0) localPair.1 collected i=lossPeers i)

theorem peer_values_congr {α : Type} (world rows row : Nat) (hr : row<rows) (a b : Nat → α)
    (same : ∀ i, i<world*rows → a i=b i) :
    (List.range world).map (fun rank => a (rank*rows+row))=
      (List.range world).map (fun rank => b (rank*rows+row)) := by
  apply List.map_congr_left
  intro rank rankIn
  apply same
  have rb := List.mem_range.mp rankIn
  have product := Nat.mul_le_mul_right rows (Nat.succ_le_of_lt rb)
  simp only [Nat.succ_mul] at product
  omega

theorem rank_reduction_values {α σ : Type} [Inhabited α] (obs : Observations α σ)
    (A : TensorArithmetic α) (rt : Runtime α σ) (lib : Library obs rt) (ops : TensorLibrary obs A rt)
    (gs gr : Nat → Nat) (comms : CollectiveLibrary obs A rt gs gr)
    (o : Options) (world rows : Nat) (localPair pair : Tensor × Tensor) (initial final : World α σ)
    (lv sv lossPeers lsePeers : Nat → α)
    (loss : FloatArray obs initial localPair.1 [rows] lv) (saved : FloatArray obs initial localPair.2 [rows] sv)
    (separate : localPair.1.region≠localPair.2.region)
    (size : world>1 → world=gs (o.processGroup.getD 0))
    (trace : RankTrace rt o world rows localPair initial pair final)
    (inputs : RankInputs comms trace lossPeers lsePeers) :
    pair.1=localPair.1 ∧ Mutation obs initial final localPair.1 ∧
      FloatArray obs final pair.1 [rows] (rankResult A.rankSum world rows lv lossPeers) ∧
      FloatArray obs final pair.2 [rows] (rankResult A.logsumexp world rows sv lsePeers) ∧
      pair.1.region≠pair.2.region := by
  cases trace with
  | direct one =>
    have leq : rankResult A.rankSum world rows lv lossPeers=lv := by funext row; simp [rankResult,one]
    have seq : rankResult A.logsumexp world rows sv lsePeers=sv := by funext row; simp [rankResult,one]
    exact ⟨rfl,Mutation.refl obs initial localPair.1,by simpa [leq] using loss,by simpa [seq] using saved,separate⟩
  | reduced many gathered globalLSE handle allocated collected pending reduced waited allocationCall gatherCall sumCall lseCall waitCall =>
    have allocatedData := lib.empty _ _ _ _ _ _ allocationCall
    have allocation := allocatedData.1
    have gather := comms.gather gathered localPair.2 (o.processGroup.getD 0) rows allocated collected
      allocation.live (allocation.frame.live _ saved.live) (by simpa [← size many] using allocation.shapeEq)
      saved.shapeEq (allocation.dtypeEq.trans saved.dtype) saved.dtype gatherCall
    have collectedFrame := allocation.frame.freshMutation gather.1 allocation.fresh
    have gatheredSeparate : gathered.region≠localPair.1.region := fresh_ne_live obs _ _ _ _ _ _ allocation _ loss.live
    have start := comms.start localPair.1 (o.processGroup.getD 0) rows collected handle pending
      (collectedFrame.live _ loss.live) loss.shapeEq loss.dtype sumCall
    have gatheredPending := gather.2.mutate start.1 gatheredSeparate
    have lseData := ops.logsumexp gathered world rows pending globalLSE reduced gatheredPending.live
      (by simpa [← size many] using gatheredPending.shapeEq) gatheredPending.dtype lseCall
    have pendingWork := comms.independentLogsumexp handle localPair.1 (gs (o.processGroup.getD 0)) rows
      (comms.reduceInputs (o.processGroup.getD 0) localPair.1 collected) pending gathered globalLSE reduced
      start.2 gatheredSeparate lseCall
    have completed := comms.wait handle localPair.1 _ rows _ reduced _ pendingWork waitCall
    have globalSeparate : globalLSE.region≠localPair.1.region := fresh_ne_live obs _ _ _ _ _ _ lseData.allocation _
      (start.1.live _ (collectedFrame.live _ loss.live))
    have lseFinal := lseData.array.mutate completed.1 globalSeparate
    have wholeFrame := ((start.1.prepend collectedFrame).append lseData.allocation.frame).trans completed.1
    refine ⟨rfl,wholeFrame,?_,?_,globalSeparate.symm⟩
    · refine ⟨completed.2.shapeEq,completed.2.dtype,completed.2.live,?_⟩
      intro row hr
      have bound : row<rows := by simpa [Tensor.numel,completed.2.shapeEq] using hr
      rw [completed.2.values row hr,← size many]
      rw [peer_values_congr world rows row bound _ _ inputs.2]
      simp only [rankResult,if_pos many]
    · refine ⟨lseFinal.shapeEq,lseFinal.dtype,lseFinal.live,?_⟩
      intro row hr
      have bound : row<rows := by simpa [Tensor.numel,lseFinal.shapeEq] using hr
      rw [lseFinal.values row hr]
      have gatheredArray : FloatArray obs pending gathered [world,rows]
          (comms.gatherInputs (o.processGroup.getD 0) localPair.2 allocated) := by
        simpa [← size many] using gatheredPending
      rw [columnValues_eq gatheredArray row bound,peer_values_congr world rows row bound _ _ inputs.1]
      simp only [rankResult,if_pos many]

#axiomsClean peer_values_congr
#axiomsClean rank_reduction_values
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Host
