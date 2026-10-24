import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.ReturnedSplit
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.SavedBoundary

namespace VeriTile.Bench.Repairs.CrossEntropyOps.Host
open VeriTile.Triton
open CrossEntropy1.Source (FP32 Cell)
open CrossEntropy1.Host (World Step Observations TensorArithmetic TensorLibrary CollectiveLibrary IntegerViews
  ObjectRegistry FloatArray Preserves RankInputs localResult rankResult value dense_base)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 8000000

inductive ForwardAggregation {α σ : Type} [Inhabited α] {M : FP32 α} {rt : Runtime α σ}
    {o : Options} {logits labels : Tensor} {initial final : World α σ} {returned : ForwardReturn}
    (e : ForwardData M rt o logits labels initial returned final) : Type
  | direct (flag : ¬(e.world>1 ∨ e.cols>65536))
  | split (flag : e.world>1 ∨ e.cols>65536)
      (trace : AggregateData rt o e.world e.rows e.count e.labels e.losses e.lse e.computed e.result e.aggregated)

theorem forward_aggregation {α σ : Type} [Inhabited α] {M : FP32 α} {rt : Runtime α σ}
    {o : Options} {logits labels : Tensor} {initial final : World α σ} {returned : ForwardReturn}
    (e : ForwardData M rt o logits labels initial returned final) : Nonempty (ForwardAggregation e) := by
  have success := e.aggregateCall
  by_cases flag : e.world>1 ∨ e.cols>65536
  · simp only [decide_eq_true flag,↓reduceIte] at success
    obtain ⟨trace⟩ := aggregate_success rt o e.world e.rows e.count e.labels e.losses e.lse e.computed e.result e.aggregated success
    exact ⟨.split flag trace⟩
  · simp only [decide_eq_false flag,Bool.false_eq_true,↓reduceIte,Step.ok.injEq] at success
    exact ⟨.direct flag⟩

def ForwardInputs {α σ : Type} [Inhabited α] {obs : Observations α σ} {A : TensorArithmetic α}
    {rt : Runtime α σ} {gs gr : Nat → Nat} (comms : CollectiveLibrary obs A rt.toRuntime gs gr)
    {M : FP32 α} {o : Options} {logits labels : Tensor} {initial final : World α σ} {returned : ForwardReturn}
    {e : ForwardData M rt o logits labels initial returned final} (a : ForwardAggregation e)
    (lossPeers lsePeers : Nat → α) : Prop :=
  match a with
  | .direct _ => True
  | .split _ trace => RankInputs comms trace.rankCalls lossPeers lsePeers

def finalLoss {α σ : Type} [Inhabited α] {obs : Observations α σ} {M : FP32 α} {rt : Runtime α σ}
    {o : Options} {logits labels : Tensor} {initial final : World α σ} {returned : ForwardReturn}
    {e : ForwardData M rt o logits labels initial returned final} (k : ForwardKernelData obs e)
    (A : TensorArithmetic α) (lossPeers lsePeers : Nat → α) : Nat → α :=
  if e.world>1 ∨ e.cols>65536 then
    aggregateLossValue A o e.world e.count e.rows k.lossValues k.lseValues lossPeers lsePeers (obs.integer initial labels)
  else k.lossValues

def finalLSE {α σ : Type} [Inhabited α] {obs : Observations α σ} {M : FP32 α} {rt : Runtime α σ}
    {o : Options} {logits labels : Tensor} {initial final : World α σ} {returned : ForwardReturn}
    {e : ForwardData M rt o logits labels initial returned final} (k : ForwardKernelData obs e)
    (A : TensorArithmetic α) (lsePeers : Nat → α) : Nat → α :=
  if e.world>1 ∨ e.cols>65536 then
    rankResult A.logsumexp e.world e.rows (localResult A.logsumexp e.count e.rows k.lseValues) lsePeers
  else k.lseValues

def finalZ {α σ : Type} [Inhabited α] {obs : Observations α σ} {M : FP32 α} {rt : Runtime α σ}
    {o : Options} {logits labels : Tensor} {initial final : World α σ} {returned : ForwardReturn}
    {e : ForwardData M rt o logits labels initial returned final} (k : ForwardKernelData obs e)
    (A : TensorArithmetic α) (lsePeers : Nat → α) : Nat → α :=
  if e.world>1 ∨ e.cols>65536 then
    fun row => correctionZ A o (obs.integer initial labels row) (finalLSE k A lsePeers row)
  else fun row => Source.zLoss M e.parameters (CrossEntropy1.Source.label e.parameters.toParameters k.launch.ready.memory row)
    (Source.forwardValues M e.parameters k.launch.ready.memory row 0).lse

theorem unsplit_z_values {α σ : Type} [Inhabited α] (obs : Observations α σ) (rt : Runtime α σ)
    (lib : LocalAPI obs rt) {M : FP32 α} {o : Options} {logits labels : Tensor}
    {initial final : World α σ} {returned : ForwardReturn}
    {e : ForwardData M rt o logits labels initial returned final} (k : ForwardKernelData obs e)
    (unsplit : ¬(e.world>1 ∨ e.cols>65536)) :
    FloatArray obs e.computed e.zlosses [e.rows] (fun row => Source.zLoss M e.parameters
      (CrossEntropy1.Source.label e.parameters.toParameters k.launch.ready.memory row)
      (Source.forwardValues M e.parameters k.launch.ready.memory row 0).lse) := by
  have ids := unsplit_identity e unsplit
  have shape : e.zlosses.shape=[e.rows] := by simpa [ids.2.2] using k.zAllocation.shapeEq
  have exit := lib.tensor.exit _ _ _ k.launch.exitCall
  refine ⟨shape,k.zAllocation.dtypeEq,exit.live _ k.zLive,?_⟩
  intro row hr
  have bound : row<e.rows := by simpa [Tensor.numel,shape] using hr
  have base := dense_base e.zlosses k.zDense (by omega)
  have split : e.parameters.split=Bool.false := by
    simp [ForwardData.parameters,forwardParameters,CrossEntropy1.Host.forwardParameters,unsplit]
  have stored := (forward_kernel_outputs obs rt lib k row 0 bound (by rw [ids.2.2]; omega)).2.2 split
  change e.computed.memory e.zlosses.region (e.zlosses.storageOffset+0*e.rows+row)=_ at stored
  simp only [base,Nat.zero_mul,Nat.zero_add] at stored
  simp only [value,k.zDense.1,Bool.false_eq_true,↓reduceIte,k.zDense.2 row hr,stored,Cell.read_same]

theorem forward_graph_values {α σ : Type} [Inhabited α] (obs : Observations α σ)
    (A : TensorArithmetic α) (rt : Runtime α σ) (lib : LocalAPI obs rt)
    (ops : TensorLibrary obs A rt.toRuntime) (zeros : ZeroAPI obs rt A.zero) (ints : IntegerViews obs)
    (objects : ObjectRegistry rt.toRuntime) (gs gr : Nat → Nat) (comms : CollectiveLibrary obs A rt.toRuntime gs gr)
    {M : FP32 α} {o : Options} {logits labels : Tensor} {initial final : World α σ} {returned : ForwardReturn}
    {e : ForwardData M rt o logits labels initial returned final} (k : ForwardKernelData obs e)
    (views : ForwardViewData obs objects e) (labelDtype : labels.dtype="torch.int64")
    (size : e.world>1 → e.world=gs (o.processGroup.getD 0))
    (aggregation : ForwardAggregation e) (lossPeers lsePeers : Nat → α)
    (inputs : ForwardInputs comms aggregation lossPeers lsePeers) :
    FloatArray obs final returned.losses [e.rows] (finalLoss k A lossPeers lsePeers) ∧
    FloatArray obs final returned.context.lse [e.rows] (finalLSE k A lsePeers) ∧
    FloatArray obs final returned.zLosses [e.rows] (finalZ k A lsePeers) ∧
    SavedForwardData obs rt objects logits labels initial final returned e.rows := by
  cases aggregation with
  | direct flag =>
    have ids := unsplit_identity e flag
    have arrays := forward_arrays obs rt lib k
    have losses : FloatArray obs e.computed e.losses [e.rows] k.lossValues := by simpa [ids.2.2] using arrays.1
    have lses : FloatArray obs e.computed e.lse [e.rows] k.lseValues := by simpa [ids.2.2] using arrays.2
    have z := unsplit_z_values obs rt lib k flag
    have post : Preserves obs e.computed final := by
      rw [←ids.2.1]
      exact (lib.tensor.save _ _ _ _ _ _ e.saveCall).trans (lib.mark _ _ _ e.markCall)
    refine ⟨?_,?_,?_,unsplit_saved obs rt lib ints objects k views labelDtype flag⟩
    · simpa only [finalLoss,if_neg flag,e.returnIdentity,ids.1] using losses.preserve post
    · simpa only [finalLSE,if_neg flag,e.returnIdentity,ids.1] using lses.preserve post
    · simpa only [finalZ,if_neg flag,e.returnIdentity,ids.1] using z.preserve post
  | split flag trace =>
    have values := returned_split_values obs A rt lib ops zeros ints gs gr comms k labelDtype size trace lossPeers lsePeers inputs
    refine ⟨?_,?_,?_,split_saved obs A rt lib ops zeros ints objects gs gr comms k views labelDtype size trace lossPeers lsePeers inputs⟩
    · simpa only [finalLoss,if_pos flag] using values.1
    · simpa only [finalLSE,if_pos flag] using values.2.1
    · simpa only [finalZ,finalLSE,if_pos flag] using values.2.2.1

#axiomsClean forward_aggregation
#axiomsClean unsplit_z_values
#axiomsClean forward_graph_values
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.Host
