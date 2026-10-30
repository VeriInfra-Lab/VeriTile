import bench.repairs.tritonbench_g.cross_entropy1.Proofs.NativeReturns

namespace VeriTile.Bench.Repairs.CrossEntropy1.Floating
open VeriTile.Triton Source Host
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 6000000

/-- One peer's actual selected FP source and host-local reduction trace. The
peer tensor values below are computed from these calls, not supplied outputs. -/
structure Peer {σ : Type} (obs : Observations Expr σ) where
  runtime : Runtime Expr σ
  library : Library obs runtime
  operations : TensorLibrary obs hostArithmetic runtime
  options : Options
  logits : Tensor
  labels : Tensor
  initial : World Expr σ
  final : World Expr σ
  returned : ForwardReturn
  forward : ForwardData algebra runtime options logits labels initial returned final
  kernel : ForwardKernelData obs forward
  aggregate : AggregateData runtime options forward.world forward.rows forward.count forward.labels forward.losses forward.lse
    forward.computed forward.pair forward.aggregated

def peerLoss {σ : Type} {obs : Observations Expr σ} (peers : Nat → Peer obs) (rows i : Nat) : Expr :=
  value obs (peers (i/rows)).aggregate.localState (peers (i/rows)).aggregate.localPair.1 (i%rows)

def peerLSE {σ : Type} {obs : Observations Expr σ} (peers : Nat → Peer obs) (rows i : Nat) : Expr :=
  value obs (peers (i/rows)).aggregate.localState (peers (i/rows)).aggregate.localPair.2 (i%rows)

theorem local_graph {σ : Type} (obs : Observations Expr σ) (peer : Peer obs) (row : Nat) (inside : row<peer.forward.rows) :
    value obs peer.aggregate.localState peer.aggregate.localPair.1 row=
      localResult hostArithmetic.sum peer.forward.count peer.forward.rows peer.kernel.lossValues row ∧
    value obs peer.aggregate.localState peer.aggregate.localPair.2 row=
      localResult hostArithmetic.logsumexp peer.forward.count peer.forward.rows peer.kernel.lseValues row := by
  have arrays := forward_arrays obs peer.runtime peer.library peer.kernel
  have locals := local_reduction_values obs hostArithmetic peer.runtime peer.operations peer.forward.count peer.forward.rows
    peer.forward.losses peer.forward.lse peer.forward.computed peer.aggregate.localPair peer.aggregate.localState
    peer.kernel.lossValues peer.kernel.lseValues arrays.1 arrays.2 peer.kernel.separation.outputDistinct.symm peer.aggregate.localCalls
  exact ⟨locals.losses.values row (by simpa [Tensor.numel,locals.losses.shapeEq] using inside),
    locals.lse.values row (by simpa [Tensor.numel,locals.lse.shapeEq] using inside)⟩

theorem transmitted_graphs {σ : Type} (obs : Observations Expr σ) (peers : Nat → Peer obs)
    (rows rank row : Nat) (shape : (peers rank).forward.rows=rows) (inside : row<rows) :
    peerLoss peers rows (rank*rows+row)=
      localResult hostArithmetic.sum (peers rank).forward.count rows (peers rank).kernel.lossValues row ∧
    peerLSE peers rows (rank*rows+row)=
      localResult hostArithmetic.logsumexp (peers rank).forward.count rows (peers rank).kernel.lseValues row := by
  have coordinates := RealHost.block_coordinates rows row rank inside
  unfold peerLoss peerLSE
  rw [coordinates.1,coordinates.2]
  simpa only [shape] using local_graph obs (peers rank) row (by simpa only [shape] using inside)

/-- At each collective index the two messages are actual peer-local output
graphs. This witness binds only communication input payloads to live tensors;
the source/host calculation of those payloads is the theorem above. -/
def Communication {σ : Type} {obs : Observations Expr σ} (peers : Nat → Peer obs) (rows : Nat)
    {rt : Runtime Expr σ} {gs gr : Nat → Nat} (comms : CollectiveLibrary obs hostArithmetic rt gs gr)
    {o : Options} {world : Nat} {localPair rankPair : Tensor × Tensor} {a b : World Expr σ}
    (calls : RankTrace rt o world rows localPair a rankPair b) : Prop :=
  RankInputs comms calls (peerLoss peers rows) (peerLSE peers rows)

#axiomsClean local_graph
#axiomsClean transmitted_graphs
#specNonCircular Communication avoiding [Math.loss,Math.gradient]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Floating
