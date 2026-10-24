import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.PublicForwardIO
import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.PublicBackwardIO
import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.KernelPhysical

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.PublicIO
open VeriTile.Triton FP.Structural Kernel Kernel.Native
noncomputable section
variable {α σ : Type}

/-- Read channels from the actual physical memory. Bases may alias; no
freshness or input/output separation is introduced by this transport. -/
def physicalForward (s : Shape) (b : Bases) (m : Memory α) :
    ChannelValues α (forwardInputs s) := by
  refine Fin.cases ?_ (Fin.cases ?_ (Fin.cases ?_ (fun k => k.elim0)))
  · intro j; exact m (b.q+dense s.T s.DK j.1.val j.2.1.val j.2.2.1.val)
  · intro j; exact m (b.k+dense s.T s.DK j.1.val j.2.1.val j.2.2.1.val)
  · intro j; exact m (b.v+dense s.T s.DV j.1.val j.2.1.val j.2.2.1.val)

def physicalBackward (s : Shape) (b : Bases) (m : Memory α) :
    ChannelValues α (backwardInputs s) := by
  refine Fin.cases ?_ (Fin.cases ?_ (Fin.cases ?_ (Fin.cases ?_ (fun k => k.elim0))))
  · intro j; exact m (b.q+dense s.T s.DK j.1.val j.2.1.val j.2.2.1.val)
  · intro j; exact m (b.k+dense s.T s.DK j.1.val j.2.1.val j.2.2.1.val)
  · intro j; exact m (b.v+dense s.T s.DV j.1.val j.2.1.val j.2.2.1.val)
  · intro j; exact m (b.g+dense s.T s.DV j.1.val j.2.1.val j.2.2.1.val)

theorem physical_forward_binding (a : Scalar α) (p : DotTrees) (s : Shape)
    (b : Bases) (m : Memory α) (tile : Tile) :
    (forwardIO a p s tile).InputBinding (physicalForward s b m) (initial s b m tile) := by
  intro port j active
  fin_cases port <;> rfl

theorem physical_backward_binding (a : Scalar α) (p : DotTrees) (s : Shape)
    (b : Bases) (m : Memory α) (tile : Tile) :
    (backwardIO a p s tile).InputBinding (physicalBackward s b m) (initial s b m tile) := by
  intro port j active
  fin_cases port <;> rfl

theorem forward_initial (s : Shape) (b : Bases) (m : Memory α) (tile : Tile) :
    (forwardInvocation α s tile).initial (initial s b m tile)=initial s b m tile := rfl

theorem backward_initial (s : Shape) (b : Bases) (m : Memory α) (tile : Tile) :
    (backwardInvocation α s tile).initial (initial s b m tile)=initial s b m tile := rfl

/-- The old physical execution is exactly the public source schedule followed
by its original masked output commit, not a second program or proof model. -/
theorem forward_transport (abi : LocalRuntime α σ) (a : Scalar α) (p : DotTrees)
    (s : Shape) (b : Bases) (tile : Tile) (output : Nat) (state : σ) :
    executeForward abi a p s b tile output state =
      (runSchedule ((forwardIO a p s tile).schedule (physicalForward s b (abi.observe state)))
        (initial s b (abi.observe state) tile)).map
          (fun final => forwardCommit abi a p s tile output final state) := by
  change forwardRun abi a p s b tile output state =
    (runSchedule [forwardInvocation α s tile] (initial s b (abi.observe state) tile)).map _
  rw [runSchedule_singleton]
  unfold forwardRun Invocation.run
  rw [forward_initial]
  exact bind_pure_comp (fun final => forwardCommit abi a p s tile output final state)
    (FP.Structural.exec (VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages.withDot algebra dots)
      (forwardKernel s) (initial s b (abi.observe state) tile))

theorem backward_transport (abi : LocalRuntime α σ) (a : Scalar α) (p : DotTrees)
    (s : Shape) (b : Bases) (tile : Tile) (dq dk dv : Nat) (state : σ) :
    executeBackward abi a p s b tile dq dk dv state =
      (runSchedule ((backwardIO a p s tile).schedule (physicalBackward s b (abi.observe state)))
        (initial s b (abi.observe state) tile)).map
          (fun final => valueCommit abi a p s tile dv final
            (keyCommit abi a p s tile "dk" dk final (keyCommit abi a p s tile "dq" dq final state))) := by
  change backwardRun abi a p s b tile dq dk dv state =
    (runSchedule [backwardInvocation α s tile] (initial s b (abi.observe state) tile)).map _
  rw [runSchedule_singleton]
  unfold backwardRun Invocation.run
  rw [backward_initial]
  exact bind_pure_comp (fun final => valueCommit abi a p s tile dv final
    (keyCommit abi a p s tile "dk" dk final (keyCommit abi a p s tile "dq" dq final state)))
    (FP.Structural.exec (VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages.withDot algebra dots)
      (backwardKernel s (programIds s tile)) (initial s b (abi.observe state) tile))

/- `Kernel.forwardCommit_read`, `Kernel.keyCommit_read`, and
`Kernel.valueCommit_read` are the exact output transport equations. The old
physical endpoint lemmas use these equations and the same source execution;
their output separation and full physical frames remain unchanged. -/
#axiomsClean physical_forward_binding
#axiomsClean physical_backward_binding
#axiomsClean forward_transport
#axiomsClean backward_transport
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.PublicIO
