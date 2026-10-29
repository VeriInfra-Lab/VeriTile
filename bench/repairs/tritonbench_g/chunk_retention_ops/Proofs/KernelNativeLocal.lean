import bench.repairs.tritonbench_g.chunk_retention.Proofs.KernelStaging
import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelOutputMemory
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelNative
open VeriTile Triton FP.Structural
namespace S
export VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelStaging
  (Term algebra lower dot gradientDot LiteralLaws)
end S
noncomputable section

def sharedOps (ops : Ops α) : VeriTile.Bench.Repairs.ChunkRetention.Whole001.Ops α :=
  ⟨ops.zero,ops.integer,ops.cast,ops.add,ops.sub,ops.mul,ops.fma,ops.exp2,ops.log2⟩

theorem shared_chain (ops : Ops α) (n : Nat) (left right : Nat→α) (seed : α) :
    VeriTile.Bench.Repairs.ChunkRetention.Whole001.chain (sharedOps ops) n left right seed=
      chain ops n left right seed := by
  induction n with
  | zero=>rfl
  | succ n ih=>
    change ops.fma (left n) (right n) (VeriTile.Bench.Repairs.ChunkRetention.Whole001.chain (sharedOps ops) n left right seed)=_
    rw [ih]
    rfl

/-- Local literal/conversion and exp2/log2 profile. No matrix or output law is
needed to derive the head-rate graph. Shared with the sibling retention proof. -/
theorem lower_shared_rate (ops : VeriTile.Bench.Repairs.ChunkRetention.Whole001.Ops α)
    (literal : ℝ→α) (laws : S.LiteralLaws ops literal)
    (c : VeriTile.Bench.Repairs.ChunkRetention.Whole001.Shape) (head : Nat) :
    S.lower ops literal
      (VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelModel.rate (S.algebra ops.zero) c head)=
      ops.log2 (ops.sub (ops.integer 1) (ops.exp2 (ops.sub (ops.integer (-5)) (ops.integer (head%c.H))))) := by
  simp [VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelModel.rate,S.algebra,S.lower,laws.one,laws.negativeFive]

theorem lower_rate (ops : Ops α) (literal : ℝ→α) (laws : S.LiteralLaws (sharedOps ops) literal)
    (c : Shape) (pid : KernelBackward.PID) :
    S.lower (sharedOps ops) literal (KernelBackward.logRate (S.algebra ops.zero) (KernelReal.parameters c) pid)=
      logRate ops c pid.z := by
  simpa only [logRate,sharedOps,KernelOutput.shape] using
    lower_shared_rate (sharedOps ops) literal laws (KernelOutput.shape c) pid.z

theorem lower_live_dot (ops : Ops α) (literal : ℝ→α) (k : Nat)
    (left right : Fin k→S.Term α) (seed : S.Term α) :
    S.lower (sharedOps ops) literal (.binary .add seed (.pendingDot k left right))=
      chain ops k
        (fun i=>if hi : i<k then S.lower (sharedOps ops) literal (left ⟨i,hi⟩) else ops.zero)
        (fun i=>if hi : i<k then S.lower (sharedOps ops) literal (right ⟨i,hi⟩) else ops.zero)
        (S.lower (sharedOps ops) literal seed) := by
  rw [VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelStaging.consume_dot,shared_chain]
  rfl

theorem lower_materialized (ops : Ops α) (literal : ℝ→α) (a b : S.Term α) :
    S.lower (sharedOps ops) literal (.binary .add a (.materialize b))=
      ops.add (S.lower (sharedOps ops) literal a) (S.lower (sharedOps ops) literal b) :=
  VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelStaging.materialized_add _ _ _ _

#axiomsClean lower_shared_rate
#axiomsClean lower_rate
#axiomsClean lower_live_dot
#axiomsClean lower_materialized
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelNative
