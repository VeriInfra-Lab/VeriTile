import bench.tritonbench_g.chunk_retention_ops.IO
import bench.tritonbench_g.chunk_retention_ops.Proofs.KernelBackwardSafety
import bench.tritonbench_g.chunk_retention_ops.Proofs.KernelGradientSafety
import bench.tritonbench_g.chunk_retention.Proofs.ForwardHSafety
import bench.tritonbench_g.chunk_retention.Proofs.ScaleFirstSafety

namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.PairIO
open VeriTile Triton FP.Structural FP.ProfiledRewrite KernelPacket PublicIO
open KernelReal (parameters)
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
open VeriTile.Bench.Repairs.ChunkLinearAttn
noncomputable section
variable {α β : Type} [Inhabited β]
set_option maxHeartbeats 1200000
set_option maxRecDepth 30000

/-- Declared finite windows bound every actual source access in each of the
four stages, through all actual loop predecessor states. No PID admissibility
or numerical equality is assumed by this safety theorem. -/
def outputCandidate (c : Shape) (scale : ℝ) : ComputeKernel :=
  Source.forwardOScaleFirst "q" "k" "v" "h" "o"
    (c.T*c.K) c.K 1 (c.T*c.V) c.V 1 (c.chunks*c.K*c.V) c.V
    scale c.H c.T c.K c.V 64 c.BK c.BV

theorem candidate_source_identity (c : Shape) (scale : ℝ) : outputCandidate c scale=
    VeriTile.Bench.Repairs.ChunkRetention.Whole001.ScaleFirstSource.kernel (KernelOutput.shape c) scale := rfl

def candidateKernel (a : Call) : ComputeKernel := match a.stage with
  | .forwardState=>KernelForward.kernel a.shape a.inputState a.outputState
  | .forwardOutput=>outputCandidate a.shape a.scale
  | .backwardState=>KernelBackward.kernel (parameters a.shape) a.scale a.inputState a.outputState
  | .backwardGradient=>KernelGradient.kernel (parameters a.shape) a.scale

theorem candidate_safe (M : Algebra β) (D : DotPrimitive β) (encode : α→β) (observe : β→α)
    (a : Call) (xs : PublicIO.Inputs α a.shape) (bounds : RegionName→Nat)
    (bounded : (io (engine (withDot M D)) encode observe a).InBounds bounds xs)
    (s : State β) :
    TraceSafe (engine (withDot M D)) bounds (candidateKernel a) ((launch (engine (withDot M D)) a).initial s) := by
  dsimp only [ProgramIO.InBounds,io,layout] at bounded
  rcases a with ⟨c,stage,scale,seed,final,pid⟩
  cases stage with
  | forwardState =>
    apply VeriTile.Bench.Repairs.ChunkRetention.Whole001.ForwardHSource.source_safe
      M (KernelOutput.sharedDot D) (KernelOutput.shape c) seed final
    · intro hs i hi
      exact bounded.2.1 4 (⟨_,hi.1⟩,⟨_,hi.2⟩,()) ⟨rfl,hs⟩
    · intro e t he ht
      simpa [inputOffset,inputs,launch,Invocation.initial,bindRegisters,
        KernelOutput.shape,parameters,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using
        bounded.2.1 1 (⟨t,ht⟩,⟨e,he⟩,()) (Or.inl rfl)
    · intro t j ht hj
      exact bounded.2.1 2 (⟨t,ht⟩,⟨j,hj⟩,()) (Or.inl rfl)
    · intro n hn i hi
      exact bounded.2.2.1 0 (⟨n,hn⟩,i) ⟨rfl,hi⟩
    · intro hf i hi
      exact bounded.2.2.1 1 i ⟨rfl,hf,hi⟩
  | forwardOutput =>
    apply VeriTile.Bench.Repairs.ChunkRetention.Whole001.ScaleFirstSource.source_safe
      M (KernelOutput.sharedDot D) (KernelOutput.shape c) scale
    · intro t e ht he
      exact bounded.2.1 0 (⟨t,ht⟩,⟨e,he⟩,()) (Or.inl rfl)
    · intro e t he ht
      simpa [inputOffset,inputs,launch,Invocation.initial,bindRegisters,
        KernelOutput.shape,parameters,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using
        bounded.2.1 1 (⟨t,ht⟩,⟨e,he⟩,()) (Or.inr (Or.inl rfl))
    · intro e j he hj
      exact bounded.2.1 6 (⟨e,he⟩,⟨j,hj⟩,()) (Or.inl rfl)
    · intro i hi
      exact bounded.2.1 2 (⟨_,hi.1⟩,⟨_,hi.2⟩,()) (Or.inr (Or.inl rfl))
    · intro i hi
      exact bounded.2.2.1 2 i ⟨rfl,hi⟩
  | backwardState =>
    apply KernelBackward.source_safe M D (parameters c) scale seed final _ (by simp [parameters])
    · intro hs i hi
      exact bounded.2.1 5 (⟨_,hi.1⟩,⟨_,hi.2⟩,()) ⟨rfl,hs⟩
    · intro e t he ht
      simpa [inputOffset,inputs,launch,Invocation.initial,bindRegisters,
        KernelOutput.shape,parameters,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using
        bounded.2.1 0 (⟨t,ht⟩,⟨e,he⟩,()) (Or.inr (Or.inl rfl))
    · intro t j ht hj
      exact bounded.2.1 3 (⟨t,ht⟩,⟨j,hj⟩,()) (Or.inl rfl)
    · intro n hn i hi
      exact bounded.2.2.1 3 (⟨n,hn⟩,i) ⟨rfl,hi⟩
    · intro hf i hi
      exact bounded.2.2.1 4 i ⟨rfl,hf,hi⟩
  | backwardGradient =>
    apply KernelGradient.source_safe M D (parameters c) scale _ (by simp [parameters])
    · intro e t he ht
      simpa [inputOffset,inputs,launch,Invocation.initial,bindRegisters,
        KernelOutput.shape,parameters,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using
        bounded.2.1 0 (⟨t,ht⟩,⟨e,he⟩,()) (Or.inr (Or.inr rfl))
    · intro t e ht he
      exact bounded.2.1 1 (⟨t,ht⟩,⟨e,he⟩,()) (Or.inr (Or.inr rfl))
    · intro t j ht hj
      exact bounded.2.1 2 (⟨t,ht⟩,⟨j,hj⟩,()) (Or.inr (Or.inr rfl))
    · intro t j ht hj
      exact bounded.2.1 3 (⟨t,ht⟩,⟨j,hj⟩,()) (Or.inr rfl)
    · intro j e hj he
      simpa [inputOffset,inputs,launch,Invocation.initial,bindRegisters,
        KernelOutput.shape,parameters,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using
        bounded.2.1 6 (⟨e,he⟩,⟨j,hj⟩,()) (Or.inr rfl)
    · intro e j he hj
      exact bounded.2.1 7 (⟨e,he⟩,⟨j,hj⟩,()) rfl
    · intro vb hv i hi
      exact bounded.2.2.1 7 (⟨vb,hv⟩,i) ⟨rfl,hi⟩
    · intro i hi
      exact bounded.2.2.1 5 i ⟨rfl,hi⟩
    · intro i hi
      exact bounded.2.2.1 6 i ⟨rfl,hi⟩

#axiomsClean candidate_source_identity
#axiomsClean candidate_safe

end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.PairIO
