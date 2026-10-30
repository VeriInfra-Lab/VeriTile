import bench.repairs.tritonbench_g.chunk_retention.Proofs.ForwardHSafety
import bench.repairs.tritonbench_g.chunk_retention.Proofs.ScaleFirstSafety
import bench.repairs.tritonbench_g.chunk_retention.Proofs.BackwardHSafety
import bench.repairs.tritonbench_g.chunk_retention.Proofs.BackwardQKVSafety
import bench.repairs.tritonbench_g.chunk_retention.KernelOnly.IO

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.PairIO
open VeriTile Triton FP.Structural FP.ProfiledRewrite KernelModel KernelPacket PublicIO
noncomputable section
variable {α β : Type} [Inhabited β]
set_option maxHeartbeats 1500000

/-- The shared IO windows imply every actual stage load/store bound. Nothing
about source access safety is assumed in the public logical input domain. -/
def candidateKernel (a : Call) : ComputeKernel := match a.stage with
  | .forwardO=>ScaleFirstSource.kernel a.shape a.scale
  | .forwardH=>ForwardHSource.kernel a.shape a.seed a.final
  | .backwardH=>BackwardHSource.kernel a.shape a.scale a.seed
  | .backwardQKV=>BackwardQKVSource.kernel a.shape a.scale

theorem candidate_safe (M : Algebra β) (D : DotPrimitive β) (encode : α→β) (observe : β→α)
    (a : Call) (s : State β) (legal : PublicIO.Legal a s.pids s.numPids)
    (xs : PublicIO.Inputs α a.shape) (bounds : RegionName→Nat)
    (bounded : (io (engine (withDot M D)) encode observe a s.pids s.numPids).InBounds bounds xs) :
    TraceSafe (engine (withDot M D)) bounds (candidateKernel a) s := by
  have rq (enabled : inputEnabled a 0) (t e : Nat) (ht : t<a.shape.T) (he : e<a.shape.K) :
      s.pids 2*(a.shape.T*a.shape.K)+t*a.shape.K+e<bounds "q" :=
    bounded.2.1 ⟨0,by simp [io,layout,inputs]⟩ (⟨s.pids 2,legal.1⟩,⟨t,ht⟩,⟨e,he⟩,()) enabled
  have rk (enabled : inputEnabled a 1) (t e : Nat) (ht : t<a.shape.T) (he : e<a.shape.K) :
      s.pids 2*(a.shape.T*a.shape.K)+t*a.shape.K+e<bounds "k" :=
    bounded.2.1 ⟨1,by simp [io,layout,inputs]⟩ (⟨s.pids 2,legal.1⟩,⟨t,ht⟩,⟨e,he⟩,()) enabled
  have rv (enabled : inputEnabled a 2) (t j : Nat) (ht : t<a.shape.T) (hj : j<a.shape.V) :
      s.pids 2*(a.shape.T*a.shape.V)+t*a.shape.V+j<bounds "v" :=
    bounded.2.1 ⟨2,by simp [io,layout,inputs]⟩ (⟨s.pids 2,legal.1⟩,⟨t,ht⟩,⟨j,hj⟩,()) enabled
  have rg (enabled : inputEnabled a 3) (t j : Nat) (ht : t<a.shape.T) (hj : j<a.shape.V) :
      s.pids 2*(a.shape.T*a.shape.V)+t*a.shape.V+j<bounds "incoming" :=
    bounded.2.1 ⟨3,by simp [io,layout,inputs]⟩ (⟨s.pids 2,legal.1⟩,⟨t,ht⟩,⟨j,hj⟩,()) enabled
  have r0 (enabled : inputEnabled a 4) (e j : Nat) (he : e<a.shape.K) (hj : j<a.shape.V) :
      s.pids 2*a.shape.K*a.shape.V+e*a.shape.V+j<bounds "h0" :=
    bounded.2.1 ⟨4,by simp [io,layout,inputs]⟩ (⟨s.pids 2,legal.1⟩,⟨e,he⟩,⟨j,hj⟩,()) enabled
  have rt (enabled : inputEnabled a 5) (e j : Nat) (he : e<a.shape.K) (hj : j<a.shape.V) :
      s.pids 2*a.shape.K*a.shape.V+e*a.shape.V+j<bounds "terminal" :=
    bounded.2.1 ⟨5,by simp [io,layout,inputs]⟩ (⟨s.pids 2,legal.1⟩,⟨e,he⟩,⟨j,hj⟩,()) enabled
  have rh (enabled : inputEnabled a 6) (n e j : Nat) (hn : n<a.shape.chunks)
      (he : e<a.shape.K) (hj : j<a.shape.V) :
      s.pids 2*(a.shape.chunks*a.shape.K*a.shape.V)+n*a.shape.K*a.shape.V+e*a.shape.V+j<bounds "h" :=
    bounded.2.1 ⟨6,by simp [io,layout,inputs]⟩
      (⟨s.pids 2,legal.1⟩,⟨n,hn⟩,⟨e,he⟩,⟨j,hj⟩,()) enabled
  have rdh (enabled : inputEnabled a 7) (n e j : Nat) (hn : n<a.shape.chunks)
      (he : e<a.shape.K) (hj : j<a.shape.V) :
      s.pids 2*(a.shape.chunks*a.shape.K*a.shape.V)+n*a.shape.K*a.shape.V+e*a.shape.V+j<bounds "dh" :=
    bounded.2.1 ⟨7,by simp [io,layout,inputs]⟩
      (⟨s.pids 2,legal.1⟩,⟨n,hn⟩,⟨e,he⟩,⟨j,hj⟩,()) enabled
  cases stage : a.stage with
  | forwardH =>
    simp only [candidateKernel,stage]
    apply ForwardHSource.source_safe
    · intro hs i hi; exact r0 ⟨stage,hs⟩ _ _ hi.1 hi.2
    · intro e t he ht
      simpa only [Nat.add_assoc,Nat.add_left_comm,Nat.add_comm] using rk (Or.inl stage) t e ht he
    · exact rv (Or.inl stage)
    · intro n hn i hi
      exact bounded.2.2.1 ⟨0,by simp [io,layout,outputs]⟩ (⟨n,hn⟩,i) ⟨stage,hi⟩
    · intro hf i hi
      exact bounded.2.2.1 ⟨1,by simp [io,layout,outputs]⟩ i ⟨stage,hf,hi⟩
  | forwardO =>
    have hc : s.pids 1<a.shape.chunks := by
      have hl := legal; simp only [PublicIO.Legal,stage] at hl; exact hl.2.2.2
    simp only [candidateKernel,stage]
    apply ScaleFirstSource.source_safe
    · exact rq (Or.inl stage)
    · intro e t he ht
      simpa only [Nat.add_assoc,Nat.add_left_comm,Nat.add_comm] using rk (Or.inr (Or.inl stage)) t e ht he
    · intro e j he hj; exact rh (Or.inl stage) _ e j hc he hj
    · intro i hi; exact rv (Or.inr (Or.inl stage)) _ _ hi.1 hi.2
    · intro i hi
      exact bounded.2.2.1 ⟨2,by simp [io,layout,outputs]⟩ i ⟨stage,hi⟩
  | backwardH =>
    simp only [candidateKernel,stage]
    apply BackwardHSource.source_safe
    · intro hs i hi; exact rt ⟨stage,hs⟩ _ _ hi.1 hi.2
    · intro e t he ht
      simpa only [Nat.add_assoc,Nat.add_left_comm,Nat.add_comm] using rq (Or.inr (Or.inl stage)) t e ht he
    · exact rg (Or.inl stage)
    · intro n hn i hi
      exact bounded.2.2.1 ⟨3,by simp [io,layout,outputs]⟩ (⟨n,hn⟩,i) ⟨stage,hi⟩
  | backwardQKV =>
    have hc : s.pids 1<a.shape.chunks := by
      have hl := legal; simp only [PublicIO.Legal,stage] at hl; exact hl.2.2.2
    simp only [candidateKernel,stage]
    apply BackwardQKVSource.source_safe
    · intro e t he ht
      simpa only [Nat.add_assoc,Nat.add_left_comm,Nat.add_comm] using rq (Or.inr (Or.inr stage)) t e ht he
    · exact rk (Or.inr (Or.inr stage))
    · exact rv (Or.inr (Or.inr stage))
    · exact rg (Or.inr stage)
    · intro j e hj he
      simpa only [Nat.add_assoc,Nat.add_left_comm,Nat.add_comm] using rh (Or.inr stage) _ e j hc he hj
    · intro e j he hj; exact rdh stage _ e j hc he hj
    · intro f hf i hi
      exact bounded.2.2.1 ⟨6,by simp [io,layout,outputs]⟩ (⟨f,hf⟩,i) ⟨stage,hi⟩
    · intro i hi
      exact bounded.2.2.1 ⟨4,by simp [io,layout,outputs]⟩ i ⟨stage,hi⟩
    · intro i hi
      exact bounded.2.2.1 ⟨5,by simp [io,layout,outputs]⟩ i ⟨stage,hi⟩

#axiomsClean candidate_safe
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.PairIO
