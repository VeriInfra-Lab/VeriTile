import bench.tritonbench_g.kldiv_triton.Proofs.Prepare

namespace VeriTile.Bench.Repairs.KldivTriton.Host
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open L2NormBwd.Physical (Matrix)
open Physical (ForwardLayout BackwardLayout jobs job_member)
noncomputable section
set_option autoImplicit false
set_option maxHeartbeats 5000000

theorem separate_symm {a b : Allocation} (h : a.separate b) : b.separate a := by
  rcases h with h | h | h
  · exact Or.inl (Ne.symm h)
  · exact Or.inr (Or.inr h)
  · exact Or.inr (Or.inl h)

theorem separate_byte {a b : Allocation} (h : a.separate b) (key : Key)
    (contains : b.contains key) (owner address : Nat) (covered : a.covers owner address) :
    ¬key.covers owner address := by
  rcases contains with ⟨ko,kl,kh⟩
  rcases covered with ⟨ao,al,ah⟩
  rintro ⟨bo,bl,bh⟩
  rcases h with h | h | h
  · exact h (ao.symm.trans (bo.trans ko))
  · omega
  · omega

theorem forward_frame {M α β κ : Type} [Inhabited κ] (ops : Ops M α β) (a : Storage.Adapter α κ)
    (p : ForwardLayout) (positive : 0<p.block) (before : World M) (kept : List Allocation)
    (backed : p.output.backed) (separate : ∀ old∈kept,old.separate p.output.view.storage) :
    Frame ops kept before {before with memory:=Storage.forwardKernel ops a before.memory p positive} := by
  refine ⟨fun _ h => h,?_,?_,rfl⟩
  · intro old ho key hk
    apply read_frame
    intro rc hrc
    obtain ⟨hr,hc⟩ := (job_member _ _ _ _).mp hrc
    exact old.spans_separate _ (separate old ho) key _ hk (backed rc.1 rc.2 hr hc)
  · intro old ho owner address covered
    apply Storage.forward_bytes
    intro r c hr hc
    exact separate_byte (separate old ho) _ (backed r c hr hc) owner address covered

theorem backward_frame {M α β κ : Type} [Inhabited κ] (ops : Ops M α β) (a : Storage.Adapter α κ)
    (p : BackwardLayout) (positive : 0<p.block) (before : World M) (kept : List Allocation)
    (backed : p.work.backed) (separate : ∀ old∈kept,old.separate p.work.view.storage) :
    Frame ops kept before {before with memory:=Storage.backwardKernel ops a before.memory p positive} := by
  refine ⟨fun _ h => h,?_,?_,rfl⟩
  · intro old ho key hk
    apply read_frame
    intro rc hrc
    obtain ⟨hr,hc⟩ := (job_member _ _ _ _).mp hrc
    exact old.spans_separate _ (separate old ho) key _ hk (backed rc.1 rc.2 hr hc)
  · intro old ho owner address covered
    apply Storage.backward_bytes
    intro r c hr hc
    exact separate_byte (separate old ho) _ (backed r c hr hc) owner address covered

#axiomsClean forward_frame
#axiomsClean backward_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivTriton.Host
