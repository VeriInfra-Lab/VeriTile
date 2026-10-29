import bench.repairs.tritonbench_g.fused_recurrent_hgrn.Proofs.Native
import bench.repairs.tritonbench_g.fused_recurrent_hgrn.Proofs.Host

namespace VeriTile.Bench.Repairs.FusedRecurrentHgrn
open VeriTile.Triton FP.Structural
namespace Native
noncomputable section
set_option maxHeartbeats 10000000

def entry (memory : RegionName → Nat → BitVec 32) : State Term where
  mem := fun r a => .mk .real (.leaf r a (memory r a))
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 0
  undef := fun d _ _ => defaultValue d

def inputFromMemory (memory : RegionName → Nat → BitVec 32) (T D : Nat) : Input Term :=
  fun r bh t d =>
    let a := if r="h0" ∨ r="dht" then stateAddr bh D d else rowAddr bh T D t d
    .leaf r a (memory r a)

theorem entry_inputs (memory : RegionName → Nat → BitVec 32) (T D : Nat) :
    Execution.Inputs (entry memory) (inputFromMemory memory T D) T D := by
  constructor
  · intro r hr bh t d
    rcases hr with rfl|rfl|rfl <;> simp [Execution.read,entry,inputFromMemory,Cell.read]
  · intro r hr bh d
    rcases hr with rfl|rfl <;> simp [Execution.read,entry,inputFromMemory,Cell.read]

def upstreamTerms (r : RegionName) (words : Nat → BitVec 32) : Nat → Term := fun a => .leaf r a (words a)

/-- Public raw-word host result. The only numerical parameters are the three
scalar instructions and a value-independent local compiler schedule. There is
no assumed execution success, computed-output equation, or whole-gradient port. -/
theorem host_native (p : Instructions) (schedule : Schedule)
    (memory : RegionName → Nat → BitVec 32) (upstream finalUpstream : Nat → BitVec 32)
    (BH T D BDF BDB : Nat) (hF : 0 < BDF) (hB : 0 < BDB) (U S hasDo hasDht : Bool) :
    let input := inputFromMemory memory T D
    let copied := Execution.hostInput sourceOps input T D hasDo
      (upstreamTerms "do" upstream) (upstreamTerms "dht" finalUpstream)
    ∃ f b,
      Execution.gridRun sourceOps (forward T D BDF U S).surfaceBody
        (Execution.programs BH D BDF) (entry memory)=some f ∧
      Execution.gridRun sourceOps (backward T D BDB U (S && hasDht)).surfaceBody
        (Execution.programs BH D BDB)
        (Execution.installCotangents sourceOps f hasDo (upstreamTerms "do" upstream)
          (upstreamTerms "dht" finalUpstream))=some b ∧
      (∀ bh, bh < BH → ∀ t, t < T → ∀ d, d < D →
        observe p schedule b "o" (rowAddr bh T D t d)=eval p schedule (state sourceOps U input bh d (t+1)) ∧
        observe p schedule b "dx" (rowAddr bh T D t d)=eval p schedule (dx sourceOps (S && hasDht) copied T bh d t) ∧
        observe p schedule b "dg" (rowAddr bh T D t d)=eval p schedule (dg sourceOps U (S && hasDht) copied T bh d t)) ∧
      (S=Bool.true → ∀ bh, bh < BH → ∀ d, d < D →
        observe p schedule b "ht" (stateAddr bh D d)=eval p schedule (state sourceOps U input bh d T)) ∧
      (∀ r a, r≠("do" : RegionName) → r≠("dht" : RegionName) →
        ¬ Execution.ForwardRegion BH T D S r a → ¬ Execution.BackwardRegion BH T D r a →
        observe p schedule b r a=memory r a) := by
  dsimp only
  obtain ⟨f,b,hf,hb,hO,hG,hH,hframe⟩ := Execution.host_result sourceOps (inputFromMemory memory T D)
    BH T D BDF BDB hF hB U S hasDo hasDht (upstreamTerms "do" upstream)
    (upstreamTerms "dht" finalUpstream) (entry memory) (entry_inputs memory T D)
  refine ⟨f,b,hf,hb,?_,?_,?_⟩
  · intro bh hbh t ht d hd
    obtain ⟨hx,hg⟩ := hG bh hbh t ht d hd
    unfold observe
    rw [hO bh hbh t ht d hd,hx,hg]
    exact ⟨rfl,rfl,rfl⟩
  · intro hS bh hbh d hd
    unfold observe
    rw [hH hS bh hbh d hd]
    rfl
  · intro r a hdo hdht hnf hnb
    unfold observe
    rw [hframe r a hdo hdht hnf hnb]
    rfl

#axiomsClean host_native
#specNonCircular inputFromMemory avoiding [FP.Structural.exec,Execution.gridRun]
end
end Native
end VeriTile.Bench.Repairs.FusedRecurrentHgrn
