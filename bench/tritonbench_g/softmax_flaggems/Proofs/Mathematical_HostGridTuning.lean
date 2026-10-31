import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_HostGridDispatch
import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_SourceSupport

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.HostGrid
open VeriTile Triton FullSource
set_option maxHeartbeats 3000000

instance (storage : Storage) (one : Bool) : Decidable (SourceProfileSupported storage one) :=
  inferInstanceAs (Decidable (one=Bool.true ∨ storage≠.fp64))

def TrialsSupported (storage : Storage) (d : Dimensions) (trials : List (Fin 6)) : Prop :=
  ∀ choice ∈ trials, SourceProfileSupported storage (backwardConfig d choice).one

structure TrialError where
  choice : Fin 6
  compilerRejected : Bool
  beforeFailure : BlockState

theorem backwardValue_stable (s t : BlockState) (P G Y : RegionName)
    (d : Dimensions) (c : LaunchConfig) (hPY : P≠Y) (hGY : G≠Y)
    (hf : BlockState.WriteWithin (fullSlice Y (d.M*d.N*d.K)) s t)
    (hu : t.undef=s.undef) : backwardValue t P G d c=backwardValue s P G d c := by
  have hp : t.mem P=s.mem P := funext fun o ↦ (hf P o (by simp [fullSlice,hPY])).symm
  have hg : t.mem G=s.mem G := funext fun o ↦ (hf G o (by simp [fullSlice,hGY])).symm
  funext a
  unfold backwardValue
  split_ifs <;> simp only [backwardInnerValue,backwardNonInnerValue,
    RealBackwardInner.expected,RealBackwardNonInner.expected,RealMath.backwardPadded,
    RealBackwardInner.sample,RealBackwardNonInner.sample,BlockState.readMem,hp,hg,hu]

theorem forward_source_domain (waves : WavesAboveOne) (d : Dimensions) (sms : Nat) :
    ForwardSourceDomain (forwardConfig waves d sms).one d.N (forwardConfig waves d sms).B := by
  intro h
  rw [forwardConfig_one,oneTile] at h
  have hn : ¬ d.N≤(forwardConfig waves d sms).B := by simpa using h
  omega

#axiomsClean backwardValue_stable
#axiomsClean forward_source_domain
#specNonCircular backwardValue avoiding [run,exec]
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.HostGrid
