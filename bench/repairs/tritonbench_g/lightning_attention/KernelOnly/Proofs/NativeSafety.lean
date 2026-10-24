import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.Proofs.NativeMaterialization
import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.Proofs.StageSafety

namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly.NativeSafety
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.LightningAttention.Native (Word Instructions Plan)
open VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry (packed_lt)
noncomputable section
set_option maxHeartbeats 6000000

theorem forwardBounds (ops : Instructions) (plan : Plan) (d : Host.Dimensions)
    (vals : (NativeIO.forward ops plan d).Inputs) (bounds : RegionName → Nat)
    (backed : (NativeIO.forward ops plan d).InBounds bounds vals)
    (pid : PID) (hp : ValidPID .forward d pid) :
    SourceSafety.ForwardBounds d.parameters pid bounds := by
  refine ⟨?_,?_,?_,?_⟩
  · intro t a ht ha
    exact backed.2.1 ⟨0,by change 0<4; decide⟩ (⟨pid.head,hp.1⟩,⟨t,ht⟩,⟨a,ha⟩,()) True.intro
  · intro t a ht ha
    exact backed.2.1 ⟨1,by change 1<4; decide⟩ (⟨pid.head,hp.1⟩,⟨t,ht⟩,⟨a,ha⟩,()) True.intro
  · intro t c ht hc
    have h := backed.2.1 ⟨2,by change 2<4; decide⟩
      (⟨pid.head,hp.1⟩,⟨t,ht⟩,⟨pid.channel*d.BM+c,packed_lt hp.2 hc⟩,()) True.intro
    simpa only [NativeIO.forward,NativeIO.forwardInputIndex,NativeIO.paddedIndex,
      Host.Dimensions.parameters,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using h
  · intro t c ht hc
    have h := backed.2.2.1 ⟨0,by change 0<1; decide⟩
      (⟨pid.head,hp.1⟩,⟨t,ht⟩,⟨pid.channel*d.BM+c,packed_lt hp.2 hc⟩,()) True.intro
    simpa only [NativeIO.forward,NativeIO.paddedIndex,
      Host.Dimensions.parameters,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using h

theorem backwardBounds (ops : Instructions) (plan : Plan) (d : Host.Dimensions)
    (vals : (NativeIO.backward ops plan d).Inputs) (bounds : RegionName → Nat)
    (backed : (NativeIO.backward ops plan d).InBounds bounds vals)
    (head : Nat) (hh : head<d.B*d.H) :
    SourceSafety.HeadBounds d.parameters head bounds := by
  refine ⟨?_,?_,?_,?_,?_,?_,?_⟩
  · intro t a ht ha
    exact backed.2.1 ⟨0,by change 0<7; decide⟩ (⟨head,hh⟩,⟨t,ht⟩,⟨a,ha⟩,()) True.intro
  · intro t a ht ha
    exact backed.2.1 ⟨1,by change 1<7; decide⟩ (⟨head,hh⟩,⟨t,ht⟩,⟨a,ha⟩,()) True.intro
  · intro t a ht ha
    exact backed.2.1 ⟨2,by change 2<7; decide⟩ (⟨head,hh⟩,⟨t,ht⟩,⟨a,ha⟩,()) True.intro
  · intro t a ht ha
    exact backed.2.1 ⟨3,by change 3<7; decide⟩ (⟨head,hh⟩,⟨t,ht⟩,⟨a,ha⟩,()) True.intro
  · intro t a ht ha
    exact backed.2.1 ⟨4,by change 4<7; decide⟩ (⟨head,hh⟩,⟨t,ht⟩,⟨a,ha⟩,()) True.intro
  · intro t a ht ha
    exact backed.2.1 ⟨5,by change 5<7; decide⟩ (⟨head,hh⟩,⟨t,ht⟩,⟨a,ha⟩,()) True.intro
  · intro t a ht ha
    exact backed.2.1 ⟨6,by change 6<7; decide⟩ (⟨head,hh⟩,⟨t,ht⟩,⟨a,ha⟩,()) True.intro

theorem schedule (ops : Instructions) (plan : Plan) (stage : Stage)
    (d : Host.Dimensions) (pids : List PID) (bounds : RegionName → Nat)
    (backed : ∀ pid∈pids, SourceSafety.StageBounds stage d pid bounds)
    (memory : Memory Word) :
    MaterializedScheduleSafe bounds (pids.map (NativeIO.launch ops plan stage d)) memory := by
  induction pids generalizing memory with
  | nil => trivial
  | cons pid rest ih =>
    refine ⟨?_,?_⟩
    · exact SourceSafety.invocation_safe Native.algebra nativeMasks Native.dots stage d pid bounds
        (backed pid (by simp)) ((NativeIO.launch ops plan stage d pid).embedded memory)
    · intro next _
      exact ih (fun id hi => backed id (by simp [hi])) next

theorem forward (ops : Instructions) (plan : Plan) (d : Host.Dimensions)
    (vals : (NativeIO.forward ops plan d).Inputs) (bounds : RegionName → Nat)
    (backed : (NativeIO.forward ops plan d).InBounds bounds vals) (memory : Memory Word) :
    MaterializedScheduleSafe bounds ((NativeIO.forward ops plan d).schedule vals) memory :=
  schedule ops plan .forward d (grid .forward d) bounds
    (fun pid hp => forwardBounds ops plan d vals bounds backed pid ((mem_grid _ _ _).mp hp)) memory

theorem backward (ops : Instructions) (plan : Plan) (d : Host.Dimensions)
    (vals : (NativeIO.backward ops plan d).Inputs) (bounds : RegionName → Nat)
    (backed : (NativeIO.backward ops plan d).InBounds bounds vals) (memory : Memory Word) :
    MaterializedScheduleSafe bounds ((NativeIO.backward ops plan d).schedule vals) memory := by
  apply (materializedScheduleSafe_append bounds (NativeIO.schedule ops plan .intra d)
    (NativeIO.schedule ops plan .inter d) memory).mpr
  refine ⟨?_,?_⟩
  · exact schedule ops plan .intra d (grid .intra d) bounds
      (fun pid hp => backwardBounds ops plan d vals bounds backed pid.head
        ((mem_grid _ _ _).mp hp).1) memory
  · intro next _
    exact schedule ops plan .inter d (grid .inter d) bounds
      (fun pid hp => backwardBounds ops plan d vals bounds backed pid.head
        ((mem_grid _ _ _).mp hp).1) next

#axiomsClean forward
#axiomsClean backward
end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly.NativeSafety
