import bench.repairs.tritonbench_g.cross_entropy2.Proofs.InvocationWitness

namespace VeriTile.Bench.Repairs.CrossEntropy2.Invocation
open Source Floating
open Floating.Profiled
open VeriTile.Bench.Repairs.CrossEntropy1.Host (World)
open VeriTile.Bench.Repairs.CrossEntropy1.Physical (Layout)
noncomputable section
set_option maxHeartbeats 10000000

def offsetInputs (p : Parameters) : Parameters :=
  {p with logitsBase:=p.logitsBase+1,labelsBase:=p.labelsBase+1}

/-- The two keys coexist in one layout: moving the two input views by one
FP32/int32 element changes their pointer divisibility and no other key field. -/
theorem offset_inputs_key {σ : Type} (layout : Layout σ) (state : World Compiled σ) (p : Parameters)
    (aligned : Scheduled.keyFor layout state p=Scheduled.alignedKey) :
    Scheduled.keyFor layout state (offsetInputs p)=Scheduled.offset4Key := by
  have a0 : Scheduled.bytePointer layout state p.loss p.lossBase%16=0 := by
    have h := congrArg (fun key : Scheduled.JITKey => key.pointerDiv16[0]!) aligned
    simpa [Scheduled.keyFor,Scheduled.alignedKey] using h
  have a1 : Scheduled.bytePointer layout state p.lse p.lseBase%16=0 := by
    have h := congrArg (fun key : Scheduled.JITKey => key.pointerDiv16[1]!) aligned
    simpa [Scheduled.keyFor,Scheduled.alignedKey] using h
  have a2 : Scheduled.bytePointer layout state p.zLoss p.zLossBase%16=0 := by
    have h := congrArg (fun key : Scheduled.JITKey => key.pointerDiv16[2]!) aligned
    simpa [Scheduled.keyFor,Scheduled.alignedKey] using h
  have a3 : Scheduled.bytePointer layout state p.logits p.logitsBase%16=0 := by
    have h := congrArg (fun key : Scheduled.JITKey => key.pointerDiv16[3]!) aligned
    simpa [Scheduled.keyFor,Scheduled.alignedKey] using h
  have a4 : Scheduled.bytePointer layout state (VeriTile.Triton.Region.cast p.labels) p.labelsBase%16=0 := by
    have h := congrArg (fun key : Scheduled.JITKey => key.pointerDiv16[4]!) aligned
    simpa [Scheduled.keyFor,Scheduled.alignedKey] using h
  simp only [Scheduled.bytePointer] at a0 a1 a2 a3 a4
  have b3 : ¬((layout.buffer state.native p.logits).base+4*(p.logitsBase+1))%16=0 := by omega
  have b4 : ¬((layout.buffer state.native (VeriTile.Triton.Region.cast p.labels)).base+4*(p.labelsBase+1))%16=0 := by omega
  have changed : Scheduled.keyFor layout state (offsetInputs p)=
      {Scheduled.keyFor layout state p with pointerDiv16:=[Bool.true,Bool.true,Bool.true,Bool.false,Bool.false]} := by
    simp [Scheduled.keyFor,offsetInputs,Scheduled.bytePointer,a0,a1,a2,b3,b4]
  rw [changed,aligned]
  rfl

/-- A concrete local-setup test runtime. This is a satisfiability witness,
not an implementation or conformance claim about CUDA. -/
def successfulSetup {σ : Type} (rt : Host.Runtime Compiled σ) : Host.Runtime Compiled σ :=
  {rt with setupForward2:=fun _ _ state => .ok () state}

/-- Both different graph selections execute successfully under the same
runtime/backend and the same layout. The second key is derived by an actual
one-element offset, rather than assumed together with a contradictory global
fixed-profile contract. The arbitrary existing runtime supplies unused APIs. -/
theorem two_calls_coexist {σ : Type} (layout : Layout σ) (rt : Host.Runtime Compiled σ)
    (p : Parameters) (count : Nat) (state : World Compiled σ)
    (aligned : Scheduled.keyFor layout state p=Scheduled.alignedKey) (nonempty : p.rows≠0) :
    ∃ first second,
      forwardLaunch layout dualABI (successfulSetup rt) p count state=
        .ok (dualABI.device Scheduled.alignedKey) first ∧
      forwardLaunch layout dualABI (successfulSetup rt) (offsetInputs p) count state=
        .ok (dualABI.device Scheduled.offset4Key) second ∧
      dualABI.device Scheduled.alignedKey 65536 (by decide)≠
        dualABI.device Scheduled.offset4Key 65536 (by decide) := by
  have first := launch_uses_actual_key layout dualABI (successfulSetup rt) p count state state nonempty rfl
  rw [aligned] at first
  have second := launch_uses_actual_key layout dualABI (successfulSetup rt) (offsetInputs p) count state state nonempty rfl
  rw [offset_inputs_key layout state p aligned] at second
  exact ⟨_,_,first,second,same_width_different_graphs⟩

#axiomsClean offset_inputs_key
#axiomsClean two_calls_coexist
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.Invocation
