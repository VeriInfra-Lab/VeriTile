import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.Forward
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.Reverse
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.Recompute
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.Memory
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.Native
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.WholeSource
noncomputable section
set_option maxHeartbeats 5000000
def forwardReference {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α)
    (bh iv : Nat) (m : Memory α) :=
  if f.final then stateStore s p (Forward.memory s f p g scale m bh iv s.T) .ht bh iv s.FBV (Forward.h s f p g m bh iv s.T)
    else Forward.memory s f p g scale m bh iv s.T
def backwardReference {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α)
    (bh iv : Nat) (m : Memory α) :=
  let reverseMem:=Reverse.memory s f p g scale m bh iv s.T
  let reverseH:=Reverse.h s f p g scale m bh iv s.T
  let mid:=if f.initialGrad then stateStore s p reverseMem .dh0 bh iv s.BBV reverseH else reverseMem
  Recompute.memory s f p g scale mid bh iv s.T
theorem forward_program {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α)
    (bh iv : Nat) (m : Memory α) : forwardProgram s f p g scale bh iv m=forwardReference s f p g scale bh iv m :=
  Forward.program_full s f p g scale m bh iv
theorem backward_program {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α)
    (bh iv : Nat) (m : Memory α) : backwardProgram s f p g scale bh iv m=backwardReference s f p g scale bh iv m := by
  simp only [backwardProgram,Reverse.loop_full s f p g scale m bh iv s.T le_rfl,Recompute.loop_full,backwardReference]
def forwardGridReference {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α)
    (m : Memory α) := (jobs s.BH s.FNV).foldl (fun m job => forwardReference s f p g scale job.1 job.2 m) m
def backwardGridReference {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α)
    (m : Memory α) := (jobs s.BH s.BNV).foldl (fun m job => backwardReference s f p g scale job.1 job.2 m) m
theorem forward_grid {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α)
    (m : Memory α) : forwardGrid s f p g scale m=forwardGridReference s f p g scale m := by
  simp only [forwardGrid,forwardGridReference,forward_program]
theorem backward_grid {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α)
    (m : Memory α) : backwardGrid s f p g scale m=backwardGridReference s f p g scale m := by
  simp only [backwardGrid,backwardGridReference,backward_program]
def forwardCallReference {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α)
    (m : Memory α) := forwardGridReference s f p g scale (prepareBeta s f p m)
def backwardCallReference {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α)
    (m : Memory α) := hostReduce s f p g (backwardGridReference s f p g scale (prepareDbeta s f p m))
theorem forward_call {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α)
    (m : Memory α) : forwardCall s f p g scale m=forwardCallReference s f p g scale m :=
  forward_grid s f p g scale _
theorem backward_call {α : Type} (s : Shape) (f : Flags) (p : Ops α) (g : Trees) (scale : α)
    (m : Memory α) : backwardCall s f p g scale m=backwardCallReference s f p g scale m := by
  simp only [backwardCall,backwardCallReference,backward_grid]
def forwardSoftwareReference (s : Shape) (f : Flags) (g : Trees) (scale : Native.Word)
    (policy : Native.Policy) (m : Memory Native.Word) :=
  Native.evaluate Native.software policy
    (forwardCallReference s f Native.graphOps g (.bits scale) (Native.graphMemory m))
def backwardSoftwareReference (s : Shape) (f : Flags) (g : Trees) (scale : Native.Word)
    (policy : Native.Policy) (m : Memory Native.Word) :=
  Native.evaluate Native.software policy
    (backwardCallReference s f Native.graphOps g (.bits scale) (Native.graphMemory m))
theorem software_forward (s : Shape) (f : Flags) (g : Trees) (scale : Native.Word)
    (policy : Native.Policy) (m : Memory Native.Word) :
    Native.forward s f g scale Native.software policy m=forwardSoftwareReference s f g scale policy m := by
  simp only [Native.forward,forwardSoftwareReference,forward_call]
theorem software_backward (s : Shape) (f : Flags) (g : Trees) (scale : Native.Word)
    (policy : Native.Policy) (m : Memory Native.Word) :
    Native.backward s f g scale Native.software policy m=backwardSoftwareReference s f g scale policy m := by
  simp only [Native.backward,backwardSoftwareReference,backward_call]

#specNonCircular forwardSoftwareReference avoiding [forwardCall,forwardGrid,forwardProgram,forwardLoop,forwardStep,Native.forward]
#specNonCircular backwardSoftwareReference avoiding [backwardCall,backwardGrid,backwardProgram,reverseLoop,reverseStep,recomputeLoop,recomputeStep,Native.backward]
#axiomsClean backward_program
#axiomsClean forward_call
#axiomsClean backward_call
#axiomsClean software_forward
#axiomsClean software_backward
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.WholeSource
