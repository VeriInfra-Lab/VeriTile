import bench.tritonbench_g.chunk_linear_attn.Mathematical
import bench.tritonbench_g.chunk_linear_attn.Proofs.IOArray

namespace VeriTile.Bench.Repairs.ChunkLinearAttn.IOReal
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 1500000
variable (c : KernelInterface.Config) (vals : ChannelValues ℝ (IOLayout.inputs c.dims))
    (before : State ℝ) (binding : (IOLayout.realMemory ℝ c).InputBinding id vals before)
include binding

theorem inputs_same (backward : Bool) (head : Fin (c.dims.B*c.dims.H)) :
    Real.inputs (c.dims.parameters backward) head.val before.mem c.seed = Mathematical.inputs c vals head := by
  unfold Real.inputs Mathematical.inputs
  congr 1
  · funext t k
    have value := binding 0 (head,t,k,PUnit.unit) (by constructor <;> simp)
    change (before.mem "query" (TileShape.linearIndex [c.dims.B*c.dims.H,c.dims.T,c.dims.K]
      (head,t,k,PUnit.unit))).read .real = vals 0 (head,t,k,PUnit.unit) at value
    rw [IOArray.offset3] at value
    simpa only [Real.query,Host.Dimensions.parameters,Nat.add_mul,Nat.mul_assoc,
      Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using value
  · funext t k
    have value := binding 1 (head,t,k,PUnit.unit) (by constructor <;> simp)
    change (before.mem "key" (TileShape.linearIndex [c.dims.B*c.dims.H,c.dims.T,c.dims.K]
      (head,t,k,PUnit.unit))).read .real = vals 1 (head,t,k,PUnit.unit) at value
    rw [IOArray.offset3] at value
    simpa only [Real.key,Host.Dimensions.parameters,Nat.add_mul,Nat.mul_assoc,
      Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using value
  · funext t v
    have value := binding 2 (head,t,v,PUnit.unit) (by constructor <;> simp)
    change (before.mem "value" (TileShape.linearIndex [c.dims.B*c.dims.H,c.dims.T,c.dims.V]
      (head,t,v,PUnit.unit))).read .real = vals 2 (head,t,v,PUnit.unit) at value
    rw [IOArray.offset3] at value
    simpa only [Real.value,Host.Dimensions.parameters,Nat.add_mul,Nat.mul_assoc,Nat.add_assoc] using value
  · funext k v
    unfold Real.endpoint
    by_cases seed : c.seed=Bool.true
    · have value := binding 4 (head,k,v,PUnit.unit) ⟨fun _ => seed,by simp⟩
      change (before.mem "initial" (TileShape.linearIndex [c.dims.B*c.dims.H,c.dims.K,c.dims.V]
        (head,k,v,PUnit.unit))).read .real = vals 4 (head,k,v,PUnit.unit) at value
      rw [IOArray.offset3] at value
      simpa only [seed,if_true,seedName,Host.Dimensions.parameters,Nat.add_mul,Nat.mul_assoc,Nat.add_assoc] using value
    · simp [seed]

theorem incoming_same (head : Fin (c.dims.B*c.dims.H)) :
    (fun t v => Real.incoming c.call.backward head.val before.mem t.val v.val) = Mathematical.incoming c vals head := by
  funext t v
  have value := binding 3 (head,t,v,PUnit.unit) (by constructor <;> simp)
  change (before.mem "incoming" (TileShape.linearIndex [c.dims.B*c.dims.H,c.dims.T,c.dims.V]
    (head,t,v,PUnit.unit))).read .real = vals 3 (head,t,v,PUnit.unit) at value
  rw [IOArray.offset3] at value
  simpa only [Real.incoming,Real.Call.backward,KernelInterface.Config.call,Mathematical.incoming,
    Host.Dimensions.parameters,Nat.add_mul,Nat.mul_assoc,Nat.add_assoc] using value

theorem terminal_same (head : Fin (c.dims.B*c.dims.H)) :
    (fun k v => Real.endpoint c.call.backward head.val before.mem Bool.true c.terminal k.val v.val) =
      Mathematical.terminal c vals head := by
  funext k v
  unfold Real.endpoint Mathematical.terminal
  by_cases terminal : c.terminal=Bool.true
  · have value := binding 5 (head,k,v,PUnit.unit) ⟨by simp,fun _ => terminal⟩
    change (before.mem "terminalGradient" (TileShape.linearIndex [c.dims.B*c.dims.H,c.dims.K,c.dims.V]
      (head,k,v,PUnit.unit))).read .real = vals 5 (head,k,v,PUnit.unit) at value
    rw [IOArray.offset3] at value
    simpa only [terminal,if_true,seedName,Real.Call.backward,KernelInterface.Config.call,
      Host.Dimensions.parameters,Nat.add_mul,Nat.mul_assoc,Nat.add_assoc] using value
  · simp [terminal]

theorem gradient_same (head : Fin (c.dims.B*c.dims.H)) :
    c.call.target head.val before.mem = Mathematical.gradient c vals head := by
  unfold Real.Call.target
  change Real.gradients c.scale (Real.inputs (c.dims.parameters Bool.true) head.val before.mem c.seed)
    (fun t v => Real.incoming c.call.backward head.val before.mem t.val v.val)
    (fun k v => Real.endpoint c.call.backward head.val before.mem Bool.true c.terminal k.val v.val) = _
  unfold Mathematical.gradient
  congr 1
  · exact inputs_same c vals before binding Bool.true head
  · exact incoming_same c vals before binding head
  · exact terminal_same c vals before binding head

theorem outputs (after : State ℝ) (result : KernelExecution.RealResult c.call before.mem after.mem) :
    (IOLayout.realMemory ℝ c).OutputBinding (fun _ _ => id) vals (Mathematical.real c vals) after := by
  intro o j enabled
  fin_cases o
  all_goals obtain ⟨head,t,v,u⟩ := j; cases u
  · have out := result.output head t v
    dsimp only [KernelInterface.Config.call,Real.Call.forward] at out
    rw [inputs_same c vals before binding Bool.false head] at out
    change mapCell id (after.mem "output" (TileShape.linearIndex [c.dims.B*c.dims.H,c.dims.T,c.dims.V] (head,t,v,PUnit.unit))) = _
    rw [IOArray.offset3,out]
    rfl
  · have out := result.query head t v
    rw [gradient_same c vals before binding head] at out
    dsimp only [KernelInterface.Config.call] at out
    change mapCell id (after.mem "queryGradient" (TileShape.linearIndex [c.dims.B*c.dims.H,c.dims.T,c.dims.K] (head,t,v,PUnit.unit))) = _
    rw [IOArray.offset3,out]
    rfl
  · have out := result.key head t v
    rw [gradient_same c vals before binding head] at out
    dsimp only [KernelInterface.Config.call] at out
    change mapCell id (after.mem "keyGradient" (TileShape.linearIndex [c.dims.B*c.dims.H,c.dims.T,c.dims.K] (head,t,v,PUnit.unit))) = _
    rw [IOArray.offset3,out]
    rfl
  · have out := result.value head t v
    rw [gradient_same c vals before binding head] at out
    dsimp only [KernelInterface.Config.call] at out
    change mapCell id (after.mem "valueGradient" (TileShape.linearIndex [c.dims.B*c.dims.H,c.dims.T,c.dims.V] (head,t,v,PUnit.unit))) = _
    rw [IOArray.offset3,out]
    rfl
  · have out := result.final (enabled.1 rfl) head t v
    dsimp only [KernelInterface.Config.call,Real.Call.forward] at out
    rw [inputs_same c vals before binding Bool.false head] at out
    change mapCell id (after.mem "final" (TileShape.linearIndex [c.dims.B*c.dims.H,c.dims.K,c.dims.V] (head,t,v,PUnit.unit))) = _
    rw [IOArray.offset3,out]
    rfl
  · have out := result.initial (enabled.2 rfl) head t v
    rw [gradient_same c vals before binding head] at out
    dsimp only [KernelInterface.Config.call] at out
    change mapCell id (after.mem "initialGradient" (TileShape.linearIndex [c.dims.B*c.dims.H,c.dims.K,c.dims.V] (head,t,v,PUnit.unit))) = _
    rw [IOArray.offset3,out]
    rfl

#axiomsClean inputs_same
#axiomsClean outputs
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.IOReal
