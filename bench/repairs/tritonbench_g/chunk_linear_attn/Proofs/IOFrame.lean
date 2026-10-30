import bench.repairs.tritonbench_g.chunk_linear_attn.Proofs.IOArray
import bench.repairs.tritonbench_g.chunk_linear_attn.Proofs.StageSafetyBasics

namespace VeriTile.Bench.Repairs.ChunkLinearAttn.IOFrame
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry (packed_lt)
noncomputable section
set_option maxHeartbeats 2000000
variable {α : Type} (c : KernelInterface.Config) (vals : ChannelValues α (IOLayout.inputs c.dims))

theorem output_outside (r : RegionName) (a : Nat)
    (outside : (IOLayout.realMemory α c).Outside vals r a)
    (o : Fin (IOLayout.outputs c.dims).length)
    (enabled : (o.val=4 → c.final=Bool.true) ∧ (o.val=5 → c.seed=Bool.true)) :
    r ≠ (IOLayout.outputs c.dims)[o].buf ∨ TileShape.product (IOLayout.outputs c.dims)[o].shape ≤ a := by
  by_cases hr : r=(IOLayout.outputs c.dims)[o].buf
  · right
    by_contra hn
    have ha : a < TileShape.product (IOLayout.outputs c.dims)[o].shape := by omega
    have h := outside.1 o (IOArray.index _ ⟨a,ha⟩) enabled
    exact h.elim (fun h => h hr) (fun h => h (IOArray.index_offset (IOLayout.outputs c.dims)[o].shape ⟨a,ha⟩).symm)
  · exact Or.inl hr

theorem scratch_outside (r : RegionName) (a : Nat)
    (outside : (IOLayout.realMemory α c).Outside vals r a)
    (o : Fin (IOLayout.scratch c.dims).length) :
    r ≠ (IOLayout.scratch c.dims)[o].buf ∨ TileShape.product (IOLayout.scratch c.dims)[o].shape ≤ a := by
  by_cases hr : r=(IOLayout.scratch c.dims)[o].buf
  · right
    by_contra hn
    have ha : a < TileShape.product (IOLayout.scratch c.dims)[o].shape := by omega
    have h := outside.2 o (IOArray.index _ ⟨a,ha⟩) trivial
    exact h.elim (fun h => h hr) (fun h => h (IOArray.index_offset (IOLayout.scratch c.dims)[o].shape ⟨a,ha⟩).symm)
  · exact Or.inl hr

theorem state_window (p : Parameters) (pid : PID) (rev final : Bool)
    (head : pid.z < p.B*p.H) (r : RegionName) (a : Nat) (hit : StateFootprint p pid rev final r a) :
    (r=historyName rev ∧ a < p.B*p.H*p.NT*p.K*p.V) ∨
      (r=finalName rev ∧ final=Bool.true ∧ a < p.B*p.H*p.K*p.V) := by
  rcases hit with ⟨hr,n,hn,j,hj,rfl⟩ | ⟨hr,hf,j,hj,rfl⟩
  · exact Or.inl ⟨hr,StageSafety.history_bound _ _ _ _ _ _ _ _ head hn hj.1 hj.2⟩
  · refine Or.inr ⟨hr,hf,?_⟩
    simpa only [fOffset,Nat.mul_assoc] using
      StageSafety.matrix_bound _ _ _ _ _ _ head hj.1 hj.2

theorem output_window (p : Parameters) (pid : PID) (head : pid.z < p.B*p.H)
    (r : RegionName) (a : Nat) (hit : OutputFootprint p pid r a) :
    r="output" ∧ a < p.B*p.H*p.T*p.V := by
  rcases hit with ⟨hr,j,hj,rfl⟩
  exact ⟨hr,StageSafety.matrix_bound _ _ _ _ _ _ head hj.1 hj.2⟩

theorem gradient_window (p : Parameters) (pid : PID)
    (block : pid.x < keyBlocks p) (head : pid.z < p.B*p.H)
    (r : RegionName) (a : Nat) (hit : GradientFootprint p pid r a) :
    ((r="queryGradient" ∨ r="keyGradient") ∧ a < p.B*p.H*p.T*p.K) ∨
      (r="valuePartials" ∧ a < keyBlocks p*(p.B*p.H)*p.T*p.V) := by
  rcases hit with ⟨hr,j,hj,rfl⟩ | ⟨hr,v,hv,j,hj,rfl⟩
  · exact Or.inl ⟨hr,StageSafety.matrix_bound _ _ _ _ _ _ head hj.1 hj.2⟩
  · exact Or.inr ⟨hr,StageSafety.matrix_bound _ _ _ _ _ _ (packed_lt block head) hj.1 hj.2⟩

/-- The visible IO exterior is contained in the original exact source frame;
all four complete grids and the finite reduction remain represented. -/
theorem outside (r : RegionName) (a : Nat)
    (outside : (IOLayout.realMemory α c).Outside vals r a) : KernelInterface.Outside c.call r a := by
  have out0 := output_outside c vals r a outside 0 (by constructor <;> simp)
  have out1 := output_outside c vals r a outside 1 (by constructor <;> simp)
  have out2 := output_outside c vals r a outside 2 (by constructor <;> simp)
  have out3 := output_outside c vals r a outside 3 (by constructor <;> simp)
  have scratch0 := scratch_outside c vals r a outside 0
  have scratch1 := scratch_outside c vals r a outside 1
  have scratch2 := scratch_outside c vals r a outside 2
  change r ≠ "output" ∨ (c.dims.B*c.dims.H)*((c.dims.T)*((c.dims.V)*(1))) ≤ a at out0
  change r ≠ "queryGradient" ∨ (c.dims.B*c.dims.H)*((c.dims.T)*((c.dims.K)*(1))) ≤ a at out1
  change r ≠ "keyGradient" ∨ (c.dims.B*c.dims.H)*((c.dims.T)*((c.dims.K)*(1))) ≤ a at out2
  change r ≠ "valueGradient" ∨ (c.dims.B*c.dims.H)*((c.dims.T)*((c.dims.V)*(1))) ≤ a at out3
  change r ≠ "history" ∨ (c.dims.B*c.dims.H)*((c.dims.NT)*((c.dims.K)*((c.dims.V)*(1)))) ≤ a at scratch0
  change r ≠ "historyGradient" ∨ (c.dims.B*c.dims.H)*((c.dims.NT)*((c.dims.K)*((c.dims.V)*(1)))) ≤ a at scratch1
  change r ≠ "valuePartials" ∨ (keyBlocks (c.dims.parameters Bool.true))*((c.dims.B*c.dims.H)*((c.dims.T)*((c.dims.V)*(1)))) ≤ a at scratch2
  simp only [Nat.mul_assoc,Nat.mul_one] at out0 out1 out2 out3 scratch0 scratch1 scratch2
  refine ⟨?_,?_,?_,?_,?_⟩
  · intro pid member hit
    have hv := (mem_grid _ _ _).mp member
    have hw := state_window c.call.forward pid Bool.false c.final hv.2.2 r a hit
    simp only [KernelInterface.Config.call,Real.Call.forward,Real.Call.backward,
      Host.Dimensions.parameters,historyName,finalName,if_true,if_false,Bool.false_eq_true,Nat.mul_assoc] at hw
    rcases hw with ⟨hr,ha⟩ | ⟨hr,hf,ha⟩
    · simpa only [KernelInterface.Config.call,Real.Call.forward,Host.Dimensions.parameters,
        historyName,Nat.mul_assoc] using scratch0.elim (fun hn => hn hr) (fun hn => Nat.not_lt_of_ge hn ha)
    · have out4 := output_outside c vals r a outside 4 ⟨fun _ => hf,by simp⟩
      change r ≠ "final" ∨ (c.dims.B*c.dims.H)*(c.dims.K*(c.dims.V*1)) ≤ a at out4
      simp only [Nat.mul_assoc,Nat.mul_one] at out4
      exact out4.elim (fun hn => hn hr) (fun hn => Nat.not_lt_of_ge hn ha)
  · intro pid member hit
    have hv := (mem_grid _ _ _).mp member
    have hw := output_window c.call.forward pid hv.2.2 r a hit
    simp only [KernelInterface.Config.call,Real.Call.forward,Host.Dimensions.parameters,Nat.mul_assoc] at hw
    exact out0.elim (fun hn => hn hw.1) (fun hn => Nat.not_lt_of_ge hn hw.2)
  · intro pid member hit
    have hv := (mem_grid _ _ _).mp member
    have hw := state_window c.call.backward pid Bool.true c.seed hv.2.2 r a hit
    simp only [KernelInterface.Config.call,Real.Call.forward,Real.Call.backward,
      Host.Dimensions.parameters,historyName,finalName,if_true,if_false,Bool.false_eq_true,Nat.mul_assoc] at hw
    rcases hw with ⟨hr,ha⟩ | ⟨hr,hf,ha⟩
    · exact scratch1.elim (fun hn => hn hr) (fun hn => Nat.not_lt_of_ge hn ha)
    · have out5 := output_outside c vals r a outside 5 ⟨by simp,fun _ => hf⟩
      change r ≠ "initialGradient" ∨ (c.dims.B*c.dims.H)*(c.dims.K*(c.dims.V*1)) ≤ a at out5
      simp only [Nat.mul_assoc,Nat.mul_one] at out5
      exact out5.elim (fun hn => hn hr) (fun hn => Nat.not_lt_of_ge hn ha)
  · intro pid member hit
    have hv := (mem_grid _ _ _).mp member
    have hw := gradient_window c.call.backward pid hv.1 hv.2.2 r a hit
    simp only [KernelInterface.Config.call,Real.Call.forward,Real.Call.backward,
      Host.Dimensions.parameters,historyName,finalName,if_true,if_false,Bool.false_eq_true,Nat.mul_assoc] at hw
    rcases hw with ⟨hr,ha⟩ | ⟨hr,ha⟩
    · rcases hr with hr|hr
      · exact out1.elim (fun hn => hn hr) (fun hn => Nat.not_lt_of_ge hn ha)
      · exact out2.elim (fun hn => hn hr) (fun hn => Nat.not_lt_of_ge hn ha)
    · exact scratch2.elim (fun hn => hn hr) (fun hn => Nat.not_lt_of_ge hn ha)
  · simpa only [valueCount,KernelInterface.Config.call,Real.Call.backward,Host.Dimensions.parameters,Nat.mul_assoc] using out3

#axiomsClean outside
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.IOFrame
