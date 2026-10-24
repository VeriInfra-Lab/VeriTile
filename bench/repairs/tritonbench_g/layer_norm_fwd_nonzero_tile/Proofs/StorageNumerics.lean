import bench.repairs.tritonbench_g.layer_norm_fwd_nonzero_tile.Proofs.StorageCompiler

namespace VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.StorageCompiler
open VeriTile.Bench.Repairs.LayerNormFwd
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
noncomputable section
set_option maxHeartbeats 3000000

def castCell {α : Type} [Inhabited α] (casts : StoreCasts α) (dtype : Wrapper.DType) (cell : Cell α) : Cell α :=
  .mk .real (casts.convertFP32 dtype (cell.read .real))

def castResult {α : Type} [Inhabited α] (profile : Profile) (casts : StoreCasts α) (f : Flags)
    (result : TypedSource.Result α M N K) : TypedSource.Result α M N K :=
  { result with
    y := fun i j => castCell casts profile.y (result.y i j)
    y1 := result.y1.map (fun value i j => castCell casts profile.y (value i j))
    residual := if f.residualOut then (fun i j => castCell casts profile.residual (result.residual i j))
      else result.residual }

/-- Independent ordered FP reference plus exactly the source-required output
casts. Its arithmetic remains NativeParameters' owner-specific source trees;
no expression graph, actual emitted store or final memory is an input. -/
def reference {α : Type} [Inhabited α] (profile : Profile) (ops : NativeRow.Ops α) (casts : StoreCasts α)
    (schedule : NativeMachine.Schedule block) (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : α) (s : State α) : TypedSource.Result α M N (HostReturn.seedCount f M) :=
  castResult profile casts f
    (NativeSchedule.oracle ops schedule.reductions schedule.plan schedule.plan1 f r stride M N eps p s)

theorem Artifact.observe_cast_graph {α : Type} [Inhabited α] (artifact : Artifact profile f N block)
    (ops : NativeRow.Ops α) (casts : StoreCasts α) (r : Regions) (stride : Strides)
    (M : Nat) (eps p : α) (s : State α) :
    artifact.observe ops casts r stride M eps p s=
      castResult profile casts f (artifact.graph.observe ops r stride M eps p s) := by
  apply TypedSource.Result.ext
  all_goals simp only [Artifact.observe,Compiler.Artifact.observe,castResult,castCell,Cell.read_same]
  · cases h : f.w1 <;> simp [h,castCell]
  · cases h : f.residualOut <;> simp [h,castCell]

/-- Local opcode/ordered-operand/source-site bindings, including each variance
leaf's own mean owner, derive the independent requested-storage reference. -/
theorem Artifact.observe_eq {α : Type} [Inhabited α] (artifact : Artifact profile f N block)
    (ops : NativeRow.Ops α) (casts : StoreCasts α) (r : Regions) (stride : Strides)
    (M : Nat) (eps p : α) (s : State α) :
    artifact.observe ops casts r stride M eps p s=
      reference profile ops casts artifact.graph.schedule f r stride M N eps p s := by
  rw [artifact.observe_cast_graph ops casts r stride M eps p s,
    artifact.graph.observe_eq ops r stride M eps p s]
  rfl

#axiomsClean Artifact.observe_cast_graph
#axiomsClean Artifact.observe_eq
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.StorageCompiler
