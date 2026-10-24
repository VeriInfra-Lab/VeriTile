import bench.repairs.tritonbench_g.layer_norm_fwd_nonzero_tile.Proofs.StorageExecutionReadback

namespace VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.StorageMachine
open VeriTile.Bench.Repairs.LayerNormFwd
open VeriTile.Bench.Repairs.LayerNormFwd.StorageMachine
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open Address PhysicalDomain NativeABI
open Compiler (Destination storeCoordinate)
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 12000

theorem physical_store_coordinate {α : Type} (c : Config α block)
    (artifact : StorageCompiler.Artifact c.profile c.flags c.N block) (memory : Memory c)
    (template : State α) (row : Fin c.M) (destination : Destination) :
    (artifact.stores destination).map (fun _ => TypedPhysical.read (layout c.allocations) memory (key c (row,destination)))=
      storeCoordinate c.flags (TypedSource.observe c.flags KernelEntry.regions c.strides c.M c.N
        (TypedPhysical.pull (layout c.allocations) memory template)) row destination := by
  rw [artifact.storeOperands]
  cases destination with
  | y col => by_cases h : col<c.N <;> simp [StorageCompiler.sourceStore,Compiler.sourceStore,storeCoordinate,
      TypedSource.observe,key,destinationRole,destinationCol,region,offset,h,TypedPhysical.pull]
  | y1 col =>
    cases hw : c.flags.w1 <;> by_cases h : col<c.N
    all_goals simp [StorageCompiler.sourceStore,Compiler.sourceStore,storeCoordinate,TypedSource.observe,
      key,destinationRole,destinationCol,region,offset,hw,h,TypedPhysical.pull]
  | residualOut col =>
    cases hr : c.flags.residualOut <;> by_cases h : col<c.N
    all_goals simp [StorageCompiler.sourceStore,Compiler.sourceStore,storeCoordinate,TypedSource.observe,
      key,destinationRole,destinationCol,region,offset,hr,h,TypedPhysical.pull]
  | mean => cases hr : c.flags.rms <;> simp [StorageCompiler.sourceStore,Compiler.sourceStore,storeCoordinate,
      TypedSource.observe,key,destinationRole,destinationCol,region,offset,hr,TypedPhysical.pull]
  | rstd => rfl
  | mask second col =>
    cases second <;> cases hd : c.flags.dropout <;> cases hm : c.flags.dropoutMask <;> cases hx : c.flags.x1 <;> by_cases h : col<c.N
    all_goals simp [StorageCompiler.sourceStore,Compiler.sourceStore,storeCoordinate,TypedSource.observe,
      key,destinationRole,destinationCol,region,offset,hd,hm,hx,h,TypedPhysical.pull]

/-- Actual execution consumes the emitted conversion, mask and operand, and
its stored coordinate equals the converted dependency-tree reference. -/
theorem execute_store_coordinate {α : Type} [Inhabited α] (c : Config α block)
    (artifact : StorageCompiler.Artifact c.profile c.flags c.N block) (memory : Memory c)
    (template : State α) (row : Fin c.M) (destination : Destination) :
    storeCoordinate c.flags (TypedSource.observe c.flags KernelEntry.regions c.strides c.M c.N
      (TypedPhysical.pull (layout c.allocations) (execute c artifact memory) template)) row destination=
    storeCoordinate c.flags (artifact.observe c.ops c.casts KernelEntry.regions c.strides c.M c.eps c.p
      (TypedPhysical.pull (layout c.allocations) memory template)) row destination := by
  rw [←physical_store_coordinate c artifact _ template row destination,
    ←artifact.observe_store c.ops c.casts KernelEntry.regions c.strides c.M c.eps c.p _ row destination]
  cases he : artifact.stores destination with
  | none => rfl
  | some operand =>
    simp only [Option.map_some]
    congr 1
    exact (execute_readback c artifact memory (row,destination) operand he).trans
      (by rw [snapshot_view c memory template])

/-- Every output, optional output, mask half, seed and fallback value from the
actual physical execution. Mean/rstd remain FP32 even when output storage has
a different requested dtype. -/
theorem execute_observe {α : Type} [Inhabited α] (c : Config α block)
    (artifact : StorageCompiler.Artifact c.profile c.flags c.N block) (memory : Memory c) (template : State α) :
    TypedSource.observe c.flags KernelEntry.regions c.strides c.M c.N
      (TypedPhysical.pull (layout c.allocations) (execute c artifact memory) template)=
    artifact.observe c.ops c.casts KernelEntry.regions c.strides c.M c.eps c.p
      (TypedPhysical.pull (layout c.allocations) memory template) := by
  have hc := execute_store_coordinate c artifact memory template
  have hi := run_input_read c artifact (program c) memory
  apply TypedSource.Result.ext
  · funext i j
    simpa [storeCoordinate,j.isLt] using hc i (.y j.val)
  · cases hf : c.flags.w1
    · simp [TypedSource.observe,StorageCompiler.Artifact.observe,hf]
    · simp only [TypedSource.observe,StorageCompiler.Artifact.observe,hf,ite_true]
      congr 1
      funext i j
      simpa [storeCoordinate,TypedSource.observe,StorageCompiler.Artifact.observe,hf,j.isLt] using hc i (.y1 j.val)
  · cases hf : c.flags.rms
    · simp only [TypedSource.observe,StorageCompiler.Artifact.observe,hf,Bool.false_eq_true,ite_false]
      congr 1
      funext i
      simpa [storeCoordinate,TypedSource.observe,StorageCompiler.Artifact.observe,hf] using hc i .mean
    · simp [TypedSource.observe,StorageCompiler.Artifact.observe,hf]
  · funext i
    simpa [storeCoordinate] using hc i .rstd
  · cases hf : c.flags.residualOut
    · simp only [TypedSource.observe,StorageCompiler.Artifact.observe,hf,Bool.false_eq_true,ite_false]
      funext i j
      exact hi KernelEntry.regions.x (by simp [inputRegion]) (i.val*c.strides.x+j.val)
    · funext i j
      simpa [storeCoordinate,TypedSource.observe,StorageCompiler.Artifact.observe,hf,j.isLt] using hc i (.residualOut j.val)
  · cases hf : c.flags.dropout
    · simp [TypedSource.observe,StorageCompiler.Artifact.observe,hf]
    · simp only [TypedSource.observe,StorageCompiler.Artifact.observe,hf,ite_true]
      congr 1
      funext i
      exact hi KernelEntry.regions.seeds (by simp [inputRegion]) i.val
  · cases hd : c.flags.dropout <;> cases hm : c.flags.dropoutMask
    all_goals simp only [TypedSource.observe,StorageCompiler.Artifact.observe,hd,hm,Bool.false_and,Bool.true_and,
      Bool.false_eq_true,ite_false,ite_true]
    congr 1
    funext i j
    simpa [storeCoordinate,TypedSource.observe,StorageCompiler.Artifact.observe,hd,hm,j.isLt] using hc i (.mask Bool.false j.val)
  · cases hd : c.flags.dropout <;> cases hm : c.flags.dropoutMask <;> cases hx : c.flags.x1
    all_goals simp only [TypedSource.observe,StorageCompiler.Artifact.observe,hd,hm,hx,Bool.false_and,Bool.true_and,
      Bool.false_eq_true,ite_false,ite_true]
    congr 1
    funext i j
    simpa [storeCoordinate,TypedSource.observe,StorageCompiler.Artifact.observe,hd,hm,hx,j.isLt] using hc i (.mask Bool.true j.val)

#axiomsClean physical_store_coordinate
#axiomsClean execute_store_coordinate
#axiomsClean execute_observe
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.StorageMachine
