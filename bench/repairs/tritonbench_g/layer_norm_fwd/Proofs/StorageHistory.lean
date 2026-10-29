import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.StorageExecutionReturn

namespace VeriTile.Bench.Repairs.LayerNormFwd.StorageMachine
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open Address PhysicalDomain NativeABI WrapperNative
noncomputable section
set_option maxHeartbeats 3000000

inductive DispatchEvent (profile : StorageCompiler.Profile) (f : Flags) (N block : Nat) where
  | launch (artifact : StorageCompiler.Artifact profile f N block)
  | setupFailure (code : Nat)

structure DispatchResult {α : Type} (c : Config α block) where
  memory : Memory c
  error : Option Nat

def history {α : Type} [Inhabited α] (c : Config α block) :
    List (DispatchEvent c.profile c.flags c.N block) → Memory c → DispatchResult c
  | [],memory => ⟨memory,none⟩
  | .setupFailure code::_,memory => ⟨memory,some code⟩
  | .launch artifact::rest,memory => history c rest (execute c artifact memory)

def dispatch {α : Type} [Inhabited α] (c : Config α block)
    (benchmarks : List (DispatchEvent c.profile c.flags c.N block))
    (selected : StorageCompiler.Artifact c.profile c.flags c.N block) (memory : Memory c) : DispatchResult c :=
  let prior := history c benchmarks memory
  match prior.error with
  | some _ => prior
  | none => ⟨execute c selected prior.memory,none⟩

theorem history_input_read {α : Type} [Inhabited α] (c : Config α block)
    (benchmarks : List (DispatchEvent c.profile c.flags c.N block)) (memory : Memory c)
    (region : RegionName) (hr : inputRegion region) (offset : Nat) :
    TypedPhysical.read (layout c.allocations) (history c benchmarks memory).memory (region,offset)=
      TypedPhysical.read (layout c.allocations) memory (region,offset) := by
  induction benchmarks generalizing memory with
  | nil => rfl
  | cons event rest ih =>
    cases event with
    | setupFailure code => rfl
    | launch artifact =>
      exact (ih (execute c artifact memory)).trans
        (run_input_read c artifact (program c) memory region hr offset)

theorem history_snapshot {α : Type} [Inhabited α] (c : Config α block)
    (benchmarks : List (DispatchEvent c.profile c.flags c.N block)) (memory : Memory c) :
    snapshot c (history c benchmarks memory).memory=snapshot c memory := by
  induction benchmarks generalizing memory with
  | nil => rfl
  | cons event rest ih =>
    cases event with
    | setupFailure code => rfl
    | launch artifact => exact (ih (execute c artifact memory)).trans (run_snapshot c artifact (program c) memory)

theorem history_frame {α : Type} [Inhabited α] (c : Config α block)
    (benchmarks : List (DispatchEvent c.profile c.flags c.N block)) (memory : Memory c)
    (i : Fin (layout c.allocations).capacity)
    (h : ∀ k, AliasMemory.Writes c.flags preparedRegions c.strides c.M c.N k →
      (layout c.allocations).address k≠i.val) : (history c benchmarks memory).memory i=memory i := by
  induction benchmarks generalizing memory with
  | nil => rfl
  | cons event rest ih =>
    cases event with
    | setupFailure code => rfl
    | launch artifact => exact (ih (execute c artifact memory)).trans (run_frame c artifact (program c) memory i h)

theorem dispatch_frame {α : Type} [Inhabited α] (c : Config α block)
    (benchmarks : List (DispatchEvent c.profile c.flags c.N block))
    (selected : StorageCompiler.Artifact c.profile c.flags c.N block) (memory : Memory c)
    (i : Fin (layout c.allocations).capacity)
    (h : ∀ k, AliasMemory.Writes c.flags preparedRegions c.strides c.M c.N k →
      (layout c.allocations).address k≠i.val) : (dispatch c benchmarks selected memory).memory i=memory i := by
  cases he : (history c benchmarks memory).error with
  | some code => simpa only [dispatch,he] using history_frame c benchmarks memory i h
  | none =>
    simpa only [dispatch,he] using (run_frame c selected (program c) (history c benchmarks memory).memory i h).trans
      (history_frame c benchmarks memory i h)

theorem dispatch_input_read {α : Type} [Inhabited α] (c : Config α block)
    (benchmarks : List (DispatchEvent c.profile c.flags c.N block))
    (selected : StorageCompiler.Artifact c.profile c.flags c.N block) (memory : Memory c)
    (region : RegionName) (hr : inputRegion region) (offset : Nat) :
    TypedPhysical.read (layout c.allocations) (dispatch c benchmarks selected memory).memory (region,offset)=
      TypedPhysical.read (layout c.allocations) memory (region,offset) := by
  cases he : (history c benchmarks memory).error with
  | some code => simpa only [dispatch,he] using history_input_read c benchmarks memory region hr offset
  | none =>
    simpa only [dispatch,he] using (run_input_read c selected (program c) (history c benchmarks memory).memory region hr offset).trans
      (history_input_read c benchmarks memory region hr offset)

theorem history_observe {α : Type} [Inhabited α] (c : Config α block)
    (benchmarks : List (DispatchEvent c.profile c.flags c.N block))
    (selected : StorageCompiler.Artifact c.profile c.flags c.N block) (memory : Memory c) (template : State α) :
    selected.observe c.ops c.casts preparedRegions c.strides c.M c.eps c.p
      (TypedPhysical.pull (layout c.allocations) (history c benchmarks memory).memory template)=
    selected.observe c.ops c.casts preparedRegions c.strides c.M c.eps c.p
      (TypedPhysical.pull (layout c.allocations) memory template) := by
  apply TypedSource.Result.ext
  all_goals simp only [StorageCompiler.Artifact.observe,←snapshot_view,history_snapshot]
  · cases hf : c.flags.residualOut <;> simp only [hf,Bool.false_eq_true,ite_false,ite_true]
    funext i j
    exact history_input_read c benchmarks memory preparedRegions.x (by simp [inputRegion]) _
  · cases hf : c.flags.dropout <;> simp only [hf,Bool.false_eq_true,ite_false,ite_true]
    congr 1
    funext i
    exact history_input_read c benchmarks memory preparedRegions.seeds (by simp [inputRegion]) _

theorem dispatch_return {α : Type} [Inhabited α] (c : Config α block)
    (benchmarks : List (DispatchEvent c.profile c.flags c.N block))
    (selected : StorageCompiler.Artifact c.profile c.flags c.N block) (memory : Memory c) (template : State α)
    (h : (history c benchmarks memory).error=none) :
    TypedSource.observe c.flags preparedRegions c.strides c.M c.N
      (TypedPhysical.pull (layout c.allocations) (dispatch c benchmarks selected memory).memory template)=
    selected.observe c.ops c.casts preparedRegions c.strides c.M c.eps c.p
      (TypedPhysical.pull (layout c.allocations) memory template) := by
  simp only [dispatch,h]
  exact (execute_observe c selected (history c benchmarks memory).memory template).trans
    (history_observe c benchmarks selected memory template)

#axiomsClean history_input_read
#axiomsClean history_snapshot
#axiomsClean history_frame
#axiomsClean dispatch_frame
#axiomsClean dispatch_input_read
#axiomsClean history_observe
#axiomsClean dispatch_return
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.StorageMachine
