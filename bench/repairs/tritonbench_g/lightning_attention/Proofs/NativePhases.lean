import bench.repairs.tritonbench_g.lightning_attention.Proofs.NativeComposition
namespace VeriTile.Bench.Repairs.LightningAttention.Native
open VeriTile Triton FP.Structural
noncomputable section

/-- Forward's graph consumes only its current memory and actual forward grid. -/
def forwardReference (p : Instructions) (plan : Plan) (d : Host.Dimensions)
    (m : WordMemory) : WordMemory :=
  gridReference p plan .forward d (grid .forward d) m

/-- Backward begins at its own later memory. Intra consumes that memory and
inter consumes the actual newly written intra gradients. -/
def backwardReference (p : Instructions) (plan : Plan) (d : Host.Dimensions)
    (later : WordMemory) : WordMemory :=
  gridReference p plan .inter d (grid .inter d)
    (gridReference p plan .intra d (grid .intra d) later)

def forwardExecute (p : Instructions) (plan : Plan) (d : Host.Dimensions)
    (m : WordMemory) : Option WordMemory :=
  gridExecute p plan .forward d (grid .forward d) m

def backwardExecute (p : Instructions) (plan : Plan) (d : Host.Dimensions)
    (later : WordMemory) : Option WordMemory := do
  let intra ← gridExecute p plan .intra d (grid .intra d) later
  gridExecute p plan .inter d (grid .inter d) intra

theorem forward_execution (p : Instructions) (plan : Plan) (d : Host.Dimensions)
    (m : WordMemory) : forwardExecute p plan d m = some (forwardReference p plan d m) :=
  grid_execution _ _ _ _ _ _

theorem backward_execution (p : Instructions) (plan : Plan) (d : Host.Dimensions)
    (later : WordMemory) : backwardExecute p plan d later = some (backwardReference p plan d later) := by
  simp only [backwardExecute,grid_execution,bind,Option.bind_some,backwardReference]

def ForwardWritten (d : Host.Dimensions) (r : RegionName) (o : Nat) : Prop :=
  ∃ pid∈grid .forward d, Footprint .forward d pid r o

def BackwardWritten (d : Host.Dimensions) (r : RegionName) (o : Nat) : Prop :=
  (∃ pid∈grid .intra d, Footprint .intra d pid r o) ∨
  (∃ pid∈grid .inter d, Footprint .inter d pid r o)

theorem forward_frame (p : Instructions) (plan : Plan) (d : Host.Dimensions)
    (m : WordMemory) (r : RegionName) (o : Nat) (outside : ¬ForwardWritten d r o) :
    forwardReference p plan d m r o = m r o :=
  grid_frame _ _ _ _ _ _ _ _ (fun pid hp hw => outside ⟨pid,hp,hw⟩)

theorem backward_frame (p : Instructions) (plan : Plan) (d : Host.Dimensions)
    (later : WordMemory) (r : RegionName) (o : Nat) (outside : ¬BackwardWritten d r o) :
    backwardReference p plan d later r o = later r o := by
  unfold backwardReference
  rw [grid_frame _ _ _ _ _ _ _ _ (fun pid hp hw => outside (Or.inr ⟨pid,hp,hw⟩)),
    grid_frame _ _ _ _ _ _ _ _ (fun pid hp hw => outside (Or.inl ⟨pid,hp,hw⟩))]

theorem forward_inputs (p : Instructions) (plan : Plan) (d : Host.Dimensions)
    (m : WordMemory) (r : RegionName) (o : Nat)
    (hr : r∈(["query","key","value","incoming"] : List RegionName)) :
    forwardReference p plan d m r o = m r o := by
  apply forward_frame
  rintro ⟨pid,hp,hw⟩
  exact input_not_written _ _ _ _ _ hr hw

theorem backward_inputs (p : Instructions) (plan : Plan) (d : Host.Dimensions)
    (later : WordMemory) (r : RegionName) (o : Nat)
    (hr : r∈(["query","key","value","incoming"] : List RegionName)) :
    backwardReference p plan d later r o = later r o := by
  apply backward_frame
  rintro (⟨pid,hp,hw⟩ | ⟨pid,hp,hw⟩) <;>
    exact input_not_written _ _ _ _ _ hr hw

theorem forward_only_output (p : Instructions) (plan : Plan) (d : Host.Dimensions)
    (m : WordMemory) (r : RegionName) (o : Nat) (hr : r≠"output") :
    forwardReference p plan d m r o = m r o := by
  apply forward_frame
  rintro ⟨pid,hp,hw⟩
  exact hr hw.1

theorem backward_preserves_output (p : Instructions) (plan : Plan) (d : Host.Dimensions)
    (later : WordMemory) (o : Nat) :
    backwardReference p plan d later "output" o = later "output" o := by
  apply backward_frame
  rintro (⟨pid,hp,hw⟩ | ⟨pid,hp,hw⟩)
  · rcases hw with ⟨h,_⟩ | ⟨h,_⟩ | ⟨h,_⟩ <;> simp at h
  · rcases hw with ⟨h,_⟩ | ⟨h,_⟩ | ⟨h,_⟩ <;> simp at h

theorem forward_native (actual selected : Instructions) (plan : Plan) (d : Host.Dimensions)
    (m : WordMemory) (numeric : LocalConformance actual selected) :
    forwardExecute actual plan d m = some (forwardReference selected plan d m) ∧
    (∀ r o, ¬ForwardWritten d r o → forwardReference selected plan d m r o = m r o) := by
  rw [instructions_eq actual selected numeric]
  exact ⟨forward_execution _ _ _ _,forward_frame _ _ _ _⟩

theorem backward_native (actual selected : Instructions) (plan : Plan) (d : Host.Dimensions)
    (later : WordMemory) (numeric : LocalConformance actual selected) :
    backwardExecute actual plan d later = some (backwardReference selected plan d later) ∧
    (∀ r o, ¬BackwardWritten d r o → backwardReference selected plan d later r o = later r o) := by
  rw [instructions_eq actual selected numeric]
  exact ⟨backward_execution _ _ _ _,backward_frame _ _ _ _⟩

structure ForwardRuntime where
  launch : WordMemory → Option WordMemory

structure BackwardRuntime where
  intra : WordMemory → Option WordMemory
  inter : WordMemory → Option WordMemory

/-- Forward binding has no backward runtime, future cotangent, or reference
numerics. It binds the actual selected source/layout graph on the actual grid. -/
structure ForwardBinding (actual : Instructions) (plan : Plan) (d : Host.Dimensions)
    (runtime : ForwardRuntime) : Prop where
  launch : ∀ m, runtime.launch m = some (forwardReference actual plan d m)

/-- Backward's ordinary binding starts from an independently supplied later
memory and preserves the intra-to-inter current-memory handoff. -/
structure BackwardBinding (actual : Instructions) (plan : Plan) (d : Host.Dimensions)
    (runtime : BackwardRuntime) : Prop where
  intra : ∀ m, runtime.intra m = some (gridReference actual plan .intra d (grid .intra d) m)
  inter : ∀ m, runtime.inter m = some (gridReference actual plan .inter d (grid .inter d) m)

def runtimeForward (runtime : ForwardRuntime) (m : WordMemory) : Option WordMemory :=
  runtime.launch m

def runtimeBackward (runtime : BackwardRuntime) (later : WordMemory) : Option WordMemory := do
  let intra ← runtime.intra later
  runtime.inter intra

theorem forward_runtime (actual : Instructions) (plan : Plan) (d : Host.Dimensions)
    (runtime : ForwardRuntime) (m : WordMemory) (binding : ForwardBinding actual plan d runtime) :
    runtimeForward runtime m = some (forwardReference actual plan d m) := binding.launch m

theorem backward_runtime (actual : Instructions) (plan : Plan) (d : Host.Dimensions)
    (runtime : BackwardRuntime) (later : WordMemory) (binding : BackwardBinding actual plan d runtime) :
    runtimeBackward runtime later = some (backwardReference actual plan d later) := by
  simp only [runtimeBackward,binding.intra,binding.inter,bind,Option.bind_some,backwardReference]

theorem forward_runtime_native (actual selected : Instructions) (plan : Plan) (d : Host.Dimensions)
    (runtime : ForwardRuntime) (m : WordMemory) (binding : ForwardBinding actual plan d runtime)
    (numeric : LocalConformance actual selected) :
    runtimeForward runtime m = some (forwardReference selected plan d m) ∧
    (∀ r o, ¬ForwardWritten d r o → forwardReference selected plan d m r o = m r o) := by
  rw [←instructions_eq actual selected numeric]
  exact ⟨forward_runtime _ _ _ _ _ binding,forward_frame _ _ _ _⟩

theorem backward_runtime_native (actual selected : Instructions) (plan : Plan) (d : Host.Dimensions)
    (runtime : BackwardRuntime) (later : WordMemory) (binding : BackwardBinding actual plan d runtime)
    (numeric : LocalConformance actual selected) :
    runtimeBackward runtime later = some (backwardReference selected plan d later) ∧
    (∀ r o, ¬BackwardWritten d r o → backwardReference selected plan d later r o = later r o) := by
  rw [←instructions_eq actual selected numeric]
  exact ⟨backward_runtime _ _ _ _ _ binding,backward_frame _ _ _ _⟩

#axiomsClean forward_execution
#axiomsClean backward_execution
#axiomsClean forward_frame
#axiomsClean backward_frame
#axiomsClean forward_inputs
#axiomsClean backward_inputs
#axiomsClean forward_only_output
#axiomsClean backward_preserves_output
#axiomsClean forward_native
#axiomsClean backward_native
#axiomsClean forward_runtime
#axiomsClean backward_runtime
#axiomsClean forward_runtime_native
#axiomsClean backward_runtime_native
#specNonCircular forwardReference avoiding [forwardExecute,runtimeForward,backwardExecute,runtimeBackward]
#specNonCircular backwardReference avoiding [forwardExecute,runtimeForward,backwardExecute,runtimeBackward]
#specNonCircular ForwardBinding avoiding [LocalConformance,runtimeForward,runtimeBackward,backwardReference]
#specNonCircular BackwardBinding avoiding [LocalConformance,runtimeForward,runtimeBackward,forwardReference]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LightningAttention.Native
