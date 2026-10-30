import bench.tritonbench_g.lightning_attention.NativeComposition
namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly.Native
open VeriTile.Bench.Repairs.LightningAttention.Native
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

#axiomsClean forward_native
#axiomsClean backward_native
end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly.Native
