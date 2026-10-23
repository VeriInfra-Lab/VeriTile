import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.PhysicalFrame

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.PhysicalKernel
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 10000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def event (access : Access α Store Byte) (a : Arithmetic α) (materialize : ScalarMaterialization α)
    (native : Bool) (i : Contract.Inputs) (q : RmsMatmulRbe.Contract.Tensor) (initial : Store)
    (epsilon theta : HostScalar) (c : Commit) : Write α :=
  ⟨qkvOutputFP8 (useFP8 i) c.phase,outputAddress i q c.id c.phase c.lane,
    castOutput a (qkvOutputFP8 (useFP8 i) c.phase)
      (calleeTarget a native (selectedRotaryPlan c.phase) (useFP8 i) (qkvArgument (arguments i) c.phase)
        (state access a.zero i q initial c.id) (materialize.apply epsilon) (materialize.apply theta) c.lane)⟩

theorem valueEvents_commits (access : Access α Store Byte) (a : Arithmetic α) (materialize : ScalarMaterialization α)
    (native : Bool) (i : Contract.Inputs) (q : RmsMatmulRbe.Contract.Tensor) (initial : Store)
    (id : Program) (epsilon theta : HostScalar) :
    valueEvents access a materialize native i q initial id epsilon theta=
      (commits i id).map (event access a materialize native i q initial epsilon theta) := by
  simp only [valueEvents,commits,List.map_flatMap,List.map_map,Function.comp_def,event]

theorem valueEvents_output (access : Access α Store Byte) (a : Arithmetic α) (materialize : ScalarMaterialization α)
    (native : Bool) (i : Contract.Inputs) (q : RmsMatmulRbe.Contract.Tensor) (initial : Store)
    (id : Program) (valid : id.Valid i) (epsilon theta : HostScalar) :
    ∀ e∈valueEvents access a materialize native i q initial id epsilon theta,OutputWrite i q e := by
  rw [valueEvents_commits]
  intro e member
  obtain ⟨c,hc,rfl⟩ := List.mem_map.mp member
  obtain ⟨same,live⟩ := (commits_mem i id c).mp hc
  exact ⟨c,⟨same.symm ▸ valid,same.symm ▸ live⟩,rfl,rfl⟩

theorem event_stable (access : Access α Store Byte) (a : Arithmetic α) (materialize : ScalarMaterialization α)
    (native : Bool) (i : Contract.Inputs) (checked : Contract.InputContract i) (ordinary : OrdinaryInputs i)
    (q : RmsMatmulRbe.Contract.Tensor) (hq : ExistingQ i q) (initial : Store) (epsilon theta : HostScalar)
    (events : List (Write α)) (writes : ∀ e∈events,OutputWrite i q e) (c : Commit) :
    event access a materialize native i q (run access events initial) epsilon theta c=
      event access a materialize native i q initial epsilon theta c := by
  have stable := calleeTarget_stable a native (selectedRotaryPlan c.phase) (useFP8 i)
    (qkvArgument (arguments i) c.phase) (state access a.zero i q (run access events initial) c.id)
    (state access a.zero i q initial c.id)
    (state_input_frame access a.zero i checked ordinary q hq events writes initial c.id c.phase)
    rfl (materialize.apply epsilon) (materialize.apply theta)
  simp only [event,stable]

theorem valueEvents_stable (access : Access α Store Byte) (a : Arithmetic α) (materialize : ScalarMaterialization α)
    (native : Bool) (i : Contract.Inputs) (checked : Contract.InputContract i) (ordinary : OrdinaryInputs i)
    (q : RmsMatmulRbe.Contract.Tensor) (hq : ExistingQ i q) (initial : Store) (epsilon theta : HostScalar)
    (events : List (Write α)) (writes : ∀ e∈events,OutputWrite i q e) (id : Program) :
    valueEvents access a materialize native i q (run access events initial) id epsilon theta=
      valueEvents access a materialize native i q initial id epsilon theta := by
  rw [valueEvents_commits,valueEvents_commits]
  exact List.map_congr_left (fun c _ => event_stable access a materialize native i checked ordinary q hq
    initial epsilon theta events writes c)

def eventsFor (access : Access α Store Byte) (a : Arithmetic α) (materialize : ScalarMaterialization α)
    (native : Bool) (i : Contract.Inputs) (q : RmsMatmulRbe.Contract.Tensor) (initial : Store)
    (epsilon theta : HostScalar) (ids : List Program) : List (Write α) :=
  ids.flatMap (fun id => valueEvents access a materialize native i q initial id epsilon theta)

def allEvents (access : Access α Store Byte) (a : Arithmetic α) (materialize : ScalarMaterialization α)
    (native : Bool) (i : Contract.Inputs) (q : RmsMatmulRbe.Contract.Tensor) (initial : Store)
    (epsilon theta : HostScalar) : List (Write α) :=
  (allCommits i).map (event access a materialize native i q initial epsilon theta)

theorem allEvents_programs (access : Access α Store Byte) (a : Arithmetic α) (materialize : ScalarMaterialization α)
    (native : Bool) (i : Contract.Inputs) (q : RmsMatmulRbe.Contract.Tensor) (initial : Store)
    (epsilon theta : HostScalar) :
    allEvents access a materialize native i q initial epsilon theta=
      eventsFor access a materialize native i q initial epsilon theta (programs i) := by
  simp only [allEvents,allCommits,List.map_flatMap,eventsFor,valueEvents_commits]

def gridAux (access : Access α Store Byte) (a : Arithmetic α) (materialize : ScalarMaterialization α)
    (native : Bool) (i : Contract.Inputs) (q : RmsMatmulRbe.Contract.Tensor) (epsilon theta : HostScalar) :
    List Program → Store → Option Store
  | [],s => some s
  | id::ids,s => (program access a materialize native i q s id epsilon theta).bind
      (gridAux access a materialize native i q epsilon theta ids)

def grid (access : Access α Store Byte) (a : Arithmetic α) (materialize : ScalarMaterialization α)
    (native : Bool) (i : Contract.Inputs) (q : RmsMatmulRbe.Contract.Tensor) (initial : Store)
    (epsilon theta : HostScalar) : Option Store :=
  gridAux access a materialize native i q epsilon theta (programs i) initial

theorem gridAux_complete (access : Access α Store Byte) (a : Arithmetic α) (z : ZeroLaws a)
    (materialize : ScalarMaterialization α) (native : Bool) (i : Contract.Inputs)
    (checked : Contract.InputContract i) (ordinary : OrdinaryInputs i)
    (q : RmsMatmulRbe.Contract.Tensor) (hq : ExistingQ i q) (initial : Store) (epsilon theta : HostScalar)
    (ids : List Program) (valid : ∀ id∈ids,id.Valid i) (before : List (Write α))
    (writes : ∀ e∈before,OutputWrite i q e) :
    gridAux access a materialize native i q epsilon theta ids (run access before initial)=
      some (run access (before++eventsFor access a materialize native i q initial epsilon theta ids) initial) := by
  induction ids generalizing before with
  | nil => simp [gridAux,eventsFor]
  | cons id ids ih =>
    rw [gridAux,program_complete access a z materialize native i checked]
    rw [valueEvents_stable access a materialize native i checked ordinary q hq initial epsilon theta before writes]
    simp only [Option.bind_some]
    rw [←run_append]
    rw [ih (fun id h => valid id (List.mem_cons_of_mem _ h))]
    · simp only [eventsFor,List.flatMap_cons,List.append_assoc]
    · intro e he
      rcases List.mem_append.mp he with old|current
      · exact writes e old
      · exact valueEvents_output access a materialize native i q initial id (valid id (by simp)) epsilon theta e current

theorem grid_complete (access : Access α Store Byte) (a : Arithmetic α) (z : ZeroLaws a)
    (materialize : ScalarMaterialization α) (native : Bool) (i : Contract.Inputs)
    (checked : Contract.InputContract i) (ordinary : OrdinaryInputs i)
    (q : RmsMatmulRbe.Contract.Tensor) (hq : ExistingQ i q) (initial : Store) (epsilon theta : HostScalar) :
    grid access a materialize native i q initial epsilon theta=
      some (run access (allEvents access a materialize native i q initial epsilon theta) initial) := by
  simpa only [grid,allEvents_programs,List.nil_append,run,List.foldl_nil] using
    gridAux_complete access a z materialize native i checked ordinary q hq initial epsilon theta
      (programs i) (fun id h => (programs_mem i id).mp h) [] (by simp)

#axiomsClean event_stable
#axiomsClean gridAux_complete
#axiomsClean grid_complete
#specNonCircular allEvents avoiding [grid,gridAux,program,programTrace,storedEvents]
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.PhysicalKernel
