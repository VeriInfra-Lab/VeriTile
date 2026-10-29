import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.GridExecution

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.PhysicalKernel
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 9000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem run_preserve_readback (access : Access α Store Byte) (events : List (Write α))
    (s : Store) (target : Write α) (valid : access.storable target.fp8 target.value)
    (read : access.read target.fp8 s target.address=target.value)
    (compatible : ∀ e∈events,e=target ∨
      Apart e.address (outputWidth e.fp8) target.address (outputWidth target.fp8)) :
    access.read target.fp8 (run access events s) target.address=target.value := by
  induction events generalizing s with
  | nil => exact read
  | cons e rest ih =>
    change access.read target.fp8 (run access rest (access.write e.fp8 s e.address e.value)) target.address=target.value
    apply ih
    · rcases compatible e (by simp) with same|apart
      · subst e
        exact read_write_same access _ _ _ _ valid
      · exact (read_write_apart access _ _ _ _ _ _ apart).trans read
    · exact fun e he => compatible e (List.mem_cons_of_mem _ he)

theorem run_readback_mem (access : Access α Store Byte) (events : List (Write α))
    (s : Store) (target : Write α) (member : target∈events)
    (valid : access.storable target.fp8 target.value)
    (compatible : ∀ e∈events,e=target ∨
      Apart e.address (outputWidth e.fp8) target.address (outputWidth target.fp8)) :
    access.read target.fp8 (run access events s) target.address=target.value := by
  induction events generalizing s with
  | nil => simp at member
  | cons e rest ih =>
    change access.read target.fp8 (run access rest (access.write e.fp8 s e.address e.value)) target.address=target.value
    rcases List.mem_cons.mp member with same|later
    · subst e
      exact run_preserve_readback access rest _ target valid (read_write_same access _ _ _ _ valid)
        (fun e he => compatible e (List.mem_cons_of_mem _ he))
    · exact ih _ later (fun e he => compatible e (List.mem_cons_of_mem _ he))

theorem allEvents_readback (access : Access α Store Byte) (a : Arithmetic α)
    (materialize : ScalarMaterialization α) (native : Bool) (i : Contract.Inputs)
    (checked : Contract.InputContract i) (ordinary : OrdinaryInputs i)
    (q : RmsMatmulRbe.Contract.Tensor) (hq : ExistingQ i q) (initial : Store)
    (epsilon theta : HostScalar) (range : OutputStorable access a) (c : Commit) (hc : c.Valid i) :
    access.read (qkvOutputFP8 (useFP8 i) c.phase)
      (run access (allEvents access a materialize native i q initial epsilon theta) initial)
      (outputAddress i q c.id c.phase c.lane)=
        (event access a materialize native i q initial epsilon theta c).value := by
  apply run_readback_mem access _ initial (event access a materialize native i q initial epsilon theta c)
  · exact List.mem_map.mpr ⟨c,(allCommits_mem i c).mpr hc,rfl⟩
  · exact range _ _
  · intro e he
    obtain ⟨other,hm,rfl⟩ := List.mem_map.mp he
    by_cases same : other=c
    · exact Or.inl (same ▸ rfl)
    · exact Or.inr (commit_apart i checked ordinary q hq other c ((allCommits_mem i other).mp hm) hc same)

/-- Exact union of live store bytes, including each output's actual width
and storage offset. Its complement includes all untouched backing bytes. -/
def WrittenByte (i : Contract.Inputs) (q : RmsMatmulRbe.Contract.Tensor) (j : Nat) : Prop :=
  ∃ c : Commit,c.Valid i ∧ outputAddress i q c.id c.phase c.lane≤j ∧
    j<outputAddress i q c.id c.phase c.lane+outputWidth (qkvOutputFP8 (useFP8 i) c.phase)

theorem allEvents_byte_frame (access : Access α Store Byte) (a : Arithmetic α)
    (materialize : ScalarMaterialization α) (native : Bool) (i : Contract.Inputs)
    (q : RmsMatmulRbe.Contract.Tensor) (initial : Store) (epsilon theta : HostScalar) (j : Nat)
    (outside : ¬WrittenByte i q j) :
    access.byte (run access (allEvents access a materialize native i q initial epsilon theta) initial) j=
      access.byte initial j := by
  apply run_byte_frame
  intro e he
  obtain ⟨c,hc,rfl⟩ := List.mem_map.mp he
  have valid := (allCommits_mem i c).mp hc
  have absent : ¬(outputAddress i q c.id c.phase c.lane≤j ∧
      j<outputAddress i q c.id c.phase c.lane+outputWidth (qkvOutputFP8 (useFP8 i) c.phase)) :=
    fun h => outside ⟨c,valid,h⟩
  change j<outputAddress i q c.id c.phase c.lane ∨
    outputAddress i q c.id c.phase c.lane+outputWidth (qkvOutputFP8 (useFP8 i) c.phase)≤j
  omega

theorem input_backing_unwritten (i : Contract.Inputs) (checked : Contract.InputContract i)
    (ordinary : OrdinaryInputs i) (q : RmsMatmulRbe.Contract.Tensor) (hq : ExistingQ i q)
    (t : RmsMatmulRbe.Contract.Tensor) (ht : t∈i.original.reads) (j : Nat)
    (inside : t.base≤j ∧ j<t.base+t.bytes) : ¬WrittenByte i q j := by
  rintro ⟨c,hc,lo,hi⟩
  have window := backing_window _ _ (output_inBacking i checked ordinary q hq c hc)
  have separate := caller_output_read_apart i checked q hq c.phase t ht
  rw [caller_output_width i checked q hq] at hi
  change address (RmsMatmulRbe.Checked.output i.original q c.phase) (logicalOffset i c.id c.lane)≤j at lo
  change j<address (RmsMatmulRbe.Checked.output i.original q c.phase) (logicalOffset i c.id c.lane)+_ at hi
  unfold RmsMatmulRbe.Contract.disjoint RmsMatmulRbe.Contract.overlap at separate
  omega

/-- Success, every live output readback and the complete byte frame are
conclusions of whole-grid execution over the same existing caller storage. -/
theorem grid_results (access : Access α Store Byte) (a : Arithmetic α) (z : ZeroLaws a)
    (materialize : ScalarMaterialization α) (native : Bool) (i : Contract.Inputs)
    (checked : Contract.InputContract i) (ordinary : OrdinaryInputs i)
    (q : RmsMatmulRbe.Contract.Tensor) (hq : ExistingQ i q) (initial : Store)
    (epsilon theta : HostScalar) (range : OutputStorable access a) :
    ∃ final,grid access a materialize native i q initial epsilon theta=some final ∧
      (∀ c : Commit,c.Valid i → access.read (qkvOutputFP8 (useFP8 i) c.phase) final
        (outputAddress i q c.id c.phase c.lane)=(event access a materialize native i q initial epsilon theta c).value) ∧
      (∀ j,¬WrittenByte i q j → access.byte final j=access.byte initial j) := by
  refine ⟨_,grid_complete access a z materialize native i checked ordinary q hq initial epsilon theta,?_,?_⟩
  · exact fun c hc => allEvents_readback access a materialize native i checked ordinary q hq initial epsilon theta range c hc
  · exact fun j outside => allEvents_byte_frame access a materialize native i q initial epsilon theta j outside

#axiomsClean allEvents_readback
#axiomsClean allEvents_byte_frame
#axiomsClean input_backing_unwritten
#axiomsClean grid_results
#specNonCircular WrittenByte avoiding [grid,gridAux,program,allEvents,event]
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.PhysicalKernel
