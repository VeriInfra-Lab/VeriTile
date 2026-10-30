import bench.tritonbench_g.batched_vecmat_mult.Proofs.HostOutputBounds

/-! The source's two load regions and exact natural-element offsets. Program
order may change only the output region. Logical output/input separation is
an explicit profile condition; it is not a native allocator/byte-memory fact.
The fixed FP configuration and reduction plans are retained without a law
equating different trees, reassociated arithmetic or real-valued sums. -/
namespace VeriTile.Bench.TritonBenchG.BatchedVecmatMult.ValueProofs
open VeriTile.Triton FP.Structural
open FP.Equational (ReductionPlan)
noncomputable section
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false
local instance scheduleInputsWord : Inhabited Word := ⟨FP.Value.zero .fp32⟩

def InputsSeparate (p : Parameters) : Prop := p.A ≠ p.output ∧ p.B ≠ p.output

/-- Addresses loaded by the two source loads over all actual loop iterations.
The A load has shape BM×BK, so it does not require a BN lane. -/
def ReadFootprint (p : Parameters) (s : State Word) (r : RegionName) (o : Nat) : Prop :=
  (r = p.A ∧ ∃ i : Fin p.BM, ∃ c : Fin (p.K / p.BK), ∃ k : Fin p.BK,
    o = aOffset p s i.val c.val k.val) ∨
  (r = p.B ∧ ∃ i : Fin p.BM, ∃ j : Fin p.BN, ∃ c : Fin (p.K / p.BK),
    ∃ k : Fin p.BK, o = bOffset p s i.val j.val c.val k.val)

/-- The actual completed source body retains both loaded register arrays.
The equalities expose their initial-memory cells and exact logical offsets. -/
theorem actual_body_input_loads (rest : Algebra Word) (cfg : FP.Config)
    (plans : ∀ n, ReductionPlan n) (p : Parameters) (initial current : State Word)
    (c : Nat) (h : Invariant cfg plans p initial c current) :
    (bodyExecution rest cfg plans p initial current c h).val.regs .real [p.BM,p.BK] "a" =
      some (fun i => (initial.mem p.A (aOffset p initial i.1.val c i.2.1.val)).read .real) ∧
    (bodyExecution rest cfg plans p initial current c h).val.regs .real [p.BN,p.BM,p.BK] "b" =
      some (fun i => (initial.mem p.B
        (bOffset p initial i.2.1.val i.1.val c i.2.2.1.val)).read .real) := by
  constructor
  all_goals simp [bodyExecution, h.mi, h.ni, h.offsets, h.carry, h.blocks, h.memory,
    aOffset, bOffset, row, col, bop, Broadcast.leftIndex, Broadcast.rightIndex,
    TileShape.dropInsertedIndex, Region.cast]
  all_goals funext i
  all_goals rfl

theorem read_footprint_region (p : Parameters) (s : State Word) (r : RegionName) (o : Nat)
    (read : ReadFootprint p s r o) : r = p.A ∨ r = p.B := by
  rcases read with read | read
  · exact Or.inl read.1
  · exact Or.inr read.1

theorem input_reads_outside_output (p : Parameters) (s : State Word)
    (separate : InputsSeparate p) (r : RegionName) (o : Nat)
    (read : ReadFootprint p s r o) : r ≠ p.output := by
  rcases read_footprint_region p s r o read with ha | hb
  · exact ha ▸ separate.1
  · exact hb ▸ separate.2

theorem expected_memory_preserves_inputs (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (p : Parameters) (s : State Word) (separate : InputsSeparate p) :
    expectedMemory cfg plans p s p.A = s.mem p.A ∧
      expectedMemory cfg plans p s p.B = s.mem p.B := by
  constructor
  · funext o
    exact expectedMemory_frame cfg plans p s p.A o (Or.inl separate.1)
  · funext o
    exact expectedMemory_frame cfg plans p s p.B o (Or.inl separate.2)

theorem expected_memory_preserves_read_footprint (cfg : FP.Config)
    (plans : ∀ n, ReductionPlan n) (p : Parameters) (s : State Word)
    (separate : InputsSeparate p) (r : RegionName) (o : Nat)
    (read : ReadFootprint p s r o) :
    expectedMemory cfg plans p s r o = s.mem r o := by
  exact expectedMemory_frame cfg plans p s r o
    (Or.inl (input_reads_outside_output p s separate r o read))

/-- A real complete source execution exists and preserves both input regions.
Input A and B may alias each other; only output separation is required. -/
theorem complete_preserves_inputs (rest : Algebra Word) (cfg : FP.Config)
    (plans : ∀ n, ReductionPlan n) (p : Parameters) (s : State Word)
    (separate : InputsSeparate p) :
    ∃ final, FP.Structural.exec (engine rest cfg plans) (source p) s = some final ∧
      final.mem = expectedMemory cfg plans p s ∧
      final.mem p.A = s.mem p.A ∧ final.mem p.B = s.mem p.B := by
  obtain ⟨final, he, hm, _⟩ := complete_correct rest cfg plans p s
  refine ⟨final, he, hm, ?_⟩
  rw [hm]
  exact expected_memory_preserves_inputs cfg plans p s separate

theorem block_sum_read_footprint_congr (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (p : Parameters) (s t : State Word) (pids : s.pids = t.pids)
    (reads : ∀ r o, ReadFootprint p s r o → s.mem r o = t.mem r o)
    (c : Nat) (hc : c < p.K / p.BK) (i : TileIndex [p.BM,p.BN]) :
    blockSum cfg plans p s c i = blockSum cfg plans p t c i := by
  unfold blockSum
  congr 1
  funext k
  have ha := reads p.A (aOffset p s i.1.val c k.val)
    (Or.inl ⟨rfl, i.1, ⟨c,hc⟩, k, rfl⟩)
  have hb := reads p.B (bOffset p s i.1.val i.2.1.val c k.val)
    (Or.inr ⟨rfl, i.1, i.2.1, ⟨c,hc⟩, k, rfl⟩)
  have oa : aOffset p s i.1.val c k.val = aOffset p t i.1.val c k.val := by
    simp only [aOffset, row, pids]
  have ob : bOffset p s i.1.val i.2.1.val c k.val =
      bOffset p t i.1.val i.2.1.val c k.val := by
    simp only [bOffset, row, col, pids]
  rw [ha, hb, oa, ob]

theorem accumulated_read_footprint_congr (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (p : Parameters) (s t : State Word) (pids : s.pids = t.pids)
    (reads : ∀ r o, ReadFootprint p s r o → s.mem r o = t.mem r o)
    (c : Nat) (hc : c ≤ p.K / p.BK) (i : TileIndex [p.BM,p.BN]) :
    accumulated cfg plans p s c i = accumulated cfg plans p t c i := by
  induction c with
  | zero => rfl
  | succ c ih =>
    rw [accumulated, accumulated, ih (by omega),
      block_sum_read_footprint_congr cfg plans p s t pids reads c (by omega) i]

theorem accumulated_input_regions_congr (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (p : Parameters) (s t : State Word) (pids : s.pids = t.pids)
    (ha : s.mem p.A = t.mem p.A) (hb : s.mem p.B = t.mem p.B)
    (c : Nat) (hc : c ≤ p.K / p.BK) (i : TileIndex [p.BM,p.BN]) :
    accumulated cfg plans p s c i = accumulated cfg plans p t c i := by
  apply accumulated_read_footprint_congr cfg plans p s t pids _ c hc i
  intro r o read
  rcases read_footprint_region p s r o read with h | h
  · subst r
    exact congrFun ha o
  · subst r
    exact congrFun hb o

/-- The write set records actual unmasked source stores, not merely a bounding
interval. Empty tile dimensions produce no writes. -/
def ProgramWrites (p : Parameters) (initial : State Word) (pid : Nat)
    (r : RegionName) (o : Nat) : Prop :=
  r = p.output ∧ ∃ i : TileIndex [p.BM,p.BN],
    o = outputOffset p (programState p initial initial.mem pid) i

theorem distinct_program_write_sets (p : Parameters) (initial : State Word)
    (k l : Nat) (hk : k < gridCount p) (hl : l < gridCount p) (different : k ≠ l)
    (r : RegionName) (o : Nat) :
    ¬ (ProgramWrites p initial k r o ∧ ProgramWrites p initial l r o) := by
  rintro ⟨⟨_, i, hi⟩, ⟨_, j, hj⟩⟩
  exact different (actual_program_output_unique p initial initial.mem initial.mem
    k l hk hl i j (hi.symm.trans hj)).1

theorem program_writes_outside_input_reads (p : Parameters) (initial s : State Word)
    (separate : InputsSeparate p) (pid : Nat) (r : RegionName) (o : Nat) :
    ¬ (ProgramWrites p initial pid r o ∧ ReadFootprint p s r o) := by
  rintro ⟨write, read⟩
  exact input_reads_outside_output p s separate r o read write.1

#axiomsClean read_footprint_region
#axiomsClean actual_body_input_loads
#axiomsClean input_reads_outside_output
#axiomsClean expected_memory_preserves_inputs
#axiomsClean expected_memory_preserves_read_footprint
#axiomsClean complete_preserves_inputs
#axiomsClean block_sum_read_footprint_congr
#axiomsClean accumulated_read_footprint_congr
#axiomsClean accumulated_input_regions_congr
#axiomsClean distinct_program_write_sets
#axiomsClean program_writes_outside_input_reads
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BatchedVecmatMult.ValueProofs
