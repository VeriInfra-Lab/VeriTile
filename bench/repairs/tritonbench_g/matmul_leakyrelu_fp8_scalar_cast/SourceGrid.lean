import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8_scalar_cast.TypedReference

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.Execution
open VeriTile.Triton FP.Structural
open MatmulLeakyreluFp8 MatmulLeakyreluFp8.Checked
noncomputable section
open Classical
set_option maxHeartbeats 8000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

/-- Each program starts with independent registers and the current shared
memory. View offsets refer to actual Tensor first-element pointers. -/
def programState (b : Buffers) (memory : Memory α) (tile : Nat) (scratch : State α) : State α :=
  {scratch with
    pids := fun axisId => if axisId = 0 then tile else scratch.pids axisId
    mem := fun region offset =>
      if region = "A" then Cell.mk .real (memory (b.A.storage,b.A.offset+offset))
      else if region = "B" then Cell.mk .real (memory (b.B.storage,b.B.offset+offset))
      else if region = "C" then Cell.mk .fp16 (memory (b.C.storage,b.C.offset+offset))
      else scratch.mem region offset}

/-- The physical adapter reads the executed C view. Other view storage is
preserved, with the typed program's frame theorem proving that no other region
is written. No target/result expression is inserted into this projection. -/
def project {α : Type} [Inhabited α] (b : Buffers) (before : Memory α) (s : State α) : Memory α :=
  fun point => if point.1 = b.C.storage ∧ b.C.offset ≤ point.2 then
    (s.mem "C" (point.2-b.C.offset)).read .fp16 else before point

theorem program_inputs (b : Buffers) (memory : Memory α) (tile : Nat) (scratch : State α) :
    Inputs b (programState b memory tile scratch) memory := by
  constructor <;> intro o <;> simp [Inputs,programState]

theorem output_point {α : Type} (p : Problem) (b : Buffers) (s : State α)
    (i : CIndex) (ht : s.pids 0 < p.tiles) (hi : active p s i) :
    (b.C.storage,b.C.offset+cOffset p b s i) = Source.outputAddress p b (s.pids 0) (i.1,i.2.1) := by
  rw [Source.output_address p b _ _ ((active_eq p s i ht).mp hi),cOffset_eq p b s i ht hi]
  simp [Tensor.address,Nat.add_assoc]

theorem active_injective {α : Type} (p : Problem) (b : Buffers) (layout : Layout p b)
    (s : State α) (ht : s.pids 0 < p.tiles) (i j : CIndex)
    (hi : active p s i) (hj : active p s j) (he : cOffset p b s i = cOffset p b s j) : i = j := by
  have hh : Source.outputAddress p b (s.pids 0) (i.1,i.2.1) =
      Source.outputAddress p b (s.pids 0) (j.1,j.2.1) := by
    rw [←output_point p b s i ht hi,←output_point p b s j ht hj,he]
  have lanes := Source.lanes_unique p b layout _ _ _
    ((active_eq p s i ht).mp hi) ((active_eq p s j ht).mp hj) hh
  have hr : i.1 = j.1 := congrArg (fun x : Fin 128 × Fin 256 => x.1) lanes
  have hc : i.2.1 = j.2.1 := congrArg (fun x : Fin 128 × Fin 256 => x.2) lanes
  exact Prod.ext hr (Prod.ext hc (Subsingleton.elim _ _))

theorem program_projection {α : Type} [Inhabited α] (M : Algebra α) (ops : Operations α)
    (law : ScalarLaws M ops) (p : Problem) (b : Buffers) (layout : Layout p b)
    (memory : Memory α) (tile : Nat) (ht : tile < p.tiles) (scratch : State α) :
    ∃ final, FP.Structural.exec (backend M (selectedDot ops)) (source p b)
        (programState b memory tile scratch) = some final ∧
      project b memory final = Source.step (sourceArithmetic ops) p b tile memory ∧
      (∀ (region : RegionName) offset, region ≠ "C" → final.mem region offset =
        (programState b memory tile scratch).mem region offset) := by
  let start := programState b memory tile scratch
  have pid : start.pids 0 = tile := by simp [start,programState]
  have bound : start.pids 0 < p.tiles := by simpa [pid] using ht
  obtain ⟨final,he,_,hv,hf⟩ := programExecution M (selectedDot ops) p b start
    (active_injective p b layout start bound)
  refine ⟨final,he,?_,fun region offset hr => hf region offset (Or.inl hr)⟩
  funext point
  by_cases written : Source.writes p b tile point
  · obtain ⟨lane,ha,hpoint⟩ := written
    let i : CIndex := (lane.1,lane.2,PUnit.unit)
    have activeI : active p start i := (active_eq p start i bound).mpr (by simpa [pid,i] using ha)
    have pp : (b.C.storage,b.C.offset+cOffset p b start i) = point :=
      (output_point p b start i bound activeI).trans (by simpa [pid,i] using hpoint)
    have value := hv i activeI
    have val := output_source M ops law p b start memory (program_inputs b memory tile scratch) bound i
    have expected : Source.step (sourceArithmetic ops) p b tile memory point =
        ops.castHalf (Checked.activate ops p.enabled (sourceAccumulator ops p b memory tile p.kTiles i)) := by
      rw [Source.step,CrossEntropy1.Schedule.put,if_pos (show Source.writes p b tile point from ⟨lane,ha,hpoint⟩)]
      rw [←hpoint,Source.value_at_lane (sourceArithmetic ops) p b layout memory tile lane ha]
      rfl
    rw [expected,←pp]
    simp [project,value,val,pid]

  · rw [Source.step,CrossEntropy1.Schedule.put,if_neg written]
    by_cases covered : point.1 = b.C.storage ∧ b.C.offset ≤ point.2
    · have frame := hf "C" (point.2-b.C.offset) (Or.inr (by
        intro i hi same
        apply written
        have pp := output_point p b start i bound hi
        have pe : (b.C.storage,b.C.offset+cOffset p b start i) = point := by
          apply Prod.ext covered.1.symm
          omega
        refine ⟨(i.1,i.2.1),?_,?_⟩
        · simpa [pid] using (active_eq p start i bound).mp hi
        · simpa [pid] using pp.symm.trans pe))
      simp [project,covered,frame,start,programState,covered.1,Nat.add_sub_of_le covered.2,Cell.read_same]
      exact congrArg memory (Prod.ext covered.1.symm rfl)
    · simp [project,covered]

/-- Full source interpreter for the grid. Every listed PID executes the shared
readable typed quotation, then threads its projected current memory onward. -/
def runPrograms {α : Type} [Inhabited α] (M : Algebra α) (ops : Operations α)
    (p : Problem) (b : Buffers) (scratch : State α) : List Nat → Memory α → Option (Memory α)
  | [],memory => some memory
  | tile::rest,memory =>
    (FP.Structural.exec (backend M (selectedDot ops)) (source p b)
      (programState b memory tile scratch)).bind
        (fun final => runPrograms M ops p b scratch rest (project b memory final))

theorem programs_source {α : Type} [Inhabited α] (M : Algebra α) (ops : Operations α)
    (law : ScalarLaws M ops) (p : Problem) (b : Buffers) (layout : Layout p b)
    (scratch : State α) (tiles : List Nat) (legal : ∀ tile ∈ tiles, tile < p.tiles) (memory : Memory α) :
    runPrograms M ops p b scratch tiles memory =
      some (CrossEntropy1.Schedule.run (Source.step (sourceArithmetic ops) p b) tiles memory) := by
  induction tiles generalizing memory with
  | nil => rfl
  | cons tile rest ih =>
    obtain ⟨final,he,hp,_⟩ := program_projection M ops law p b layout memory tile
      (legal tile (by simp)) scratch
    simp only [runPrograms,he,Option.bind_some,hp]
    exact ih (fun t ht => legal t (by simp [ht])) _

def grid {α : Type} [Inhabited α] (M : Algebra α) (ops : Operations α)
    (p : Problem) (b : Buffers) (scratch : State α) (memory : Memory α) : Option (Memory α) :=
  runPrograms M ops p b scratch (Source.tiles p) memory

theorem grid_source {α : Type} [Inhabited α] (M : Algebra α) (ops : Operations α)
    (law : ScalarLaws M ops) (p : Problem) (b : Buffers) (layout : Layout p b)
    (scratch : State α) (memory : Memory α) :
    grid M ops p b scratch memory = some (Source.run (sourceArithmetic ops) p b memory) := by
  exact programs_source M ops law p b layout scratch (Source.tiles p)
    (fun tile ht => (Source.tiles_exact p tile).mp ht) memory

#specNonCircular project avoiding [reference,logicalReference,Source.run]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.Execution
