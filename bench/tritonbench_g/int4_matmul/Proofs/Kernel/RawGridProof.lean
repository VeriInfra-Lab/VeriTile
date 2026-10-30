import bench.tritonbench_g.int4_matmul.Proofs.Kernel.GridExecution
import bench.tritonbench_g.int4_matmul.Proofs.Kernel.RawOperandBridge

namespace VeriTile.Bench.Repairs.Int4Matmul.KernelExecution.Raw
open VeriTile.Triton FP.Structural KernelSource
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000

theorem event_memory [Inhabited α] (x : Arguments) (hx : Geometry x) (t : Tile)
    (ht : t∈Host.sourceTiles) (o : Ops α) (rest : Algebra α)
    (initial current : ScalarState α) (scratch s : RegisterState α)
    (hm : s.mem=memory current scratch.mem) (l : Lane) (hl : l.Valid x.shape t) :
    (event o rest (parameters x t) (attach initial scratch) l s).mem=
      memory (if x.shape.Active t (laneWork x.shape t l) then
        Source.commit x.shape t o (laneWork x.shape t l)
          (partialValue x.shape t o initial (laneWork x.shape t l)) current else current) scratch.mem := by
  let program := atProgram (attach initial scratch) l.pid l.split
  let i : CIndex (parameters x t) := (⟨l.i,hl.2.2.1⟩,⟨l.j,hl.2.2.2⟩,PUnit.unit)
  have hw : work x t program i=laneWork x.shape t l := by
    cases l
    rfl
  have hv := source_partial x hx t ht o program initial (attached_inputs initial scratch l.pid l.split) i
  have ha : active (parameters x t) program i ↔ x.shape.Active t (laneWork x.shape t l) := by
    change (_ ∧ _) ↔ (_ ∧ _ ∧ _)
    simp only [←hw,←row_work,←column_work,split_work,program,atProgram]
    simp [hl.2.1]
    rfl
  have ho : outputOffset (parameters x t) program i=cAddress x.shape (laneWork x.shape t l).row (laneWork x.shape t l).col := by
    rw [←hw,←row_work,←column_work]
    simp [outputOffset,parameters,cAddress,Nat.mul_comm]
  simp only [event,parameters,hl.2.2.1,hl.2.2.2,and_self,↓reduceDIte]
  change (writeLane o (parameters x t) program (computedValues o rest (parameters x t) program) s i).mem=_
  rw [computed_values]
  unfold writeLane
  simp only [ha]
  by_cases live : x.shape.Active t (laneWork x.shape t l)
  · simp only [live,ite_true]
    rw [ho,hv,hw]
    have read : (s.mem "out" (cAddress x.shape (laneWork x.shape t l).row (laneWork x.shape t l).col)).read .real=
        current.values .output (cAddress x.shape (laneWork x.shape t l).row (laneWork x.shape t l).col) := by
      simp [hm,memory]
    simp only [parameters,read,Source.commit]
    exact memory_store current scratch.mem _ _ s hm
  · simpa only [live,ite_false] using hm

theorem events_memory [Inhabited α] (x : Arguments) (hx : Geometry x) (t : Tile)
    (ht : t∈Host.sourceTiles) (o : Ops α) (rest : Algebra α)
    (initial current : ScalarState α) (scratch s : RegisterState α)
    (hm : s.mem=memory current scratch.mem) (lanes : List Lane)
    (valid : ∀ l∈lanes,l.Valid x.shape t) :
    (lanes.foldl (fun st l => event o rest (parameters x t) (attach initial scratch) l st) s).mem=
      memory (Source.prepared x.shape t o initial (laneSchedule x.shape t lanes) current) scratch.mem := by
  induction lanes generalizing current s with
  | nil => exact hm
  | cons l lanes ih =>
    rw [List.foldl_cons]
    have he := event_memory x hx t ht o rest initial current scratch s hm l (valid l (by simp))
    have next := ih _ _ he (fun q hq => valid q (by simp [hq]))
    by_cases ha : x.shape.Active t (laneWork x.shape t l)
    · simpa [laneSchedule,ha,Source.prepared] using next
    · simpa [laneSchedule,ha,Source.prepared] using next

theorem schedule_active (s : Shape) (t : Tile) (lanes : List Lane) (w : Work)
    (h : w∈laneSchedule s t lanes) : s.Active t w := by
  obtain ⟨l,hl,rfl⟩ := List.mem_map.mp h
  exact of_decide_eq_true (List.mem_filter.mp hl).2

/-- Parallel quoted-source execution equals the established scalar commit
model, after deriving every commit value from its actual source execution. -/
theorem parallel_memory [Inhabited α] (x : Arguments) (hx : Geometry x) (t : Tile)
    (ht : t∈Host.sourceTiles) (o : Ops α) (rest : Algebra α)
    (initial : ScalarState α) (scratch : RegisterState α) (lanes : List Lane)
    (valid : ∀ l∈lanes,l.Valid x.shape t) :
    (parallel o rest (parameters x t) (attach initial scratch) lanes).mem=
      memory (execute x.shape t o (laneSchedule x.shape t lanes) initial) scratch.mem := by
  have h := events_memory x hx t ht o rest initial initial scratch (attach initial scratch) rfl lanes valid
  rw [Source.prepared_execute x.shape t o initial initial _
    (schedule_active x.shape t lanes) ⟨fun _ _ _ => rfl,rfl⟩] at h
  exact h

/-- Kernel launch state after the source wrapper's required output reset.
Zero initialization is a pre-kernel memory boundary, not a claim that the
quoted kernel performs the Torch reset or allocation itself. -/
def launch [Inhabited α] (x : Arguments) (t : Tile) (o : Ops α) (rest : Algebra α)
    (initial : ScalarState α) (scratch : RegisterState α) (lanes : List Lane) :=
  parallel o rest (parameters x t) (attach (zeroOutput x.shape o initial) scratch) lanes

theorem launch_memory [Inhabited α] (x : Arguments) (hx : Geometry x) (t : Tile)
    (ht : t∈Host.sourceTiles) (o : Ops α) (rest : Algebra α)
    (initial : ScalarState α) (scratch : RegisterState α) (lanes : List Lane)
    (complete : lanes.Perm (rawLanes x.shape t)) :
    (launch x t o rest initial scratch lanes).mem=
      memory (call x.shape t o (laneSchedule x.shape t lanes) initial) scratch.mem := by
  exact parallel_memory x hx t ht o rest (zeroOutput x.shape o initial) scratch lanes
    (fun l hl => (raw_mem x.shape t l).mp (complete.mem_iff.mp hl))

theorem launch_value [Inhabited α] (x : Arguments) (hx : Geometry x) (t : Tile)
    (ht : t∈Host.sourceTiles) (o : Ops α) (rest : Algebra α)
    (initial : ScalarState α) (scratch : RegisterState α) (lanes : List Lane)
    (complete : lanes.Perm (rawLanes x.shape t)) (row col : Nat)
    (hr : row<x.shape.m) (hc : col<x.shape.n) :
    (launch x t o rest initial scratch lanes).mem "out" (cAddress x.shape row col)=
      .mk .real (referenceOutput x.shape t o initial row col (laneSchedule x.shape t lanes) o.zero) := by
  rw [launch_memory x hx t ht o rest initial scratch lanes complete]
  simp only [memory,show ("out" : RegionName)≠"a" by decide,show ("out" : RegionName)≠"weight" by decide,
    show ("out" : RegionName)≠"scale" by decide,show ("out" : RegionName)≠"zero" by decide,↓reduceIte]
  rw [call_result x.shape t hx.shape (Host.all_tiles_valid t ht) o _
    (schedule_complete x.shape t lanes complete) initial row col hr hc]

theorem launch_real (x : Arguments) (hx : Geometry x) (t : Tile)
    (ht : t∈Host.sourceTiles) (rest : Algebra ℝ)
    (initial : ScalarState ℝ) (scratch : RegisterState ℝ) (lanes : List Lane)
    (complete : lanes.Perm (rawLanes x.shape t)) (row col : Nat)
    (hr : row<x.shape.m) (hc : col<x.shape.n) :
    (launch x t realOps rest initial scratch lanes).mem "out" (cAddress x.shape row col)=
      .mk .real (mathematical x.shape initial row col) := by
  rw [launch_memory x hx t ht realOps rest initial scratch lanes complete]
  simp only [memory,show ("out" : RegionName)≠"a" by decide,show ("out" : RegionName)≠"weight" by decide,
    show ("out" : RegionName)≠"scale" by decide,show ("out" : RegionName)≠"zero" by decide,↓reduceIte]
  rw [whole_real x.shape t hx.shape (Host.all_tiles_valid t ht) initial _
    (schedule_complete x.shape t lanes complete) row col hr hc]

theorem launch_frame [Inhabited α] (x : Arguments) (hx : Geometry x) (t : Tile)
    (ht : t∈Host.sourceTiles) (o : Ops α) (rest : Algebra α)
    (initial : ScalarState α) (scratch : RegisterState α) (lanes : List Lane)
    (complete : lanes.Perm (rawLanes x.shape t)) (r : RegionName) (i : Nat)
    (outside : r ≠ ("out" : RegionName) ∨ x.shape.m * x.shape.n ≤ i) :
    (launch x t o rest initial scratch lanes).mem r i=(attach initial scratch).mem r i := by
  rw [launch_memory x hx t ht o rest initial scratch lanes complete]
  have readOnly := execute_readonly x.shape t o (laneSchedule x.shape t lanes)
    initial (zeroOutput x.shape o initial) (zero_readonly x.shape o initial)
  have ha := readOnly.1 .a i (by decide)
  have hs := readOnly.1 .scale i (by decide)
  change (call x.shape t o (laneSchedule x.shape t lanes) initial).values .a i=initial.values .a i at ha
  change (call x.shape t o (laneSchedule x.shape t lanes) initial).values .scale i=initial.values .scale i at hs
  have hp : (call x.shape t o (laneSchedule x.shape t lanes) initial).packed=initial.packed := readOnly.2
  simp only [attach,memory,hp,ha,hs]
  split_ifs with h1 h2 h3 h4 hout
  · rfl
  · rfl
  · rfl
  · rfl
  · rw [call_frame x.shape t hx.shape (Host.all_tiles_valid t ht) o _
      (schedule_complete x.shape t lanes complete) initial .output i
      (Or.inr (outside.resolve_left (by simpa [hout])))]
  · rfl


#axiomsClean parallel_memory
#axiomsClean launch_real
#axiomsClean launch_value
#axiomsClean launch_frame
end
end VeriTile.Bench.Repairs.Int4Matmul.KernelExecution.Raw
