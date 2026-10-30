import bench.tritonbench_g.rope_transform.IO
import bench.tritonbench_g.rope_transform.Proofs.AccessSafety

namespace VeriTile.Bench.Repairs.RopeTransform.PublicIO
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
set_option linter.unusedSectionVars false

variable {α β : Type} [Inhabited α] [Inhabited β]

theorem invocation_run (M : Algebra β) (r : Request) (row : Nat) (s : State β) :
    (invocation M r row).run s=some (Grid.step M r row s) :=
  (T.execution M r.parameters (Grid.setup r row s)).property

theorem schedule_run (M : Algebra β) (r : Request) (s : State β) (n : Nat) :
    runSchedule ((List.range n).map (invocation M r)) s=some (Grid.run M r s n) := by
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [List.range_succ,List.map_append,runSchedule_append,ih,Option.bind_some]
    simpa only [List.map_singleton,runSchedule_singleton,Grid.run] using
      invocation_run M r n (Grid.run M r s n)

private theorem safe_list (bounds : RegionName → Nat) (jobs : List (Invocation β))
    (each : ∀ job∈jobs,∀ s,job.Safe bounds s) (s : State β) : ScheduleSafe bounds jobs s := by
  induction jobs generalizing s with
  | nil => trivial
  | cons job rest ih =>
    exact ⟨each job (by simp) s,fun t _ => ih (fun next h => each next (by simp [h])) t⟩

theorem sequence_positive (r : Request) (row : Nat) (hr : row<r.rows) : 0<r.sequence := by
  by_contra h
  have zero : r.sequence=0 := by omega
  simp [Request.rows,zero] at hr

theorem io_safe (M : Algebra β) (encode : α → β) (observe : β → α)
    (r : Request) (xs : Inputs α r) (bounds : RegionName → Nat)
    (bounded : (sourceIO M encode observe r).InBounds bounds xs) (s : State β) :
    ScheduleSafe bounds ((List.range r.rows).map (invocation M r)) s := by
  apply safe_list
  intro job member initial
  obtain ⟨row,hr,rfl⟩ := List.mem_map.mp member
  have rowBound := List.mem_range.mp hr
  have seq := sequence_positive r row rowBound
  apply SourceSafety.source_safe M r (Grid.setup r row initial) bounds
  · intro sine j hj
    have tableRow := Nat.mod_lt row seq
    cases sine
    · exact bounded.2.1 ⟨2,by simp [sourceIO,inputs]⟩
        (⟨row%r.sequence,tableRow⟩,⟨j.val,hj⟩,PUnit.unit) (by omega)
    · exact bounded.2.1 ⟨3,by simp [sourceIO,inputs]⟩
        (⟨row%r.sequence,tableRow⟩,⟨j.val,hj⟩,PUnit.unit) (by omega)
  · intro key side h j hh hj
    have column := column_lt r side j.val hj
    cases key
    · exact bounded.2.1 ⟨0,by simp [sourceIO,inputs]⟩
        (⟨row,rowBound⟩,⟨h.val,hh⟩,⟨r.column side j.val,column⟩,PUnit.unit) trivial
    · exact bounded.2.1 ⟨1,by simp [sourceIO,inputs]⟩
        (⟨row,rowBound⟩,⟨h.val,hh⟩,⟨r.column side j.val,column⟩,PUnit.unit) trivial

theorem q_binding (M : Algebra β) (encode : α → β) (observe : β → α)
    (r : Request) (xs : Inputs α r) (s : State β)
    (pins : (sourceIO M encode observe r).InputBinding xs s)
    (key : Bool) (row head col : Nat) (hr : row<r.rows)
    (hh : head<r.heads key) (hc : col<r.dimension) :
    (s.mem (region key) (r.offset key row head col)).read .fp32 =
      encode (q xs key row head col) := by
  cases key
  · have heads : head<r.qHeads := hh
    have h := pins ⟨0,by simp [sourceIO,inputs]⟩
      (⟨row,hr⟩,⟨head,hh⟩,⟨col,hc⟩,PUnit.unit) trivial
    simpa [q,hr,hc,Request.heads,region,heads,mapValue,inputs] using h
  · have heads : head<r.kHeads := hh
    have h := pins ⟨1,by simp [sourceIO,inputs]⟩
      (⟨row,hr⟩,⟨head,hh⟩,⟨col,hc⟩,PUnit.unit) trivial
    simpa [q,hr,hc,Request.heads,region,heads,mapValue,inputs] using h

theorem coefficient_binding (M : Algebra β) (encode : α → β) (observe : β → α)
    (r : Request) (xs : Inputs α r) (s : State β)
    (pins : (sourceIO M encode observe r).InputBinding xs s)
    (sine : Bool) (row lane : Nat) (hr : row<r.rows) (hl : lane<r.half) :
    (s.mem (if sine then "S" else "C") (row%r.sequence*r.half+lane)).read .fp32 =
      encode (coefficient xs sine (row%r.sequence) lane) := by
  have ht := Nat.mod_lt row (sequence_positive r row hr)
  cases sine
  · have h := pins ⟨2,by simp [sourceIO,inputs]⟩
      (⟨row%r.sequence,ht⟩,⟨lane,hl⟩,PUnit.unit) (by omega)
    simpa [coefficient,ht,hl,mapValue] using h
  · have h := pins ⟨3,by simp [sourceIO,inputs]⟩
      (⟨row%r.sequence,ht⟩,⟨lane,hl⟩,PUnit.unit) (by omega)
    simpa [coefficient,ht,hl,mapValue] using h

/-- Every offset in the dense output region has a declared finite coordinate. -/
theorem dense_coordinate (r : Request) (key : Bool) (offset : Nat)
    (inside : offset<r.rows*r.stride key) :
    ∃ row head col,row<r.rows ∧ head<r.heads key ∧ col<r.dimension ∧
      r.offset key row head col=offset := by
  have dpos : 0<r.dimension := by have := r.halfPositive;unfold Request.dimension;omega
  have spos : 0<r.stride key := Nat.mul_pos (heads_positive r key) dpos
  have hr : offset/r.stride key<r.rows := (Nat.div_lt_iff_lt_mul spos).mpr inside
  have rem : offset%r.stride key<r.stride key := Nat.mod_lt offset spos
  have hh : offset%r.stride key/r.dimension<r.heads key :=
    (Nat.div_lt_iff_lt_mul dpos).mpr rem
  have hc : offset%r.stride key%r.dimension<r.dimension := Nat.mod_lt _ dpos
  refine ⟨offset/r.stride key,offset%r.stride key/r.dimension,
    offset%r.stride key%r.dimension,hr,hh,hc,?_⟩
  have h1 := Nat.div_add_mod offset (r.stride key)
  have h2 := Nat.div_add_mod (offset%r.stride key) r.dimension
  unfold Request.offset
  nlinarith [h1,h2]

theorem io_frame (M : Algebra β) (encode : α → β) (observe : β → α)
    (r : Request) (xs : Inputs α r) (s : State β) :
    (sourceIO M encode observe r).Frame xs s (Grid.run M r s r.rows) := by
  intro name offset outside
  apply Grid.prefix_frame
  intro key
  by_cases same : name=region key
  · right
    by_contra notBound
    obtain ⟨row,head,col,hr,hh,hc,address⟩ := dense_coordinate r key offset (by omega)
    cases key
    · have excluded := outside.1 ⟨0,by simp [sourceIO,outputs]⟩
        (⟨row,hr⟩,⟨head,hh⟩,⟨col,hc⟩,PUnit.unit) trivial
      exact excluded.elim (fun h => h same) (fun h => h address.symm)
    · have excluded := outside.1 ⟨1,by simp [sourceIO,outputs]⟩
        (⟨row,hr⟩,⟨head,hh⟩,⟨col,hc⟩,PUnit.unit) trivial
      exact excluded.elim (fun h => h same) (fun h => h address.symm)
  · exact Or.inl same

/-- Exact source-derived scalar target only used to compose the shared IO
proof. Public specifications specialize this to independent Real/native formulas. -/
def scalarTarget (M : Algebra β) (encode : α → β) (observe : β → α) :=
  target (fun a b c sn side => observe (scalar M (encode a) (encode b) (encode c) (encode sn) side))

theorem output_value (M : Algebra β) (encode : α → β) (observe : β → α)
    (r : Request) (xs : Inputs α r) (s : State β)
    (pins : (sourceIO M encode observe r).InputBinding xs s)
    (key : Bool) (row head col : Nat) (hr : row<r.rows)
    (hh : head<r.heads key) (hc : col<r.dimension) :
    mapCell observe ((Grid.run M r s r.rows).mem (region key) (r.offset key row head col))=
      Cell.mk .fp32
        (let lane := if col<r.half then col else col-r.half
         observe (scalar M
           (encode (q xs key row head lane)) (encode (q xs key row head (lane+r.half)))
           (encode (coefficient xs Bool.false (row%r.sequence) lane))
           (encode (coefficient xs Bool.true (row%r.sequence) lane)) (decide (r.half≤col)))) := by
  let lane := if col<r.half then col else col-r.half
  let side := decide (r.half≤col)
  have hl : lane<r.half := by
    dsimp [lane]
    have := r.halfPositive
    unfold Request.dimension at hc
    split <;> omega
  have hfirst : lane<r.dimension := lt_of_lt_of_le hl (by unfold Request.dimension;omega)
  have hsecond : lane+r.half<r.dimension := by unfold Request.dimension;omega
  have hcol : r.column side lane=col := by
    dsimp [Request.column,side,lane]
    by_cases low : col<r.half <;> simp [low,show (r.half≤col)↔¬col<r.half from by omega] <;> omega
  have cell := Grid.outputs M r s r.rows key row head side lane hr hh hl
  rw [hcol] at cell
  rw [cell]
  unfold value
  rw [q_binding M encode observe r xs s pins key row head lane hr hh hfirst,
    q_binding M encode observe r xs s pins key row head (lane+r.half) hr hh hsecond]
  have cos := coefficient_binding M encode observe r xs s pins Bool.false row lane hr hl
  have sin := coefficient_binding M encode observe r xs s pins Bool.true row lane hr hl
  simp only [Bool.false_eq_true,↓reduceIte] at cos sin
  rw [cos,sin]
  rfl

theorem implements (M : Algebra β) (encode : α → β) (observe : β → α) (r : Request) :
    (sourceIO M encode observe r).Implements (scalarTarget M encode observe r) := by
  apply ProgramIO.Implements.intro
  · intro xs bounds bounded initial _
    exact io_safe M encode observe r xs bounds bounded initial
  · intro xs initial pins
    refine ⟨Grid.run M r initial r.rows,schedule_run M r initial r.rows,?_,
      io_frame M encode observe r xs initial⟩
    intro o j _
    fin_cases o
    · exact output_value M encode observe r xs initial pins Bool.false
        j.1.val j.2.1.val j.2.2.1.val j.1.isLt j.2.1.isLt j.2.2.1.isLt
    · exact output_value M encode observe r xs initial pins Bool.true
        j.1.val j.2.1.val j.2.2.1.val j.1.isLt j.2.1.isLt j.2.2.1.isLt

#axiomsClean implements
end
end VeriTile.Bench.Repairs.RopeTransform.PublicIO
