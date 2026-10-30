import bench.repairs.tritonbench_g.rbe_triton_transform_checked_launch.IO
import bench.repairs.tritonbench_g.rbe_triton_transform_checked_launch.Proofs.AccessSafety
import bench.repairs.tritonbench_g.rbe_triton_transform_checked_launch.Proofs.TypedGrid

namespace VeriTile.Bench.Repairs.RbeTritonTransformCheckedLaunch.IOCorrectness
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.RbeTritonTransform
noncomputable section
set_option maxHeartbeats 3000000
variable {α β : Type} [Inhabited α] [Inhabited β]

@[simp] theorem invocation_initial (ops : Operations β) (r : Request) (X Y : RegionName)
    (q : Runtime.Program) (s : State β) :
    (invocation ops r X Y q).initial s=Runtime.setup s q := rfl

/-- The public schedule executes the same selected source at every old launch. -/
theorem run_schedule (ops : Operations β) (r : Request) (X Y : RegionName)
    (qs : List Runtime.Program) (s : State β) :
    runSchedule (qs.map (invocation ops r X Y)) s=Runtime.run ops r X Y qs s := by
  induction qs generalizing s with
  | nil => rfl
  | cons q qs ih =>
    change (FP.Structural.exec ops.backend (Selected.source (parameters r X Y)) (Runtime.setup s q)).bind
      (runSchedule (qs.map (invocation ops r X Y))) =
      (FP.Structural.exec ops.backend (Selected.source (parameters r X Y)) (Runtime.setup s q)).bind
      (Runtime.run ops r X Y qs)
    cases h : FP.Structural.exec ops.backend (Selected.source (parameters r X Y)) (Runtime.setup s q) <;>
      simp only [Option.bind_none,Option.bind_some,ih]

/-- Explicit contiguous windows discharge source safety for each actual launch. -/
theorem invocation_safe (ops : Operations β) (r : Request) (X Y : RegionName)
    (q : Runtime.Program) (hq : q.1<r.batches) (bounds : RegionName → Nat) (s : State β)
    (hin : ∀ i,i<r.size → i<bounds X) (hout : ∀ i,i<r.size → i<bounds Y) :
    (invocation ops r X Y q).Safe bounds s := by
  change TraceSafe ops.backend bounds (Selected.source (parameters r X Y)) (Runtime.setup s q)
  apply AccessSafety.source
  · intro im i active
    have h : Source.row (parameters r X Y) (Runtime.setup s q) i<r.rows ∧
        Source.column (parameters r X Y) (Runtime.setup s q) i+(if im then 1 else 0)<r.columns := by
      cases im <;> simpa [Source.active,parameters,Nat.add_comm] using active
    have within := offset_lt r q.1 _ _ hq h.1 h.2
    apply hin
    convert within using 1 <;>
      simp [Source.inputOffset,parameters,Runtime.setup,Request.offset] <;> ring
  · intro im i active
    have h : Source.row (parameters r X Y) (Runtime.setup s q) i<r.rows ∧
        Source.column (parameters r X Y) (Runtime.setup s q) i+(if im then 1 else 0)<r.columns := by
      cases im <;> simpa [Source.active,parameters,Nat.add_comm] using active
    have within := offset_lt r q.1 _ _ hq h.1 h.2
    apply hout
    convert within using 1 <;>
      simp [Source.outputOffset,parameters,Runtime.setup,Request.offset] <;> ring

/-- Safety follows every invocation's actual predecessor state. -/
theorem schedule_safe (ops : Operations β) (r : Request) (X Y : RegionName)
    (qs : List Runtime.Program) (valid : ∀ q∈qs,q.1<r.batches)
    (bounds : RegionName → Nat) (s : State β)
    (hin : ∀ i,i<r.size → i<bounds X) (hout : ∀ i,i<r.size → i<bounds Y) :
    ScheduleSafe bounds (qs.map (invocation ops r X Y)) s := by
  induction qs generalizing s with
  | nil => trivial
  | cons q qs ih =>
    refine ⟨invocation_safe ops r X Y q (valid q (List.mem_cons_self ..)) bounds s hin hout,?_⟩
    intro t _
    exact ih (fun p hp => valid p (List.mem_cons_of_mem _ hp)) t

/-- Every flat logical output denotes exactly one legal dense tensor coordinate. -/
theorem coordinates_valid (r : Request) (i : Nat) (hi : i<r.size) :
    (coordinates r i).1<r.batches ∧ (coordinates r i).2.1<r.rows ∧
      (coordinates r i).2.2<r.columns ∧
      r.offset (coordinates r i).1 (coordinates r i).2.1 (coordinates r i).2.2=i := by
  have positive : 0<r.batches*r.rows*r.columns := lt_of_le_of_lt (Nat.zero_le i) hi
  have cols : 0<r.columns := by
    by_contra h
    have z : r.columns=0 := by omega
    simp [z] at positive
  have rows : 0<r.rows := by
    by_contra h
    have z : r.rows=0 := by omega
    simp [z] at positive
  have product : 0<r.rows*r.columns := Nat.mul_pos rows cols
  have batch : i/(r.rows*r.columns)<r.batches :=
    (Nat.div_lt_iff_lt_mul product).mpr (by simpa [Request.size,Nat.mul_assoc] using hi)
  refine ⟨batch,Nat.mod_lt _ rows,Nat.mod_lt _ cols,?_⟩
  have quotient : i/(r.rows*r.columns)=(i/r.columns)/r.rows := by
    rw [Nat.div_div_eq_div_mul,Nat.mul_comm]
  have hrow : i/(r.rows*r.columns)*r.rows+(i/r.columns)%r.rows=i/r.columns := by
    simpa only [quotient,Nat.mul_comm,Nat.add_comm] using Nat.mod_add_div (i/r.columns) r.rows
  change (i/(r.rows*r.columns)*r.rows+(i/r.columns)%r.rows)*r.columns+i%r.columns=i
  rw [hrow]
  simpa only [Nat.mul_comm,Nat.add_comm] using Nat.mod_add_div i r.columns

/-- The scalar formula reads only the declared dense input window. -/
theorem scalar_congr (ops : Operations β) (r : Request) (x y : Nat → β)
    (same : ∀ i,i<r.size → x i=y i) (b m c : Nat)
    (hb : b<r.batches) (hm : m<r.rows) (hc : c<r.columns) :
    scalarResult ops r x b m c=scalarResult ops r y b m c := by
  have even : c/2*2<r.columns := lt_of_le_of_lt (Nat.div_mul_le_self c 2) hc
  have hx := same _ (offset_lt r b m (c/2*2) hb hm even)
  simp only [scalarResult,scalarPair,hx]
  by_cases hp : c/2*2+1<r.columns
  · rw [if_pos hp,if_pos hp,same _ (offset_lt r b m _ hb hm hp)]
  · rw [if_neg hp,if_neg hp]

/-- A source-independent primitive scalar target on the public logical arrays. -/
def target (ops : Operations β) (r : Request) (X Y : RegionName) (qs : List Runtime.Program)
    (encode : α → β) (observe : β → α)
    (xs : (rotationIO ops r X Y qs encode observe).Inputs) :
    (rotationIO ops r X Y qs encode observe).Outputs := by
  intro o
  have h : o=0 := Fin.eq_zero o
  subst o
  exact fun j => observe (scalarResult ops r (fun i => encode (logicalInput r (xs 0) i))
    (coordinates r j.1.val).1 (coordinates r j.1.val).2.1 (coordinates r j.1.val).2.2)

/-- Compose the existing complete-grid value/frame result with real source safety. -/
theorem implements (ops : Operations β) (r : Request) (X Y : RegionName) (different : X≠Y)
    (qs : List Runtime.Program) (all : qs.Perm (Runtime.programs r))
    (encode : α → β) (observe : β → α) :
    (rotationIO ops r X Y qs encode observe).Implements (target ops r X Y qs encode observe) := by
  apply ProgramIO.Implements.intro
  · intro xs bounds bounded initial _
    apply schedule_safe ops r X Y qs
      (fun q hq => ((Runtime.mem_programs r q).mp (all.mem_iff.mp hq)).1) bounds initial
    · intro i hi
      exact bounded.2.1 0 (⟨i,hi⟩,PUnit.unit) trivial
    · intro i hi
      exact bounded.2.2.1 0 (⟨i,hi⟩,PUnit.unit) trivial
  · intro xs initial bound
    obtain ⟨final,ran,values,frame⟩ := TypedGrid.whole ops r X Y qs all initial different
    refine ⟨final,(run_schedule ops r X Y qs initial).trans ran,?_,?_⟩
    · intro o j _
      have oz : o=0 := Fin.eq_zero o
      subst o
      have valid := coordinates_valid r j.1.val j.1.isLt
      have val := values _ _ _ valid.1 valid.2.1 valid.2.2.1
      rw [valid.2.2.2] at val
      change mapCell observe (final.mem Y j.1.val)=Cell.mk .fp32 _
      rw [val]
      change Cell.mk .fp32 (observe (scalarResult ops r _ _ _ _))=Cell.mk .fp32
        (observe (scalarResult ops r (fun i => encode (logicalInput r (xs 0) i)) _ _ _))
      apply congrArg (fun v : β => Cell.mk .fp32 (observe v))
      apply scalar_congr ops r _ _ _ _ _ _ valid.1 valid.2.1 valid.2.2.1
      intro i hi
      have read := bound 0 (⟨i,hi⟩,PUnit.unit) trivial
      simpa only [rotationIO,mapValue,logicalInput,hi,↓reduceDIte] using read
    · intro region i outside
      apply frame
      by_cases name : region=Y
      · right
        by_contra h
        have hi : i<r.size := Nat.lt_of_not_ge h
        have miss := outside.1 0 (⟨i,hi⟩,PUnit.unit) trivial
        simp [rotationIO,name] at miss
        exact miss rfl
      · exact Or.inl name

#axiomsClean implements
end
end VeriTile.Bench.Repairs.RbeTritonTransformCheckedLaunch.IOCorrectness
