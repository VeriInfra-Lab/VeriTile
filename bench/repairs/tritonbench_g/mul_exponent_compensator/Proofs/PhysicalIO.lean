import bench.repairs.tritonbench_g.mul_exponent_compensator.Proofs.IOComplete
import bench.repairs.tritonbench_g.mul_exponent_compensator.Proofs.Native

/-! Connection of the shared actual-source IO to the existing finite physical
interpreter. It uses the established source lowering/index/mask/address theorem,
not a new execution callback. The same-address in-place case is represented by
equal logical regions. Disjoint ranges retain two regions. Every exterior flat
cell is preserved when the actual source output is written back. -/
namespace VeriTile.Bench.Repairs.MulExponentCompensator.Contiguous.PhysicalIO
open VeriTile.Triton FP.Structural Interface
open VeriTile.Bench.Repairs.TritonMul2 (Layout Allocated Memory launch)
noncomputable section
variable {α : Type} [Inhabited α]

def inputRegion (l : Layout) : RegionName := if l.input=l.output then "dst" else "src"

def values (l : Layout) (before : Memory α) : ChannelValues α (inputChannels (inputRegion l) l.count) :=
  fun port => by
    have h : port=0 := Fin.eq_zero port
    subst port
    exact fun j=>before (l.input+j.1.val)

def initial (l : Layout) (before : Memory α) : State α where
  mem := fun r i=>Cell.mk .real (before ((if r=inputRegion l then l.input else l.output)+i))
  regs := fun _ _ _=>none
  pids := fun _=>0
  numPids := fun _=>grid l.count l.block
  undef := fun d _ _=>defaultValue d

def bounds (l : Layout) (r : RegionName) : Nat :=
  if r=inputRegion l ∨ r="dst" then l.count else 0

def writeBack (l : Layout) (before : Memory α) (final : State α) : Memory α :=
  fun cell=>if l.output ≤ cell ∧ cell<l.output+l.count
    then (final.mem "dst" (cell-l.output)).read .real else before cell

theorem writeBack_value (l : Layout) (before : Memory α) (final : State α)
    (i : Nat) (hi : i<l.count) :
    writeBack l before final (l.output+i)=(final.mem "dst" i).read .real := by
  simp [writeBack,hi]

theorem writeBack_frame (l : Layout) (before : Memory α) (final : State α)
    (cell : Nat) (outside : ¬∃ i,i<l.count ∧ cell=l.output+i) :
    writeBack l before final cell=before cell := by
  apply if_neg
  rintro ⟨lo,hi⟩
  exact outside ⟨cell-l.output,by omega,by omega⟩

theorem in_bounds (M : Algebra α) (l : Layout) (order : List Nat) (before : Memory α) :
    (io M (inputRegion l) "dst" l.count l.block order).InBounds
      (bounds l) (values l before) := by
  refine ⟨?_,?_,?_,?_⟩
  · intro r hr
    have hr' : r≠inputRegion l ∧ r≠("dst" : RegionName) := by
      simpa [io] using hr
    simp [bounds,hr'.1,hr'.2]
  · intro port j _
    have h : port=(0 : Fin 1) := Fin.eq_zero port
    subst port
    simpa [io,bounds] using j.1.isLt
  · intro port j _
    have h : port=(0 : Fin 1) := Fin.eq_zero port
    subst port
    simpa [io,bounds] using j.1.isLt
  · intro port
    exact port.elim0

theorem input_binding (M : Algebra α) (l : Layout) (order : List Nat) (before : Memory α) :
    (io M (inputRegion l) "dst" l.count l.block order).InputBinding
      (values l before) (initial l before) := by
  intro port j _
  have h : port=(0 : Fin 1) := Fin.eq_zero port
  subst port
  change ((initial l before).mem (inputRegion l) j.1.val).read .real=before (l.input+j.1.val)
  simp [initial]

theorem mapCell_id (c : Cell α) : mapCell id c=c := by
  cases c with
  | mk d v => cases d <;> rfl

/-- Both interpreters execute the same candidate source and agree after the
actual regional output is relocated to the original flat addresses. All old
finite bounds, mask tails, alias alternatives and complete orders are retained. -/
theorem transport (M : Algebra α) (f : α → α) (identify : ∀ x,scalar M x=f x)
    (l : Layout) (blocks : l.block∈supportedBlocks)
    (gridBound : grid l.count l.block ≤ 2^31-1) (allocation : Allocated l)
    (order : List Nat) (schedule : order.Perm (List.range (grid l.count l.block)))
    (before : Memory α) :
    ∃ final,
      ScheduleSafe (bounds l) (order.map (invocation M (inputRegion l) "dst" l.count l.block)) (initial l before) ∧
      runSchedule (order.map (invocation M (inputRegion l) "dst" l.count l.block)) (initial l before)=some final ∧
      (∀ i,i<l.count → final.mem "dst" i=Cell.mk .real (f (before (l.input+i)))) ∧
      (io M (inputRegion l) "dst" l.count l.block order).Frame (values l before) (initial l before) final ∧
      Native.sourceLaunch ⟨f⟩ l order before=some (writeBack l before final) := by
  have certified := IOComplete.implements M f identify (inputRegion l) "dst" l.count l.block
    (block_positive l.block blocks) order schedule
  obtain ⟨safe,final,ran,output,frame⟩ := certified (values l before) trivial (bounds l)
    (in_bounds M l order before) (initial l before) (input_binding M l order before)
  have result : ∀ i,i<l.count → final.mem "dst" i=Cell.mk .real (f (before (l.input+i))) := by
    intro i hi
    have h := output (0 : Fin 1) (⟨i,hi⟩,PUnit.unit) trivial
    change mapCell id (final.mem "dst" i)=Cell.mk .real (f (before (l.input+i))) at h
    rwa [mapCell_id] at h
  refine ⟨final,safe,ran,result,frame,?_⟩
  rw [Native.source_launch_equal ⟨f⟩ l blocks gridBound allocation order
    (fun pid hp=>List.mem_range.mp (schedule.mem_iff.mp hp)),
    TritonMul2.Scheduling.order_equal ⟨f⟩ l allocation order schedule]
  apply congrArg some
  have physical := TritonMul2.launch_correct ⟨f⟩ l allocation (block_positive l.block blocks) before
  funext cell
  by_cases touched : ∃ i,i<l.count ∧ cell=l.output+i
  · obtain ⟨i,hi,rfl⟩ := touched
    rw [writeBack_value l before final i hi,result i hi,Cell.read_same]
    exact physical.1 i hi
  · rw [writeBack_frame l before final cell touched]
    exact physical.2 cell touched

/-- Original raw output/frame statement, now obtained through the shared source
IO and its exact finite-source relocation theorem above. -/
theorem complete (M : Algebra α) (f : α → α) (identify : ∀ x,scalar M x=f x)
    (l : Layout) (blocks : l.block∈supportedBlocks)
    (gridBound : grid l.count l.block ≤ 2^31-1) (allocation : Allocated l)
    (order : List Nat) (schedule : order.Perm (List.range (grid l.count l.block)))
    (before : Memory α) :
    ∃ final,Native.sourceLaunch ⟨f⟩ l order before=some final ∧
      (∀ i,i<l.count → final (l.output+i)=f (before (l.input+i))) ∧
      (∀ cell,(¬∃ i,i<l.count ∧ cell=l.output+i) → final cell=before cell) := by
  obtain ⟨regional,_,_,output,_,executed⟩ := transport M f identify l blocks gridBound allocation order schedule before
  refine ⟨writeBack l before regional,executed,?_,writeBack_frame l before regional⟩
  intro i hi
  rw [writeBack_value l before regional i hi,output i hi,Cell.read_same]

#axiomsClean transport
#axiomsClean complete
end
end VeriTile.Bench.Repairs.MulExponentCompensator.Contiguous.PhysicalIO
