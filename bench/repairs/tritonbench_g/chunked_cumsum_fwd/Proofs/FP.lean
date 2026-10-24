import bench.repairs.tritonbench_g.chunked_cumsum_fwd.Proofs.HostStorage
import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.RawHostABI
import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.RawNumericalPorts

namespace VeriTile.Bench.Repairs.ChunkedCumsumFwd.FPProof
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd
open RawBoundary HostModel HostStorage
noncomputable section
set_option maxHeartbeats 4000000

variable {call : Call Word} {entry : PhysicalMemory Word} {old : Nat → Prop}
  {logical : Logical Word} {g : HostModel.Geometry call}
  {a : AllocationResult Word call g.host entry old} {p : Prepared call g logical a}

/-- Returned object identity, actual words and all four bytes from each same-call
output allocation. The caller frame includes every existing backing byte. -/
def Observables (b : Backings p) (plans : Plans Word p.host) (final : PhysicalMemory Word) : Prop :=
  b.sum.object = a.sumObject ∧ b.out.object = a.outObject ∧
  b.sum.view = a.sum ∧ b.out.view = a.out ∧
  b.sum.device = call.dt.device ∧ b.out.device = call.dt.device ∧
  b.sum.shape = [g.host.batches,g.host.heads,g.host.chunks,g.host.chunk] ∧
  b.out.shape = [g.host.batches,g.host.heads,g.host.chunks,g.host.chunk] ∧
  b.sum.dtype = .fp32 ∧ b.out.dtype = .fp32 ∧
  (∀ batch head chunk lane, batch < g.host.batches → head < g.host.heads → chunk < g.host.chunks →
    ∀ hj : lane < g.host.chunk,
    let off := Whole.packed g.host batch head chunk lane
    let prepared := preparedValue plans.selected.model (reblock p.host plans.selected.block) logical
      call.hasBias call.softplus call.lower call.upper batch head chunk lane
    let cumulative := cumulativeValue plans.selected.model (reblock p.host plans.selected.block)
      plans.selected.scan logical call.hasBias call.softplus call.lower call.upper batch head chunk
        ⟨lane,lt_of_lt_of_le hj g.host.chunk_le_BC⟩
    b.out.view.read final off = prepared ∧ b.sum.view.read final off = cumulative ∧
    (∀ i, i < 4 → Storage.bytes final b.out.view.allocation (4*(b.out.view.base+off)+i) = Storage.byte prepared i) ∧
    (∀ i, i < 4 → Storage.bytes final b.sum.view.allocation (4*(b.sum.view.base+off)+i) = Storage.byte cumulative i)) ∧
  (∀ allocation byte, old allocation → byte < (b.allocation allocation).bytes →
    Storage.bytes final allocation byte = Storage.bytes entry allocation byte)

theorem observables (b : Backings p) (plans : Plans Word p.host) (final : PhysicalMemory Word)
    (values : HostModel.Values call logical p plans final)
    (frame : ∀ allocation off, old allocation → final allocation off = entry allocation off) :
    Observables b plans final := by
  refine ⟨b.sumObject,b.outObject,b.sumView,b.outView,b.sumDevice,b.outDevice,b.sumShape,b.outShape,b.sum.fp32,b.out.fp32,?_,?_⟩
  · intro batch head chunk lane hb hh hk hj
    have val := values batch head chunk lane hb hh hk hj
    have out : b.out.view = p.views.out := b.outView
    have sum : b.sum.view = p.views.sum := b.sumView
    refine ⟨by simpa only [out] using val.1,by simpa only [sum] using val.2,?_,?_⟩
    · intro i hi
      rw [Storage.read_bytes final b.out _ i hi,out,val.1]
    · intro i hi
      rw [Storage.read_bytes final b.sum _ i hi,sum,val.2]
  · intro allocation byte hold _
    exact Storage.all_byte_frame entry final old frame allocation byte hold

/-- The same actual returned descriptors now denote one flat byte memory per
device. Entry bytes need represent only preexisting allocations; the launch
heap represents all actual allocations after ordinary allocation.
All output byte values are read from the derived source result. -/
def FlatObservables (b : Backings p) (final : PhysicalMemory Word) : Prop :=
  ∀ entryFlat launchFlat : Storage.FlatHeap,
    Storage.Represents b.allocation old entry entryFlat →
    Storage.Represents b.allocation b.live p.memory launchFlat →
    let returned := Storage.realize b.allocation b.live final launchFlat
    Storage.Represents b.allocation b.live final returned ∧
    (∀ off, off < g.host.outputSize → ∀ i, i < 4 →
      returned b.out.device ((Index.address b.out.dataPtr (Index.u off)).toNat+i) =
        Storage.byte (b.out.view.read final off) i ∧
      returned b.sum.device ((Index.address b.sum.dataPtr (Index.u off)).toNat+i) =
        Storage.byte (b.sum.view.read final off) i) ∧
    (∀ allocation byte, old allocation → byte < (b.allocation allocation).bytes →
      returned (b.allocation allocation).device ((b.allocation allocation).address+byte) =
        entryFlat (b.allocation allocation).device ((b.allocation allocation).address+byte)) ∧
    (∀ device address,
      (¬ ∃ allocation byte, b.live allocation ∧ byte < (b.allocation allocation).bytes ∧
        p.views.written p.host allocation (byte/4) ∧
        (b.allocation allocation).device = device ∧ (b.allocation allocation).address+byte = address) →
      returned device address = launchFlat device address)

theorem flat_observables (b : Backings p) (final : PhysicalMemory Word)
    (outside : ∀ allocation off, ¬ p.views.written p.host allocation off →
      final allocation off = p.memory allocation off)
    (caller : ∀ allocation off, old allocation → final allocation off = entry allocation off) :
    FlatObservables b final := by
  intro entryFlat launchFlat entryRep launchRep
  have represented := Storage.realize_represents b.allocation b.live b.separated final launchFlat
  refine ⟨represented,?_,?_,?_⟩
  · intro off ho i hi
    exact ⟨Storage.represented_native_bytes represented b.out b.outLive off (b.outCovers off ho) i hi,
      Storage.represented_native_bytes represented b.sum b.sumLive off (b.sumCovers off ho) i hi⟩
  · intro allocation byte held bounded
    exact Storage.represented_frame entryRep represented b.callerLive caller allocation byte held bounded
  · intro device address untouched
    exact Storage.realize_outside launchRep outside device address untouched

/-- Every autotune trial is the unchanged complete source execution, threaded
through the preceding trial's heap. Its native byte representation and frames
follow from the same allocation descriptors as the selected launch. -/
theorem trials_flat (b : Backings p) (trials : List (LaunchPlan Word p.host)) :
    ∃ final, runTrials p.views p.host call.hasBias call.softplus call.lower call.upper trials p.memory = some final ∧
      FlatObservables b final := by
  obtain ⟨final,ran,outside,kept⟩ := trials_frame p.views p.host p.fresh
    call.hasBias call.softplus call.lower call.upper trials p.memory
  refine ⟨final,ran,flat_observables b final outside ?_⟩
  intro allocation off caller
  exact (kept allocation off (p.callerProtected allocation caller)).trans
    ((p.frame allocation off caller).trans (a.frame allocation off caller))

def HostSpec (call : Call Word) (logical : Logical Word)
    (rt : HostModel.Runtime call logical entry old) (storage : OrdinaryStorage rt)
    (outcome : HostModel.Outcome Word) : Prop :=
  HostModel.HostSpec call logical rt outcome ∧
  (HostModel.Success call logical rt outcome → ∃ g a p plans final,
    ∃ hg : HostModel.geometry call = .ok g, ∃ ha : rt.allocate g = .ok a,
    ∃ hp : rt.inputs g a = p, rt.setup g a p = .ok plans ∧
    outcome = .returned a.sumObject a.outObject a.sum a.out
      [g.host.batches,g.host.heads,g.host.chunks,g.host.chunk] call.dt.device final ∧
    Facts (storage g a p ha hp) ∧ Observables (storage g a p ha hp) plans final ∧
    FlatObservables (storage g a p ha hp) final ∧
    (∀ allocation off, ¬ p.views.written p.host allocation off → final allocation off = p.memory allocation off))

theorem outcomes (call : Call Word) (logical : Logical Word)
    (rt : HostModel.Runtime call logical entry old) (storage : OrdinaryStorage rt)
    (checkedSizes : ∀ g : HostModel.Geometry call, Native.Sizes g.host) :
    HostSpec call logical rt storage (HostModel.execute call logical rt) := by
  refine ⟨HostModel.outcomes call logical rt,?_⟩
  rintro ⟨g,a,p,plans,final,hg,ha,hp,ht,returned,val,frame,caller⟩
  exact ⟨g,a,p,plans,final,hg,ha,hp,ht,returned,
    actual (storage g a p ha hp) (prepared_sizes p (checkedSizes g)),
    observables (storage g a p ha hp) plans final val caller,
    flat_observables (storage g a p ha hp) final frame caller,frame⟩

/-- Source launcher conversion occurs after setup, before any autotune launch.
The complete x86-64 binary64→FP32 ABI algorithm is reused unchanged. -/
def guarded (raw : Call RawABI.Scalar)
    (logical : Logical Word) (rt : HostModel.Runtime (RawABI.materialize raw) logical entry old) :
    HostModel.Runtime (RawABI.materialize raw) logical entry old :=
  { rt with setup := fun g a p => match rt.setup g a p with
      | .error reason => .error reason
      | .ok plans => match RawABI.extractBounds raw.lower raw.upper with
        | .error reason => .error reason
        | .ok _ => .ok plans }

theorem bounds_binding (raw : Call RawABI.Scalar) (logical : Logical Word)
    (rt : HostModel.Runtime (RawABI.materialize raw) logical entry old)
    (g : HostModel.Geometry (RawABI.materialize raw))
    (a : AllocationResult Word (RawABI.materialize raw) g.host entry old)
    (p : Prepared (RawABI.materialize raw) g logical a) (plans : Plans Word p.host)
    (ok : (guarded raw logical rt).setup g a p = .ok plans) :
    RawABI.extractBounds raw.lower raw.upper =
      .ok ((RawABI.materialize raw).lower,(RawABI.materialize raw).upper) := by
  cases hs : rt.setup g a p <;> cases hb : RawABI.extractBounds raw.lower raw.upper <;>
    simp [guarded,hs,hb] at ok
  exact congrArg Except.ok (RawABI.materialize_binding raw _ hb)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkedCumsumFwd.FPProof
