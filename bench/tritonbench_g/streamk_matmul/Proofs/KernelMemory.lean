import bench.tritonbench_g.streamk_matmul.Proofs.SourceBridge
import bench.tritonbench_g.streamk_matmul.Proofs.ValueMemory

namespace VeriTile.Bench.Repairs.StreamK.TileChecked.KernelMemory
open Whole001 Source Checked
noncomputable section
set_option maxHeartbeats 12000000

/-- Source-driven physical value execution on caller-provided buffers. The
wrapper/allocator/dispatch are outside this kernel theorem. -/
def runValues (p : Source.Parameters) (ops : Instructions α) (v : ByteExecution.Layout p)
    (initial : Source.State α p exterior) (before : TensorBinding.ValueRead α) :
    Option (ValueMemory.Machine α p exterior) := do
  let program ← Reader.compile kernels
  if p.h.m=0 ∨ p.h.n=0 then some (initial,before) else
    let e ← Reader.events p program ops
    some (ValueMemory.run v ops.atomicAdd e.full
      (ValueMemory.run v ops.atomicAdd e.reduce (ValueMemory.run v ops.atomicAdd e.first (initial,before))))

/-- Byte stores and current-value atomics execute the extracted source events. -/
def runBytes (p : Source.Parameters) (ops : Instructions TensorBinding.Word) (v : ByteExecution.Layout p)
    (initial : Source.State TensorBinding.Word p exterior) (before : ByteMemory.Read) :
    Option (ByteExecution.Machine p exterior) := do
  let program ← Reader.compile kernels
  if p.h.m=0 ∨ p.h.n=0 then some (initial,before) else
    let e ← Reader.events p program ops
    some (ByteExecution.run v ops.atomicAdd e.full
      (ByteExecution.run v ops.atomicAdd e.reduce (ByteExecution.run v ops.atomicAdd e.first (initial,before))))

theorem reduction_allowed (p : Source.Parameters) (v : ByteExecution.Layout p)
    (e : Source.Event α p) (he : e∈reduceWave p) : ByteExecution.Allowed v e := by
  unfold reduceWave reducePairs at he
  obtain ⟨pair,hpair,rfl⟩ := List.mem_map.mp he
  obtain ⟨tile,ht,hpair⟩ := List.mem_flatMap.mp hpair
  have hs : 0<(Plan.select p.h).streamTiles := by have := List.mem_range.mp ht; omega
  simp [ByteExecution.Allowed,v.scratch_eq,hs]

theorem values_render (p : Source.Parameters) (ops : Instructions α) (v : ByteExecution.Layout p)
    (laws : ByteExecution.Laws v) (initial : Source.State α p exterior) (before : TensorBinding.ValueRead α) :
    runValues p ops v initial (ValueMemory.rendered before v initial)=
      some (Source.call p ops initial,ValueMemory.rendered before v (Source.call p ops initial)) := by
  unfold runValues
  rw [Reader.source_compiles]
  simp only [Option.bind_eq_bind,Option.bind_some,Option.pure_def]
  unfold Source.call
  split
  · rfl
  · rw [Reader.all_events]
    simp only [Option.bind_eq_bind,Option.bind_some,Option.pure_def]
    rw [ValueMemory.run_render v laws _ _ (ByteExecution.first_allowed v ops),
      ValueMemory.run_render v laws _ _ (reduction_allowed p v),
      ValueMemory.run_render v laws _ _ (ByteExecution.full_allowed v ops)]

theorem bytes_render (p : Source.Parameters) (ops : Instructions TensorBinding.Word) (v : ByteExecution.Layout p)
    (laws : ByteExecution.Laws v) (initial : Source.State TensorBinding.Word p exterior) (before : ByteMemory.Read) :
    runBytes p ops v initial (ByteExecution.rendered before v initial)=
      some (Source.call p ops initial,ByteExecution.rendered before v (Source.call p ops initial)) := by
  unfold runBytes
  rw [Reader.source_compiles]
  simp only [Option.bind_eq_bind,Option.bind_some,Option.pure_def]
  unfold Source.call
  split
  · rfl
  · rw [Reader.all_events]
    simp only [Option.bind_eq_bind,Option.bind_some,Option.pure_def]
    rw [ByteExecution.run_render v laws _ _ (ByteExecution.first_allowed v ops),
      ByteExecution.run_render v laws _ _ (reduction_allowed p v),
      ByteExecution.run_render v laws _ _ (ByteExecution.full_allowed v ops)]

/-- Initial output and scratch payloads are arbitrary bytes already present in
memory. These ordinary read bindings impose no computed-result condition. -/
structure ValueBinding (v : ByteExecution.Layout p) (s : Source.State α p exterior)
    (before : TensorBinding.ValueRead α) : Prop where
  output : ∀ i,s.output i=before (v.outputBase+4*i.val)
  scratch : v.scratchActive=Bool.true → ∀ i,s.scratch i=before (v.scratchBase+4*i.val)

structure ByteBinding (v : ByteExecution.Layout p) (s : Source.State TensorBinding.Word p exterior)
    (before : ByteMemory.Read) : Prop where
  output : ∀ i,(s.output i).bits=ByteMemory.read32 before (v.outputBase+4*i.val)
  scratch : v.scratchActive=Bool.true → ∀ i,(s.scratch i).bits=ByteMemory.read32 before (v.scratchBase+4*i.val)

theorem initial_values (v : ByteExecution.Layout p) (s : Source.State α p exterior)
    (before : TensorBinding.ValueRead α) (h : ValueBinding v s before) :
    ValueMemory.rendered before v s=before := by
  unfold ValueMemory.rendered
  split
  next active => rw [ValueMemory.initial _ _ _ (h.scratch active),ValueMemory.initial _ _ _ h.output]
  next => exact ValueMemory.initial _ _ _ h.output

theorem initial_bytes (v : ByteExecution.Layout p) (s : Source.State TensorBinding.Word p exterior)
    (before : ByteMemory.Read) (h : ByteBinding v s before) :
    ByteExecution.rendered before v s=before := by
  unfold ByteExecution.rendered
  have hs : ByteMemory.Bound before (ByteExecution.sView v) (ByteMemory.extend s.scratch) := by
    intro i hi
    by_cases active : v.scratchActive=Bool.true
    · have bound : i<p.h.programs*p.h.bm*p.h.bn := by simpa [ByteExecution.sView,active] using hi
      simpa only [ByteMemory.extend,dif_pos bound] using h.scratch active ⟨i,bound⟩
    · simp [ByteExecution.sView,active] at hi
  rw [ByteMemory.render_initial _ _ _ hs]
  apply ByteMemory.render_initial
  intro i hi
  change i<p.h.m*p.h.n at hi
  simpa only [ByteMemory.extend,dif_pos hi,ByteExecution.cView] using h.output ⟨i,hi⟩

theorem values_run (p : Source.Parameters) (ops : Instructions α) (v : ByteExecution.Layout p)
    (laws : ByteExecution.Laws v) (initial : Source.State α p exterior) (before : TensorBinding.ValueRead α)
    (binding : ValueBinding v initial before) :
    runValues p ops v initial before=
      some (Source.call p ops initial,ValueMemory.rendered before v (Source.call p ops initial)) := by
  have result := values_render p ops v laws initial before
  rwa [initial_values v initial before binding] at result

theorem bytes_run (p : Source.Parameters) (ops : Instructions TensorBinding.Word) (v : ByteExecution.Layout p)
    (laws : ByteExecution.Laws v) (initial : Source.State TensorBinding.Word p exterior) (before : ByteMemory.Read)
    (binding : ByteBinding v initial before) :
    runBytes p ops v initial before=
      some (Source.call p ops initial,ByteExecution.rendered before v (Source.call p ops initial)) := by
  have result := bytes_render p ops v laws initial before
  rwa [initial_bytes v initial before binding] at result

#axiomsClean values_run
#axiomsClean bytes_run
#specNonCircular ValueBinding avoiding [Source.call,Reader.execute,runValues,runBytes]
#specNonCircular ByteBinding avoiding [Source.call,Reader.execute,runValues,runBytes]
#specNonCircular runValues avoiding [Source.call,Source.firstWave,Source.fullWave,Source.reduceWave]
#specNonCircular runBytes avoiding [Source.call,Source.firstWave,Source.fullWave,Source.reduceWave]
end
end VeriTile.Bench.Repairs.StreamK.TileChecked.KernelMemory
