import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.Proofs.NativeResults
import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.InputsPhysical

/-! Every old arbitrary-word input has the declared finite IO representation.
The same representation samples the existing actual Tensor byte reader; no
extra finite-word, non-NaN, geometry, or precision hypothesis is introduced. -/
namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly.NativePhysicalIO
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.LightningAttention.Native (Word Instructions Plan)
open NativeCoverage NativeMaterialization
noncomputable section
set_option maxHeartbeats 6000000

def forwardValues (d : Host.Dimensions) (words : Native.WordMemory) :
    ChannelValues Word (NativeIO.forwardInputs d) :=
  fun i j => match i with
  | ⟨0,_⟩ => words "query" (denseIndex j)
  | ⟨1,_⟩ => words "key" (denseIndex j)
  | ⟨2,_⟩ => words "value" (NativeIO.paddedIndex d j)
  | ⟨3,_⟩ => words "output" (NativeIO.paddedIndex d j)

def backwardValues (d : Host.Dimensions) (words : Native.WordMemory) :
    ChannelValues Word (NativeIO.backwardInputs d) :=
  fun i j => match i with
  | ⟨0,_⟩ => words "query" (denseIndex j)
  | ⟨1,_⟩ => words "key" (denseIndex j)
  | ⟨2,_⟩ => words "value" (denseIndex j)
  | ⟨3,_⟩ => words "incoming" (denseIndex j)
  | ⟨4,_⟩ => words "queryGradient" (denseIndex j)
  | ⟨5,_⟩ => words "keyGradient" (denseIndex j)
  | ⟨6,_⟩ => words "valueGradient" (denseIndex j)

theorem forward_binding (ops : Instructions) (plan : Plan) (d : Host.Dimensions)
    (words : Native.WordMemory) :
    (NativeIO.forward ops plan d).InputCells (forwardValues d words) (typed words) := by
  intro i j _
  fin_cases i <;> rfl

theorem backward_binding (ops : Instructions) (plan : Plan) (d : Host.Dimensions)
    (words : Native.WordMemory) :
    (NativeIO.backward ops plan d).InputCells (backwardValues d words) (typed words) := by
  intro i j _
  fin_cases i <;> rfl

theorem forward_reference (ops : Instructions) (plan : Plan) (d : Host.Dimensions)
    (words : Native.WordMemory) (j : TileIndex [d.B*d.H,d.N,NativeIO.paddedWidth d]) :
    NativeReference.forward ops plan d (forwardValues d words) ⟨0,by change 0<1; decide⟩ j =
      Native.forwardReference ops plan d words "output" (NativeIO.paddedIndex d j) := by
  obtain ⟨final,ran,values,_⟩ := NativeResults.forward ops plan d (forwardValues d words)
    (typed words) (forward_binding ops plan d words)
  have actual := grid_typed ops plan .forward d (grid .forward d) words
  have same : final = typed (Native.forwardReference ops plan d words) :=
    Option.some.inj (ran.symm.trans actual)
  subst final
  have hv := values ⟨0,by change 0<1; decide⟩ j True.intro
  have result := congrArg (Cell.read .real) hv
  exact result.symm

theorem backward_reference (ops : Instructions) (plan : Plan) (d : Host.Dimensions)
    (words : Native.WordMemory) (o : Fin (backwardChannels d).length)
    (j : TileIndex (backwardChannels d)[o].shape) :
    Cell.mk (backwardChannels d)[o].dtype
      (NativeReference.backward ops plan d (backwardValues d words) o j) =
      .mk .real (Native.backwardReference ops plan d words
        (backwardChannels d)[o].buf (backwardIndex d o j)) := by
  obtain ⟨final,ran,values,_⟩ := NativeResults.backward ops plan d (backwardValues d words)
    (typed words) (backward_binding ops plan d words)
  have actual : runMaterializedSchedule ((NativeIO.backward ops plan d).schedule (backwardValues d words))
      (typed words) = some (typed (Native.backwardReference ops plan d words)) := by
    change runMaterializedSchedule (NativeIO.schedule ops plan .intra d ++
      NativeIO.schedule ops plan .inter d) (typed words) = _
    simp only [runMaterializedSchedule_append,NativeIO.schedule,grid_typed,Option.bind_some]
    rfl
  have same : final = typed (Native.backwardReference ops plan d words) :=
    Option.some.inj (ran.symm.trans actual)
  subst final
  have hv := values o j True.intro
  fin_cases o <;> exact hv.symm

/-- Actual Tensor bytes enter the public IO through the unchanged source
pointer reader. Host payload/stride/negative-bit laws remain in native_reads. -/
theorem forward_physical_binding (ops : Instructions) (plan : Plan)
    (r : Request .forward) (heap : ByteMemory) :
    (NativeIO.forward ops plan r.dimensions).InputCells
      (forwardValues r.dimensions (inputWords r heap)) (typed (inputWords r heap)) :=
  forward_binding ops plan r.dimensions (inputWords r heap)

theorem backward_physical_binding (ops : Instructions) (plan : Plan)
    (r : Request .backward) (heap : ByteMemory) :
    (NativeIO.backward ops plan r.dimensions).InputCells
      (backwardValues r.dimensions (inputWords r heap)) (typed (inputWords r heap)) :=
  backward_binding ops plan r.dimensions (inputWords r heap)

#axiomsClean forward_reference
#axiomsClean backward_reference
#axiomsClean forward_physical_binding
#axiomsClean backward_physical_binding
end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly.NativePhysicalIO

