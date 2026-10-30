import bench.repairs.tritonbench_g.f8_conversion_utils.Proofs.SourceNative

namespace VeriTile.Bench.Repairs.F8ConversionUtils.Source
open VeriTile.Triton FP.Structural Bytes
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000

def outputCell {α : Type} (p : Profile) (value : α) : Cell α :=
  .mk p.output.toTileDType (ofFloat p.output value)

/-- The typed starting state is constructed from the actual raw heap. Input
and output codecs have their own physical widths; no Real-byte encoding or
lossless input/output conversion is required. -/
def backedState {α : Type} [Inhabited α] (M : Algebra α) (p : Profile)
    (physical : Word p.inputWidth → Word p.outputWidth) (binding : ScalarBinding M p physical)
    (storage : OutputStorage p binding.storeWord) (original : Heap)
    (input output pid : Nat) : State α where
  mem := fun r i => if r="x" then inputCell p (binding.loadWord
      (read p.inputWidth original (Memory.address input p.inputWidth i)))
    else outputCell p (storage.load (read p.outputWidth original (Memory.address output p.outputWidth i)))
  regs := fun _ _ _ => none
  pids := fun _ => pid
  numPids := fun _ => 0
  undef := fun dtype _ _ => defaultValue dtype

theorem backed_inputs {α : Type} [Inhabited α] (M : Algebra α) (p : Profile)
    (physical : Word p.inputWidth → Word p.outputWidth) (binding : ScalarBinding M p physical)
    (storage : OutputStorage p binding.storeWord) (original : Heap) (input output pid i : Nat) :
    (backedState M p physical binding storage original input output pid).mem "x" i=
      inputCell p (binding.loadWord (read p.inputWidth original (Memory.address input p.inputWidth i))) := by
  simp [backedState]

theorem backed_outputs {α : Type} [Inhabited α] (M : Algebra α) (p : Profile)
    (physical : Word p.inputWidth → Word p.outputWidth) (binding : ScalarBinding M p physical)
    (storage : OutputStorage p binding.storeWord) (original : Heap) (input output pid i : Nat) :
    binding.storeWord (outputValue p ((backedState M p physical binding storage original input output pid).mem "y" i))=
      read p.outputWidth original (Memory.address output p.outputWidth i) := by
  have scalar := storage.roundtrip (read p.outputWidth original (Memory.address output p.outputWidth i))
  cases p <;> simpa [backedState,outputValue,outputCell,Profile.output,ofFloat,toFloat,Cell.read] using scalar

private theorem address_shift (base start width i : Nat) :
    Memory.address (base+start*width) width i=Memory.address base width (start+i) := by
  simp [Memory.address,Nat.add_mul,Nat.add_assoc]

/-- A successful source run and full output-state refinement exist for every
actual finite launch on an actual heap. The only codec roundtrip is the
ordinary same-width output storage representation. -/
theorem program_exists {α : Type} [Inhabited α] (M : Algebra α) (p : Profile)
    (physical : Word p.inputWidth → Word p.outputWidth) (binding : ScalarBinding M p physical)
    (storage : OutputStorage p binding.storeWord)
    (x : Native.Allocation n p.inputWidth) (y : Native.Allocation n p.outputWidth)
    (start count pid : Nat) (small : count<2^31) (launched : pid<NativeIndex.grid count)
    (globalActive : ∀ i,i<count → start+i<n) (original : Heap) :
    let initial := backedState M p physical binding storage original
      (x.base+start*p.inputWidth) (y.base+start*p.outputWidth) pid
    ∃ final,FP.Structural.exec (M.withDefaultPrecision .fp32) (kernel p "y" "x" count 1024) initial=some final ∧
      (∀ i,binding.storeWord (outputValue p (final.mem "y" i))=
        read p.outputWidth (Lowering.program (Bytes.primitives p.inputWidth p.outputWidth) p.twice physical
          (x.base+start*p.inputWidth) (y.base+start*p.outputWidth) count pid original)
          (Memory.address y.base p.outputWidth (start+i))) ∧
      (∀ (r : RegionName) i,r≠("y" : RegionName) → final.mem r i=initial.mem r i) := by
  apply program_refines M p physical binding x y start count pid small launched globalActive original "y" "x" _ rfl
  · intro i
    simpa only [address_shift] using backed_inputs M p physical binding storage original
      (x.base+start*p.inputWidth) (y.base+start*p.outputWidth) pid i
  · intro i
    simpa only [address_shift] using backed_outputs M p physical binding storage original
      (x.base+start*p.inputWidth) (y.base+start*p.outputWidth) pid i

#axiomsClean program_exists
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.F8ConversionUtils.Source
