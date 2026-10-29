import bench.repairs.tritonbench_g.cross_entropy1.Proofs.JointExecution

namespace VeriTile.Bench.Repairs.CrossEntropy1.Physical
open VeriTile.Triton Source Host Floating
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section

/-- Ordinary outside-write memory semantics for each consecutive local step
of the same image. No task output or arithmetic equality is assumed. -/
structure ImageMemory {σ : Type} (loaded : LoadedImage σ) (observe : σ → Nat → Nat → Byte) : Prop where
  frame : ∀ location source d image, loaded location source d image → ∀ tick, tick < image.length →
    ∀ device address, (device,address)∉image.writes tick →
      observe (image.state (tick+1)) device address=observe (image.state tick) device address

theorem ImageMemory.interval {σ : Type} {loaded observe} (api : ImageMemory (σ:=σ) loaded observe)
    {location source d image} (object : loaded location source d image)
    (first last : Nat) (order : first≤last) (bound : last ≤ image.length) (device address : Nat)
    (outside : ∀ tick, first≤tick → tick < last → (device,address)∉image.writes tick) :
    observe (image.state last) device address=observe (image.state first) device address := by
  have steps : ∀ count, first+count≤last →
      observe (image.state (first+count)) device address=observe (image.state first) device address := by
    intro count
    induction count with
    | zero => intro _; simp
    | succ count ih =>
      intro finish
      have lt : first+count < last := by omega
      exact (api.frame _ _ _ _ object (first+count) (by omega) device address (outside _ (by omega) lt)).trans (ih (by omega))
  simpa only [Nat.add_sub_of_le order] using steps (last-first) (by omega)

theorem ImageMemory.suffix {σ : Type} {loaded observe} (api : ImageMemory (σ:=σ) loaded observe)
    {location source d image} (object : loaded location source d image)
    (first : Nat) (bound : first ≤ image.length) (device address : Nat)
    (outside : ∀ tick, first≤tick → tick < image.length → (device,address)∉image.writes tick) :
    observe (image.state image.length) device address=observe (image.state first) device address := by
  have steps : ∀ count, first+count ≤ image.length →
      observe (image.state (first+count)) device address=observe (image.state first) device address := by
    intro count
    induction count with
    | zero => intro _; simp
    | succ count ih =>
      intro last
      have lt : first+count < image.length := by omega
      exact (api.frame _ _ _ _ object (first+count) lt device address (outside _ (by omega) lt)).trans (ih (by omega))
  simpa only [Nat.add_sub_of_le bound] using steps (image.length-first) (by omega)

/-- Required return path passes through the actual call completion. The
old arbitrary `LinkedReturn.tail` is not used by the installed theorem below. -/
structure CallCompletion {σ : Type} {source abi selection loaded}
    (objects : ObjectSelection (σ:=σ) source abi selection loaded)
    {layout : Layout σ} {observe events readWord expression state tensor index}
    (binding : LinkedReturn (selection:=selection) layout observe events objects.registry readWord expression state tensor index) where
  boundaryStable : layout.buffer binding.site.call.after tensor.region=layout.buffer binding.stored tensor.region
  returnStable : layout.buffer state tensor.region=layout.buffer binding.site.call.after tensor.region
  localOutside : ∀ tick,
    (objects.image binding.site.call).tick binding.site.coordinate binding.site.instruction+1≤tick →
    tick < (objects.image binding.site.call).length → ∀ byte : Fin 4,
    ((layout.buffer binding.stored tensor.region).device,
      (layout.buffer binding.stored tensor.region).base+4*tensor.index index+byte.val)∉(objects.image binding.site.call).writes tick
  writes : List ByteAddress
  tail : ByteTrace events binding.site.call.after state writes
  outside : ∀ byte : Fin 4,
    ((layout.buffer binding.site.call.after tensor.region).device,
      (layout.buffer binding.site.call.after tensor.region).base+4*tensor.index index+byte.val)∉writes

theorem CallCompletion.retained {σ : Type} {source abi selection loaded}
    {objects : ObjectSelection (σ:=σ) source abi selection loaded}
    {layout : Layout σ} {observe events readWord expression state tensor index}
    {binding : LinkedReturn (selection:=selection) layout observe events objects.registry readWord expression state tensor index}
    (completion : CallCompletion objects binding) (localMemory : ImageMemory loaded observe)
    (memory : ByteContracts events observe selection) (byte : Fin 4) :
    tensorByte layout observe state tensor index byte=tensorByte layout observe binding.stored tensor index byte := by
  let object := objects.exact binding.site.call binding.active.1
  have storeState := (object.states.store binding.site.coordinate binding.site.instruction
    (binding.code.nativeGraph.output objects.registry) binding.stored tensor index binding.active.2 binding.store).1
  have bound := object.states.bound binding.site.coordinate binding.site.instruction binding.active.2
  have localFrame := localMemory.suffix object.object
    ((objects.image binding.site.call).tick binding.site.coordinate binding.site.instruction+1) (by omega)
    (layout.buffer binding.stored tensor.region).device
    ((layout.buffer binding.stored tensor.region).base+4*tensor.index index+byte.val)
    (fun tick lo hi => completion.localOutside tick lo hi byte)
  rw [object.states.final,←storeState] at localFrame
  have hostFrame := completion.tail.frame memory _ _ (completion.outside byte)
  unfold tensorByte
  rw [completion.boundaryStable] at hostFrame
  rw [completion.returnStable,completion.boundaryStable]
  exact hostFrame.trans localFrame

theorem CallCompletion.bytes {σ : Type} {source abi selection loaded}
    {objects : ObjectSelection (σ:=σ) source abi selection loaded}
    {layout : Layout σ} {observe events readWord expression state tensor index trace native reference}
    {binding : LinkedReturn (selection:=selection) layout observe events objects.registry readWord expression state tensor index}
    (completion : CallCompletion objects binding) (instructions : InstructionExecution objects.registry trace)
    (api : PrimitiveContracts trace native readWord (tensorByte layout observe))
    (localMemory : ImageMemory loaded observe) (memory : ByteContracts events observe selection)
    (ports : Obligations native reference) (laws : Peepholes native) (constants : ConstantPorts native) (byte : Fin 4) :
    tensorByte layout observe state tensor index byte=
      byteOf (if tensor.isNeg then flipSign (binding.plan.reference reference) else binding.plan.reference reference) byte := by
  rw [completion.retained localMemory memory byte]
  rw [api.store _ _ _ _ _ (instructions.store _ _ _ _ _ binding.active binding.store) byte]
  exact congrArg (fun x => byteOf (if tensor.isNeg then flipSign x else x) byte)
    ((binding.code.executed instructions).reference_value api ports laws constants)

#axiomsClean ImageMemory.suffix
#axiomsClean CallCompletion.retained
#axiomsClean CallCompletion.bytes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Physical
