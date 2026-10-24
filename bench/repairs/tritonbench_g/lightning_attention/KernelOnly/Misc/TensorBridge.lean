import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.Misc.NativeCorrect
import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.PhysicalIOBinding
import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.Proofs.NativePhysicalIO
import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.Contract
import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.Stores
import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.TensorValues
import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.PhaseReal
import bench.repairs.tritonbench_g.lightning_attention.Proofs.TargetLoss
import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.InputsPhysical
import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.NativePhases

/-! Complete Tensor applications, physical byte/readback frames and API bounds. -/
namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly.TensorBridge
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.LightningAttention.Native (Instructions Plan LocalConformance)
noncomputable section
set_option maxHeartbeats 6000000

def realExecute : {phase : Phase} → Request phase → (Nat → ℝ) → Option (Memory ℝ)
  | .forward,r,read => executeForward Real.algebra Real.dotPrimitive r.dimensions (initialReal r read)
  | .backward,r,read => executeBackward Real.algebra Real.dotPrimitive r.dimensions (initialReal r read)

/-- Independent causal output and VJP over actual logical Tensor inputs. -/
def TensorPost : {phase : Phase} → Request phase → (Nat → ℝ) → (Nat → ℝ) → Prop
  | .forward,r,before,after =>
      ∀ (head : Fin (r.dimensions.B*r.dimensions.H)) (t : Fin r.dimensions.N) (c : Fin r.dimensions.E),
        logical after (r.result 0) (head.val*r.dimensions.N*r.dimensions.E+t.val*r.dimensions.E+c.val)=
          Target.output (tensorQ r before head) (tensorK r before head) (tensorV r before head) t c
  | .backward,r,before,after =>
      (∀ (head : Fin (r.dimensions.B*r.dimensions.H)) (t : Fin r.dimensions.N) (a : Fin r.dimensions.D),
        logical after (r.result 0) (head.val*r.dimensions.N*r.dimensions.D+t.val*r.dimensions.D+a.val)=
          Target.dq (tensorK r before head) (tensorV r before head) (tensorG r before head) t a) ∧
      (∀ (head : Fin (r.dimensions.B*r.dimensions.H)) (t : Fin r.dimensions.N) (a : Fin r.dimensions.D),
        logical after (r.result 1) (head.val*r.dimensions.N*r.dimensions.D+t.val*r.dimensions.D+a.val)=
          Target.dk (tensorQ r before head) (tensorV r before head) (tensorG r before head) t a) ∧
      (∀ (head : Fin (r.dimensions.B*r.dimensions.H)) (t : Fin r.dimensions.N) (c : Fin r.dimensions.E),
        logical after (r.result 2) (head.val*r.dimensions.N*r.dimensions.E+t.val*r.dimensions.E+c.val)=
          Target.dv (tensorQ r before head) (tensorK r before head) (tensorG r before head) t c)

def Derivatives : {phase : Phase} → Request phase → (Nat → ℝ) → (Nat → ℝ) → Prop
  | .forward,_,_,_ => True
  | .backward,r,before,after => ∀ head : Fin (r.dimensions.B*r.dimensions.H),
      (∀ t a, HasDerivAt (fun ε => Target.loss (Target.bump (tensorQ r before head) t a ε)
        (tensorK r before head) (tensorV r before head) (tensorG r before head))
        (logical after (r.result 0) (head.val*r.dimensions.N*r.dimensions.D+t.val*r.dimensions.D+a.val)) 0) ∧
      (∀ t a, HasDerivAt (fun ε => Target.loss (tensorQ r before head)
        (Target.bump (tensorK r before head) t a ε) (tensorV r before head) (tensorG r before head))
        (logical after (r.result 1) (head.val*r.dimensions.N*r.dimensions.D+t.val*r.dimensions.D+a.val)) 0) ∧
      (∀ t c, HasDerivAt (fun ε => Target.loss (tensorQ r before head) (tensorK r before head)
        (Target.bump (tensorV r before head) t c ε) (tensorG r before head))
        (logical after (r.result 2) (head.val*r.dimensions.N*r.dimensions.E+t.val*r.dimensions.E+c.val)) 0)

theorem derivatives_of_post {phase} (r : Request phase) (before after : Nat → ℝ)
    (post : TensorPost r before after) : Derivatives r before after := by
  cases phase with
  | forward => trivial
  | backward =>
    intro head
    have deriv := Target.loss_coordinate_derivatives (tensorQ r before head) (tensorK r before head)
      (tensorV r before head) (tensorG r before head)
    exact ⟨fun t a => by rw [post.1 head t a]; exact deriv.1 t a,
      fun t a => by rw [post.2.1 head t a]; exact deriv.2.1 t a,
      fun t c => by rw [post.2.2 head t c]; exact deriv.2.2 t c⟩

theorem forward_tensor_post (r : Request .forward) (h : Contract r) (read : Nat → ℝ) :
    TensorPost r read (mathStores r (Real.afterForward r.dimensions (initialReal r read)) read) := by
  obtain ⟨_,final,ran,outputs,_⟩ := PhysicalIO.forward_application r h read
  obtain ⟨source,sourceRun,mem⟩ := IOExecution.forward r.dimensions
    (ChunkDeltaFwd.memoryState (initialReal r read))
  have same : final = source := Option.some.inj (ran.symm.trans sourceRun)
  subst source
  intro head t c
  let i : Fin (count r 0) := ⟨_,logical_bound r.dimensions r.dimensions.E head t c⟩
  change logical _ (r.result 0) i.val=_
  rw [mathStores_logical r h _ read 0 i]
  have value := outputs (0 : Fin 1) (head,t,c,()) trivial
  change mapCell id (final.mem "output" i.val) = Cell.mk .real
    (Target.output (tensorQ r read head) (tensorK r read head) (tensorV r read head) t c) at value
  rw [mem] at value
  change Cell.read .real (Real.afterForward r.dimensions (initialReal r read) "output" i.val)=_
  have mapped (cell : Cell ℝ) : mapCell id cell = cell := by
    cases cell with | mk dtype payload => cases dtype <;> rfl
  rw [mapped] at value
  exact congrArg (Cell.read .real) value

theorem backward_tensor_post (r : Request .backward) (h : Contract r) (read : Nat → ℝ)
    (prep : Prepared r read) :
    TensorPost r read (mathStores r (Real.backwardMemory r.dimensions (initialReal r read)) read) := by
  obtain ⟨_,final,ran,outputs,_⟩ := PhysicalIO.backward_application r h read prep
  obtain ⟨source,sourceRun,mem⟩ := IOExecution.backward r.dimensions
    (ChunkDeltaFwd.memoryState (initialReal r read))
  have same : final = source := Option.some.inj (ran.symm.trans sourceRun)
  subst source
  have mapped (cell : Cell ℝ) : mapCell id cell = cell := by
    cases cell with | mk dtype payload => cases dtype <;> rfl
  refine ⟨?_,?_,?_⟩
  · intro head t a
    let i : Fin (count r 0) := ⟨_,logical_bound r.dimensions r.dimensions.D head t a⟩
    change logical _ (r.result 0) i.val=_
    rw [mathStores_logical r h _ read 0 i]
    have value := outputs (0 : Fin 3) (head,t,a,()) trivial
    change mapCell id (final.mem "queryGradient" i.val) = Cell.mk .real
      (Target.dq (tensorK r read head) (tensorV r read head) (tensorG r read head) t a) at value
    rw [mem,mapped] at value
    change Cell.read .real (Real.backwardMemory r.dimensions (initialReal r read) "queryGradient" i.val)=_
    exact congrArg (Cell.read .real) value
  · intro head t a
    let i : Fin (count r 1) := ⟨_,logical_bound r.dimensions r.dimensions.D head t a⟩
    change logical _ (r.result 1) i.val=_
    rw [mathStores_logical r h _ read 1 i]
    have value := outputs (1 : Fin 3) (head,t,a,()) trivial
    change mapCell id (final.mem "keyGradient" i.val) = Cell.mk .real
      (Target.dk (tensorQ r read head) (tensorV r read head) (tensorG r read head) t a) at value
    rw [mem,mapped] at value
    change Cell.read .real (Real.backwardMemory r.dimensions (initialReal r read) "keyGradient" i.val)=_
    exact congrArg (Cell.read .real) value
  · intro head t c
    let i : Fin (count r 2) := ⟨_,logical_bound r.dimensions r.dimensions.E head t c⟩
    change logical _ (r.result 2) i.val=_
    rw [mathStores_logical r h _ read 2 i]
    have value := outputs (2 : Fin 3) (head,t,c,()) trivial
    change mapCell id (final.mem "valueGradient" i.val) = Cell.mk .real
      (Target.dv (tensorQ r read head) (tensorK r read head) (tensorG r read head) t c) at value
    rw [mem,mapped] at value
    change Cell.read .real (Real.backwardMemory r.dimensions (initialReal r read) "valueGradient" i.val)=_
    exact congrArg (Cell.read .real) value

/-- Full arbitrary-Real source result, true coordinate derivatives, exact
exterior and full backing frame. There is no FP32 representability premise. -/
theorem correct {phase} (r : Request phase) (h : Contract r) (read : Nat → ℝ)
    (prep : Preparation r read) :
    ∃ final,realExecute r read=some final ∧ TensorPost r read (mathStores r final read) ∧
      Derivatives r read (mathStores r final read) ∧
      (∀ a,(¬∃ i : Lane r,address r i=a) → mathStores r final read a=read a) ∧
      ReadFrame r read (mathStores r final read) ∧ AddressSafety r := by
  have finish (final : Memory ℝ) (run : realExecute r read=some final)
      (post : TensorPost r read (mathStores r final read)) :
      ∃ final,realExecute r read=some final ∧ TensorPost r read (mathStores r final read) ∧
        Derivatives r read (mathStores r final read) ∧
        (∀ a,(¬∃ i : Lane r,address r i=a) → mathStores r final read a=read a) ∧
        ReadFrame r read (mathStores r final read) ∧ AddressSafety r :=
    ⟨final,run,post,derivatives_of_post r read _ post,
      writeback_frame r _ read,writeback_inputs r h _ read,addresses r h⟩
  cases phase with
  | forward =>
    exact finish _ (Real.forward_correct r.dimensions h.dimensions (initialReal r read)).1
      (forward_tensor_post r h read)
  | backward =>
    exact finish _ (Real.backward_correct r.dimensions h.dimensions (initialReal r read)).1
      (backward_tensor_post r h read prep)

def nativeExecute (ops : Instructions) (plan : Plan) : {phase : Phase} → Request phase → ByteMemory → Option Native.WordMemory
  | .forward,r,heap => Native.forwardExecute ops plan r.dimensions (inputWords r heap)
  | .backward,r,heap => Native.backwardExecute ops plan r.dimensions (inputWords r heap)

/-- Independent fixed-K8 reference consumes the same current physical input
words. Intra results feed inter; no successful run or supplied result is used. -/
def nativeReference (ops : Instructions) (plan : Plan) : {phase : Phase} → Request phase → ByteMemory → Native.WordMemory
  | .forward,r,heap => Native.forwardReference ops plan r.dimensions (inputWords r heap)
  | .backward,r,heap => Native.backwardReference ops plan r.dimensions (inputWords r heap)

theorem native_execution {phase} (r : Request phase) (heap : ByteMemory)
    (actual selected : Instructions) (plan : Plan) (numeric : LocalConformance actual selected) :
    nativeExecute actual plan r heap=some (nativeReference selected plan r heap) := by
  cases phase with
  | forward => exact (Native.forward_native actual selected plan r.dimensions (inputWords r heap) numeric).1
  | backward => exact (Native.backward_native actual selected plan r.dimensions (inputWords r heap) numeric).1

/-- Exact word/byte equality for the selected source graph. The only numerical
premise is scalar FP32 add and bounded fixed-K8 MMA/WGMMA coordinate laws. -/
theorem fp_equiv {phase} (r : Request phase) (h : Contract r) (heap : ByteMemory)
    (prep : Preparation r (load32 heap)) (actual selected : Instructions) (plan : Plan)
    (numeric : LocalConformance actual selected) :
    ∃ final,nativeExecute actual plan r heap=some final ∧
      NativeReads r h heap ∧
      (∀ j (i : Fin (count r j)),
        Torch.FP32.logicalValue (fun a => ⟨load32 (nativeStores r final heap) a⟩)
          (r.result j).tensor (h.result j).1.tensor i.val =
            ⟨nativeReference selected plan r heap (resultRegion j) i.val⟩) ∧
      (∀ i : Lane r,nativeStores r final heap (address r i)=
        encode (nativeReference selected plan r heap (resultRegion i.output) i.element.val) i.byte) ∧
      (∀ a,(¬∃ i : Lane r,address r i=a) → nativeStores r final heap a=heap a) ∧
      ReadFrame r heap (nativeStores r final heap) ∧ AddressSafety r := by
  let final := nativeReference selected plan r heap
  refine ⟨final,native_execution r heap actual selected plan numeric,native_reads r h heap prep,?_,?_,?_,?_,addresses r h⟩
  · intro j i
    exact nativeStores_logical r h final heap j i
  · intro i
    exact writeback_readback r h _ heap i
  · exact writeback_frame r _ heap
  · exact writeback_inputs r h _ heap

#axiomsClean correct
#axiomsClean fp_equiv
#axiomsClean derivatives_of_post
#axiomsClean native_execution
#specNonCircular TensorPost avoiding [realExecute,mathStores,executeForward,executeBackward]
#specNonCircular Derivatives avoiding [realExecute,mathStores,executeForward,executeBackward]
#specNonCircular nativeReference avoiding [nativeExecute,FP.Structural.exec,FP.Structural.run,Native.stageExecute,Native.gridExecute]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly.TensorBridge
