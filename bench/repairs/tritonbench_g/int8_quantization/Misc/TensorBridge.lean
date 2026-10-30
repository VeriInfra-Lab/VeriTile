import bench.repairs.tritonbench_g.int8_quantization.Misc.NativeCorrect
import bench.repairs.tritonbench_g.int8_quantization.Correct
import bench.repairs.tritonbench_g.int8_quantization.FPEquiv
import bench.repairs.tritonbench_g.int8_quantization.Proofs.TensorBinding
import bench.repairs.tritonbench_g.int8_quantization.Proofs.RealFrame

/-! Complete retained Tensor applications and their raw kernel correspondence. -/
namespace VeriTile.Bench.Repairs.Int8Quantization.TensorBridge
open VeriTile.Triton FP.Structural Revision002 Words Direct
noncomputable section

/-- Explicit correspondence of the retained Tensor wrapper's live-kernel
branch with the same complete raw result used by both primary interfaces.
Empty and zero-channel wrapper branches remain in the full applications. -/
theorem kernel_correspondence {α : Type} [Inhabited α] (c : Request)
    (M : Algebra α) (ops : Model.Primitives α) (entry : State α)
    (nonempty : 0<count c) (positive : 0<channels c.x.metadata) :
    Grid.launch M ops (p c) entry=some (typedRun c M ops entry) ∧
      (∀ w,w∈Schedule.programs 0 (count c) → ∀ i : Lane (p c),Grid.active (p c) w i →
        (typedRun c M ops entry).mem outRegion (Grid.offset (p c) w i)=
          .mk .int (quantized M ops (Grid.parameters (p c) w) (width_positive c.x.metadata c.block)
            (Grid.atProgram entry w) i)) ∧
      (∀ w,w∈Schedule.programs 0 (count c) →
        (typedRun c M ops entry).mem scaleRegion w.index=
          .mk .real (savedScale M ops (Grid.parameters (p c) w) (width_positive c.x.metadata c.block)
            (Grid.atProgram entry w))) ∧
      RawKernel.Frame (p c) entry (typedRun c M ops entry) := by
  obtain ⟨final,run,outputs,scales,frame⟩ := RawKernel.complete M ops (p c)
    (width_positive c.x.metadata c.block) c.contract.blockPositive entry
  have same : final=typedRun c M ops entry :=
    Option.some.inj (run.symm.trans (complete_source c M ops entry nonempty positive))
  subst final
  exact ⟨run,outputs,scales,frame⟩


/-- Host precision is independent of the FP32 kernel operations. These are the
binary64 literal conversion and two production operations, retained as
primitive model boundaries. The decimal 1.44269504 is rounded to binary64
before multiplication; the final constant is then rounded to FP32 at its use. -/
structure HostPrimitives where
  literal64 : ℝ → ℝ
  binary64Power : ℝ → ℝ → ℝ
  binary64Multiply : ℝ → ℝ → ℝ

def hostFactor (host : HostPrimitives) (channels : Nat) : ℝ :=
  host.binary64Multiply
    (host.binary64Power (host.literal64 channels) (host.literal64 (-0.5)))
    (host.literal64 1.44269504)

theorem all_coordinates (c : Request) (batch row channel : Nat)
    (hb : batch < batches c.x.metadata) (hr : row < length c.x.metadata) (hc : channel < channels c.x.metadata) :
    ∃ w∈Schedule.programs 0 (count c),∃ i : Lane (p c),Grid.active (p c) w i ∧
      Grid.offset (p c) w i=batch*length c.x.metadata*channels c.x.metadata+row*channels c.x.metadata+channel :=
  coordinate_covered c batch row channel hb hr hc

/-- Successful complete source launches, independent total mathematical
values for all four tensors, and every byte outside both output allocations. -/
theorem per_block_int8_real_complete (q k : Request)
    (_qRoute : q.query=Bool.true) (_kRoute : k.query=Bool.false)
    (rest : Algebra ℝ) (store32 : ℝ → Word) (storage : Nat → ℝ) (before : Heap)
    (qa : Allocation q before) (ka : Allocation k (RealHost.after q rest store32 storage before qa)) :
    ∃ result, RealHost.pair q k rest store32 storage before qa ka=.ok result ∧
      result.q=qa.output ∧ result.qScale=qa.scales ∧ result.k=ka.output ∧ result.kScale=ka.scales ∧
      RealHost.TotalValues q store32 storage result.q result.qScale result.heap.memory ∧
      RealHost.TotalValues k store32 storage result.k result.kScale result.heap.memory ∧
      Extension before result.heap ∧
      (0<count q → 0<channels q.x.metadata →
        Grid.launch (Math.algebra rest) Math.primitives (p (RealHost.ideal q)) (RealHost.entry q storage)=
          some (typedRun (RealHost.ideal q) (Math.algebra rest) Math.primitives (RealHost.entry q storage))) ∧
      (0<count k → 0<channels k.x.metadata →
        Grid.launch (Math.algebra rest) Math.primitives (p (RealHost.ideal k)) (RealHost.entry k storage)=
          some (typedRun (RealHost.ideal k) (Math.algebra rest) Math.primitives (RealHost.entry k storage))) ∧
      (∀ a, Outside q qa a → Outside k ka a → result.heap.memory a=before.memory a) := by
  obtain ⟨r,success,hq,hsq,hk,hsk,qv,kv,ext⟩ :=
    RealHost.pair_total_correct q k rest store32 storage before qa ka
  refine ⟨r,success,hq,hsq,hk,hsk,qv,kv,ext,?_,?_,?_⟩
  · exact fun hn hc => (kernel_correspondence (RealHost.ideal q) (Math.algebra rest) Math.primitives _ hn hc).1
  · exact fun hn hc => (kernel_correspondence (RealHost.ideal k) (Math.algebra rest) Math.primitives _ hn hc).1
  · have resultEq : r=⟨qa.output,qa.scales,ka.output,ka.scales,
        RealHost.after k rest store32 storage (RealHost.after q rest store32 storage before qa) ka⟩ := by
      simpa [RealHost.pair,(checkTensorInputs_iff _ _).mpr q.contract,
        (checkTensorInputs_iff _ _).mpr k.contract,Except.bind,bind,pure,Except.pure] using success.symm
    subst r
    exact RealHost.pair_frame q k rest store32 storage before qa ka

/-- Whole wrapper equivalence to an independently evaluated software pipeline,
conditional only on context/load primitives and the fourteen reached numerical
ports. Both references consume the caller's original inputs, including aliases. -/
theorem per_block_int8_equiv_if_local (q k : Request)
    (_qRoute : q.query=Bool.true) (_kRoute : k.query=Bool.false)
    (host : HostPrimitives)
    (_hostConstant : 0<count q → 0<channels q.x.metadata →
      q.factor=hostFactor host (channels q.x.metadata))
    (M : Algebra Word) (ops : Model.Primitives Word) (before : Heap)
    (qa : Allocation q before) (ka : Allocation k (after q M ops before qa))
    (qLive : InputLive q before) (kLive : InputLive k before)
    (software : Floating.Numerics Word)
    (qContext : ∀ w,w∈Schedule.programs 0 (count q) → 0 < channels q.x.metadata →
      FPTrace.Context M (Grid.parameters (p q) w) (Grid.atProgram (Bytes.bind (layout q qa) before.memory) w))
    (kContext : ∀ w,w∈Schedule.programs 0 (count k) → 0 < channels k.x.metadata →
      FPTrace.Context M (Grid.parameters (p k) w)
        (Grid.atProgram (Bytes.bind (layout k ka) (after q M ops before qa).memory) w))
    (qLocal : ∀ w,w∈Schedule.programs 0 (count q) → 0 < channels q.x.metadata →
      Floating.Local (FPTrace.constants M (p q)) (FPTrace.numerics M ops) software q.query (logicalInputs q M before.memory w))
    (kLocal : ∀ w,w∈Schedule.programs 0 (count k) → 0 < channels k.x.metadata →
      Floating.Local (FPTrace.constants M (p k)) (FPTrace.numerics M ops) software k.query (logicalInputs k M before.memory w)) :
    ∃ result,pair q k M ops before qa ka=.ok result ∧
      result.q=qa.output ∧ result.qScale=qa.scales ∧ result.k=ka.output ∧ result.kScale=ka.scales ∧
      Values q M ops before.memory (layout q qa) result.q result.qScale result.heap.memory ∧
      Values k M ops (after q M ops before qa).memory (layout k ka) result.k result.kScale result.heap.memory ∧
      NumericalValues q M software before.memory result.q result.qScale result.heap.memory ∧
      NumericalValues k M software before.memory result.k result.kScale result.heap.memory ∧
      Extension before result.heap := by
  obtain ⟨r,success,hq,hsq,hk,hsk,qv,kv,ext⟩:=pair_correct q k M ops before qa ka
  refine ⟨r,success,hq,hsq,hk,hsk,qv,kv,?_,?_,ext⟩
  · exact numerical_values q M ops before before qa r.q r.qScale r.heap.memory qv
      ⟨fun _ h=>h,fun _ _=>rfl⟩ qLive qContext software qLocal
  · exact numerical_values k M ops before (after q M ops before qa) ka r.k r.kScale r.heap.memory kv
      (extension q M ops before qa) kLive kContext software kLocal

/-- Full selected wrapper: successful Q and K source launches, exact returned
objects/shapes/devices, every output byte and scale, and the caller frame.
The independent graph expands each maximum to ordered scalar operations.
There are no numerical-equivalence hypotheses and no hidden input narrowing. -/
theorem per_block_int8_ordered_graph (q k : Request)
    (_qRoute : q.query=Bool.true) (_kRoute : k.query=Bool.false)
    (host : HostPrimitives)
    (_hostConstant : 0<count q → 0<channels q.x.metadata →
      q.factor=hostFactor host (channels q.x.metadata))
    (rest : Algebra Word) (op : NativeGraph.Instructions) (plans : NativeGraph.Plans)
    (before : Heap) (qa : Allocation q before)
    (ka : Allocation k (after q (NativeGraph.algebra rest op plans) (NativeGraph.comparisons op) before qa))
    (qLive : InputLive q before) (kLive : InputLive k before) :
    let M := NativeGraph.algebra rest op plans
    let ops := NativeGraph.comparisons op
    ∃ result, pair q k M ops before qa ka=.ok result ∧
      result.q=qa.output ∧ result.qScale=qa.scales ∧ result.k=ka.output ∧ result.kScale=ka.scales ∧
      Values q M ops before.memory (layout q qa) result.q result.qScale result.heap.memory ∧
      Values k M ops (after q M ops before qa).memory (layout k ka) result.k result.kScale result.heap.memory ∧
      NativeGraph.Values q op plans before.memory result.q result.qScale result.heap.memory ∧
      NativeGraph.Values k op plans before.memory result.k result.kScale result.heap.memory ∧
      Extension before result.heap ∧
      (0<count q → 0<channels q.x.metadata →
        Grid.launch M ops (p q) (Bytes.bind (layout q qa) before.memory)=
          some (typedRun q M ops (Bytes.bind (layout q qa) before.memory))) ∧
      (0<count k → 0<channels k.x.metadata →
        Grid.launch M ops (p k) (Bytes.bind (layout k ka) (after q M ops before qa).memory)=
          some (typedRun k M ops (Bytes.bind (layout k ka) (after q M ops before qa).memory))) ∧
      (∀ a, Outside q qa a → Outside k ka a → result.heap.memory a=before.memory a) := by
  let M := NativeGraph.algebra rest op plans
  let ops := NativeGraph.comparisons op
  obtain ⟨r,success,hq,hsq,hk,hsk,qv,kv,ext⟩ := pair_correct q k M ops before qa ka
  refine ⟨r,success,hq,hsq,hk,hsk,qv,kv,?_,?_,ext,?_,?_,?_⟩
  · exact NativeGraph.values q rest op plans before before qa r.q r.qScale r.heap.memory qv
      ⟨fun _ h => h,fun _ _ => rfl⟩ qLive
  · exact NativeGraph.values k rest op plans before (after q M ops before qa) ka r.k r.kScale r.heap.memory kv
      (extension q M ops before qa) kLive
  · exact fun hn hc => (kernel_correspondence q M ops _ hn hc).1
  · exact fun hn hc => (kernel_correspondence k M ops _ hn hc).1
  · have resultEq : r=⟨qa.output,qa.scales,ka.output,ka.scales,
        after k M ops (after q M ops before qa) ka⟩ := by
      simpa [pair,(checkTensorInputs_iff _ _).mpr q.contract,
        (checkTensorInputs_iff _ _).mpr k.contract,Except.bind,bind,pure,Except.pure] using success.symm
    subst r
    exact pair_frame q k M ops before qa ka


theorem python_checks_iff_preconditions (x : ArgumentMetadata) (block : BlockArgument) :
    checkInputs x block=.ok () ↔ AdmittedArguments x block := checkInputs_iff x block

/-- Arbitrary original Real inputs. Each block whose transformed values exceed
the finite FP32 range returns all-zero bytes and scale; every other block uses
the independent max/127 target. All four tensors and caller bytes are covered. -/
theorem per_block_int8_real_wrapper_total (q k : Request)
    (_qRoute : q.query=Bool.true) (_kRoute : k.query=Bool.false)
    (rest : Algebra ℝ) (store32 : ℝ → Word) (storage : Nat → ℝ) (before : Heap)
    (qa : Allocation q before) (ka : Allocation k (RealHost.after q rest store32 storage before qa)) :
    ∃ result, RealHost.pair q k rest store32 storage before qa ka=.ok result ∧
      result.q=qa.output ∧ result.qScale=qa.scales ∧ result.k=ka.output ∧ result.kScale=ka.scales ∧
      RealHost.TotalValues q store32 storage result.q result.qScale result.heap.memory ∧
      RealHost.TotalValues k store32 storage result.k result.kScale result.heap.memory ∧
      Extension before result.heap :=
  RealHost.pair_total_correct q k rest store32 storage before qa ka

/-- All four exact returned tensors, every value, independent shape/device,
zero-channel scales and caller frame. Every logical coordinate is covered by
`TensorBridge.all_coordinates`; `native_pointers` derives the complete widened addresses. -/
theorem per_block_int8_real_wrapper_correct (q k : Request)
    (_qRoute : q.query=Bool.true) (_kRoute : k.query=Bool.false)
    (rest : Algebra ℝ) (store32 : ℝ → Word) (storage : Nat → ℝ) (before : Heap)
    (qa : Allocation q before) (ka : Allocation k (RealHost.after q rest store32 storage before qa))
    (finiteQ : RealHost.Finite q storage) (finiteK : RealHost.Finite k storage) :
    ∃ result, RealHost.pair q k rest store32 storage before qa ka=.ok result ∧
      result.q=qa.output ∧ result.qScale=qa.scales ∧ result.k=ka.output ∧ result.kScale=ka.scales ∧
      RealHost.Values q store32 storage result.q result.qScale result.heap.memory ∧
      RealHost.Values k store32 storage result.k result.kScale result.heap.memory ∧
      Extension before result.heap :=
  RealHost.pair_correct q k rest store32 storage before qa ka finiteQ finiteK

theorem original_inputs (c : Request) (storage : Nat → ℝ) (i : Nat) (hi : i < c.x.metadata.numel) :
    ((RealHost.entry c storage).mem xRegion i).read .real=RealHost.input c storage i :=
  RealHost.input_pointer c storage i hi

theorem per_block_int8_wrapper_fp32 (q k : Request)
    (_qRoute : q.query=Bool.true) (_kRoute : k.query=Bool.false)
    (host : HostPrimitives)
    (_hostConstant : 0<count q → 0<channels q.x.metadata →
      q.factor=hostFactor host (channels q.x.metadata))
    (M : Algebra Word) (ops : Model.Primitives Word) (before : Heap)
    (qa : Allocation q before) (ka : Allocation k (after q M ops before qa)) :
    ∃ result,pair q k M ops before qa ka=.ok result ∧
      result.q=qa.output ∧ result.qScale=qa.scales ∧ result.k=ka.output ∧ result.kScale=ka.scales ∧
      Values q M ops before.memory (layout q qa) result.q result.qScale result.heap.memory ∧
      Values k M ops (after q M ops before qa).memory (layout k ka) result.k result.kScale result.heap.memory ∧
      Extension before result.heap := pair_correct q k M ops before qa ka

/-- Directly observed input Tensor values are the typed kernel's input words. -/
theorem original_tensor_words (c : Request) (before : Heap) (out : Allocation c before)
    (i : Nat) (valid : i<c.x.metadata.numel) :
    fp32Word (((Bytes.bind (layout c out) before.memory).mem xRegion i).read .real)=
      VeriTile.Torch.FP32.logicalValue (tensorRead before.memory) c.x c.laws i :=
  bound_load_tensor_logical c before out i valid

#axiomsClean kernel_correspondence
#axiomsClean all_coordinates
#axiomsClean per_block_int8_real_complete
#axiomsClean per_block_int8_equiv_if_local
#axiomsClean per_block_int8_ordered_graph
#axiomsClean python_checks_iff_preconditions
#axiomsClean per_block_int8_real_wrapper_total
#axiomsClean per_block_int8_real_wrapper_correct
#axiomsClean original_inputs
#axiomsClean per_block_int8_wrapper_fp32
#axiomsClean original_tensor_words
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.Int8Quantization.TensorBridge
