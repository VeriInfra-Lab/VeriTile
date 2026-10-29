import bench.repairs.tritonbench_g.matmul_dequantize.Proofs.TensorBinding
import bench.repairs.tritonbench_g.matmul_dequantize.Whole001.Proofs.TorchMM
namespace VeriTile.Bench.Repairs.MatmulDequantizeWhole001.RealWhole
open VeriTile.Triton FP.Structural Host Checked
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section
set_option maxHeartbeats 4000000
abbrev Heap := RegionName → Nat → ℝ

/-- Ideal real values reside in the same actual named allocations. Packed
integers are always decoded from the original caller bytes. -/
def cell (raw : ByteBridge.Memory) (heap : Heap) (t : Host.Tensor) (i : Nat) : Cell ℝ :=
  match t.dtype with
  | .int32 => .mk .int (ByteBridge.raw32 raw t i).toInt
  | .fp16 => .mk .fp16 (heap t.region (t.storageOffset+i))
  | .fp32 => .mk .fp32 (heap t.region (t.storageOffset+i))
def entry (a : Arguments) (raw : ByteBridge.Memory) (heap : Heap) : State ℝ where
  mem := fun r i =>
    if r=a.q.region then cell raw heap a.q i
    else if r=a.z.region then cell raw heap a.z i
    else if r=a.a.region then cell raw heap a.a i
    else if r=a.s.region then cell raw heap a.s i
    else if r=a.c.region then cell raw heap a.c i
    else if r=a.d.region then cell raw heap a.d i else .mk .fp32 (heap r i)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 0
  undef := fun d _ _ => defaultValue d

theorem packed (a : Arguments) (route : Route) (h : Admissible a route) (raw : ByteBridge.Memory) (heap : Heap) :
    Packed a (entry a raw heap) := by
  have apart := h.separate
  simp only [tensors,List.pairwise_cons,List.mem_cons,List.mem_singleton,forall_eq_or_imp] at apart
  have zq : a.z.region≠a.q.region := by tauto
  constructor
  · intro i hi; exact ⟨ByteBridge.raw32 raw a.q i,by simp [entry,cell,h.packedDtype.1]⟩
  · intro i hi; exact ⟨ByteBridge.raw32 raw a.z i,by simp [entry,cell,h.packedDtype.2,zq]⟩

structure Result where
  state : State ℝ
  output : TensorBinding.Tensor
  returned : Option TensorBinding.Tensor
  allocated : Bool
  scratchReleased : Bool

def observe (a : Arguments) (route : Route) (t : TensorBinding.Tensors a route) (state : State ℝ) : Result :=
  ⟨state,t.get (TensorBinding.output a route).region,TensorBinding.returned a route t,TensorBinding.allocated a route,decide (route=.s1)⟩
def invoke (a : Arguments) (route : Route) (t : TensorBinding.Tensors a route) (action : Option (State ℝ)) :
    Except Error (Option Result) := do
  checkInputs a route
  pure (action.map (observe a route t))

def floatRead (half : Bool) (c : Cell ℝ) : ℝ := if half then c.read .fp16 else c.read .fp32
/-- Read the actual output Tensor's physical coordinate through the ordinary
view-origin binding. NativeWhole separately proves the exact byte stores. -/
def value (out : Host.Tensor) (result : Result) (half : Bool) (r n : Nat) : ℝ :=
  floatRead half (result.state.mem result.output.storageName
    (result.output.physicalIndex (r*out.columns+n)-result.output.storageOffset))

def Guarantee (a : Arguments) (route : Route) (cfg : FirstConfig) (t : TensorBinding.Tensors a route)
    (g : Geometry.Layout) (half : Bool) (initial : State ℝ) (reference : Geometry.Cell g → ℝ) (result : Result) : Prop :=
  result.output=t.get (TensorBinding.output a route).region ∧
  result.returned=TensorBinding.returned a route t ∧ result.allocated=TensorBinding.allocated a route ∧
  result.scratchReleased=decide (route=.s1) ∧
  TensorBinding.Binding (TensorBinding.output a route) result.output ∧
  ABI.SourceExact a route cfg ∧
  (∀ x : Geometry.Cell g,value (TensorBinding.output a route) result half x.1.val x.2.val=reference x) ∧
  (∀ r o,(route=.s1 → r≠a.d.region) →
    (r≠(TensorBinding.output a route).region ∨ ∀ x : Geometry.Cell g,o≠Geometry.address g x) →
    result.state.mem r o=initial.mem r o)

theorem complete (a : Arguments) (route : Route) (cfg : FirstConfig) (t : TensorBinding.Tensors a route)
    (h : InputContract a route) (candidate : cfg∈candidates route)
    (g : Geometry.Layout) (half : Bool) (initial : State ℝ) (reference : Geometry.Cell g → ℝ)
    (action : Option (State ℝ)) (state : State ℝ) (run : action=some state)
    (shape : (TensorBinding.output a route).rows=g.rows ∧ (TensorBinding.output a route).columns=g.columns)
    (strides : (TensorBinding.output a route).rowStride=g.rowStride ∧ (TensorBinding.output a route).columnStride=g.columnStride)
    (values : ∀ x : Geometry.Cell g,floatRead half (state.mem (TensorBinding.output a route).region (Geometry.address g x))=reference x)
    (frame : ∀ r o,(route=.s1 → r≠a.d.region) →
      (r≠(TensorBinding.output a route).region ∨ ∀ x : Geometry.Cell g,o≠Geometry.address g x) → state.mem r o=initial.mem r o) :
    ∃ result,invoke a route t action=.ok (some result) ∧ Guarantee a route cfg t g half initial reference result := by
  have binding := TensorBinding.output_bound a route t
  refine ⟨observe a route t state,?_,rfl,rfl,rfl,rfl,binding,
    ABI.source_exact a route cfg (indices a route cfg candidate h),?_,frame⟩
  · simp [invoke,(checkInputs_iff a route).mpr h,run]
  · intro x
    have coord := binding.coordinates x.1.val x.2.val (by simpa only [shape.1] using x.1.isLt) (by simpa only [shape.2] using x.2.isLt)
    simp only [value,observe,binding.name,binding.offset,coord,strides.1,strides.2]
    rw [show (TensorBinding.output a route).storageOffset+x.1.val*g.rowStride+x.2.val*g.columnStride-
      (TensorBinding.output a route).storageOffset=Geometry.address g x by unfold Geometry.address;omega]
    exact values x

def gptqAction (a : Arguments) (cfg : FirstConfig) (h : InputContract a .gptq) (raw : ByteBridge.Memory) (heap : Heap) :=
  let ha := admitted a .gptq h
  Grid.run (Correct.firstProgram a cfg ha) (List.range (Geometry.count (FirstGrid.geometry (first a cfg)))) (entry a raw heap)

def gptqReference (a : Arguments) (cfg : FirstConfig) (raw : ByteBridge.Memory) (heap : Heap)
    (x : Geometry.Cell (FirstGrid.geometry (first a cfg))) : ℝ := Real.rawMatmul (RealFirst.inputs (first a cfg) (entry a raw heap)) a.K a.group x.1.val x.2.val

theorem gptq (a : Arguments) (cfg : FirstConfig) (t : TensorBinding.Tensors a .gptq) (h : InputContract a .gptq)
    (raw : ByteBridge.Memory) (heap : Heap) :
    ∃ result,invoke a .gptq t (gptqAction a cfg h raw heap)=.ok (some result) ∧
      Guarantee a .gptq cfg t (FirstGrid.geometry (first a cfg)) Bool.true (entry a raw heap) (gptqReference a cfg raw heap) result := by
  have ha := admitted a .gptq h
  have candidate : cfg∈candidates .gptq := by simp [candidates,first_exhaustive]
  obtain ⟨_,state,run,values,frame⟩ := Correct.gptq a cfg ha (indices a .gptq cfg candidate h)
    (entry a raw heap) (packed a .gptq ha raw heap)
    (List.range (Geometry.count (FirstGrid.geometry (first a cfg)))) (List.Perm.refl _)
  apply complete a .gptq cfg t h candidate (FirstGrid.geometry (first a cfg)) Bool.true (entry a raw heap)
    (gptqReference a cfg raw heap) (gptqAction a cfg h raw heap) state run
  · exact ⟨ha.cv.1,ha.cv.2.1⟩
  · exact ⟨rfl,rfl⟩
  · intro x
    have v := values x
    simp only [TensorBinding.output,reduceCtorEq,false_or,or_true,true_or,ite_false,ite_true]
    rw [v]
    rfl
  · intro r o _ outside
    exact frame r o outside

def s2Action (a : Arguments) (h : InputContract a .s2) (raw : ByteBridge.Memory) (heap : Heap) :=
  let ha := admitted a .s2 h
  Grid.run (Correct.secondProgram a ha) (List.range (Geometry.count (SecondGrid.geometry (second a)))) (entry a raw heap)

def s2Reference (a : Arguments) (raw : ByteBridge.Memory) (heap : Heap)
    (x : Geometry.Cell (SecondGrid.geometry (second a))) : ℝ := Real.rawMatmul (RealSecond.inputs (second a) (entry a raw heap)) a.K a.group x.1.val x.2.val

theorem s2 (a : Arguments) (t : TensorBinding.Tensors a .s2) (h : InputContract a .s2)
    (raw : ByteBridge.Memory) (heap : Heap) :
    ∃ result,invoke a .s2 t (s2Action a h raw heap)=.ok (some result) ∧
      Guarantee a .s2 .m64n256 t (SecondGrid.geometry (second a)) Bool.true (entry a raw heap) (s2Reference a raw heap) result := by
  have ha := admitted a .s2 h
  have candidate : FirstConfig.m64n256∈candidates .s2 := by simp [candidates]
  obtain ⟨_,state,run,values,frame⟩ := Correct.s2 a ha (indices a .s2 .m64n256 candidate h)
    (entry a raw heap) (packed a .s2 ha raw heap)
    (List.range (Geometry.count (SecondGrid.geometry (second a)))) (List.Perm.refl _)
  apply complete a .s2 .m64n256 t h candidate (SecondGrid.geometry (second a)) Bool.true (entry a raw heap)
    (s2Reference a raw heap) (s2Action a h raw heap) state run
  · exact ⟨ha.cv.1,ha.cv.2.1⟩
  · exact ⟨rfl,rfl⟩
  · intro x
    have v := values x
    simp only [TensorBinding.output,reduceCtorEq,false_or,or_true,true_or,ite_false,ite_true]
    rw [v]
    rfl
  · intro r o _ outside
    exact frame r o outside

def dequant32Action (a : Arguments) (h : InputContract a .dequant32) (raw : ByteBridge.Memory) (heap : Heap) :=
  let ha := admitted a .dequant32 h
  Grid.run (Correct.dequant32Program a ha) (List.range (Geometry.count (DequantGrid32.geometry (Host.dequant32 a)))) (entry a raw heap)

def dequant32Reference (a : Arguments) (raw : ByteBridge.Memory) (heap : Heap)
    (x : Geometry.Cell (DequantGrid32.geometry (Host.dequant32 a))) : ℝ := Real.rawDequant (RealDequant.D32.inputs (Host.dequant32 a) (entry a raw heap)) a.group x.1.val x.2.val

theorem dequant32 (a : Arguments) (t : TensorBinding.Tensors a .dequant32) (h : InputContract a .dequant32)
    (raw : ByteBridge.Memory) (heap : Heap) :
    ∃ result,invoke a .dequant32 t (dequant32Action a h raw heap)=.ok (some result) ∧
      Guarantee a .dequant32 .m64n256 t (DequantGrid32.geometry (Host.dequant32 a)) Bool.false (entry a raw heap) (dequant32Reference a raw heap) result := by
  have ha := admitted a .dequant32 h
  have candidate : FirstConfig.m64n256∈candidates .dequant32 := by simp [candidates]
  obtain ⟨_,state,run,values,frame⟩ := Correct.dequant32 a ha (indices a .dequant32 .m64n256 candidate h) rfl
    (entry a raw heap) (packed a .dequant32 ha raw heap)
    (List.range (Geometry.count (DequantGrid32.geometry (Host.dequant32 a)))) (List.Perm.refl _)
  apply complete a .dequant32 .m64n256 t h candidate (DequantGrid32.geometry (Host.dequant32 a)) Bool.false (entry a raw heap)
    (dequant32Reference a raw heap) (dequant32Action a h raw heap) state run
  · exact ⟨ha.dv.1,ha.dv.2.1⟩
  · exact ⟨rfl,rfl⟩
  · intro x
    have v := values x
    simp only [TensorBinding.output,reduceCtorEq,false_or,or_true,true_or,ite_false,ite_true]
    rw [v]
    rfl
  · intro r o _ outside
    exact frame r o outside

def dequant16Action (a : Arguments) (h : InputContract a .dequant16) (raw : ByteBridge.Memory) (heap : Heap) :=
  let ha := admitted a .dequant16 h
  Grid.run (Correct.dequant16Program a ha) (List.range (Geometry.count (DequantGrid16.geometry (Host.dequant16 a)))) (entry a raw heap)

def dequant16Reference (a : Arguments) (raw : ByteBridge.Memory) (heap : Heap)
    (x : Geometry.Cell (DequantGrid16.geometry (Host.dequant16 a))) : ℝ := Real.rawDequant (RealDequant.D16.inputs (Host.dequant16 a) (entry a raw heap)) a.group x.1.val x.2.val

theorem dequant16 (a : Arguments) (t : TensorBinding.Tensors a .dequant16) (h : InputContract a .dequant16)
    (raw : ByteBridge.Memory) (heap : Heap) :
    ∃ result,invoke a .dequant16 t (dequant16Action a h raw heap)=.ok (some result) ∧
      Guarantee a .dequant16 .m64n256 t (DequantGrid16.geometry (Host.dequant16 a)) Bool.true (entry a raw heap) (dequant16Reference a raw heap) result := by
  have ha := admitted a .dequant16 h
  have candidate : FirstConfig.m64n256∈candidates .dequant16 := by simp [candidates]
  obtain ⟨_,state,run,values,frame⟩ := Correct.dequant16 a ha (indices a .dequant16 .m64n256 candidate h) (Or.inl rfl)
    (entry a raw heap) (packed a .dequant16 ha raw heap)
    (List.range (Geometry.count (DequantGrid16.geometry (Host.dequant16 a)))) (List.Perm.refl _)
  apply complete a .dequant16 .m64n256 t h candidate (DequantGrid16.geometry (Host.dequant16 a)) Bool.true (entry a raw heap)
    (dequant16Reference a raw heap) (dequant16Action a h raw heap) state run
  · exact ⟨ha.dv.1,ha.dv.2.1⟩
  · exact ⟨rfl,rfl⟩
  · intro x
    have v := values x
    simp only [TensorBinding.output,reduceCtorEq,false_or,or_true,true_or,ite_false,ite_true]
    rw [v]
    rfl
  · intro r o _ outside
    exact frame r o outside

#specNonCircular Guarantee avoiding [invoke,observe,Grid.run]
#axiomsClean gptq
#axiomsClean s2
#axiomsClean dequant32
#axiomsClean dequant16
end
end VeriTile.Bench.Repairs.MatmulDequantizeWhole001.RealWhole
