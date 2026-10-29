import bench.repairs.tritonbench_g.rowwise_quantization_triton.Proofs.Contract
import VeriTile.Torch.FP32

/-! Direct Tensor binding and fresh output pair. The retained normalized-input
interface is specialized to identity; no copy service is called. -/
namespace VeriTile.Bench.Repairs.RowwiseQuantizationTriton.Direct
open VeriTile.Triton
noncomputable section
abbrev Tensor := VeriTile.Torch.FP32.Tensor
abbrev TensorLaws := VeriTile.Torch.FP32.TensorLaws
abbrev Allocator := ∀ (dtype : DType) (shape : List Nat) (before : State), Option (Allocation dtype shape before)

/-- torch.empty(..., device=x.device) on each of the two allocation calls. -/
def OnDevice (allocate : Allocator) (device : VeriTile.Torch.Device) : Prop :=
  ∀ dtype shape before result,allocate dtype shape before=some result → result.tensor.device=device

def launch (x : TensorMetadata) (h : InputContract x) (tree : Tree)
    (cover : tree.leaves.Perm (List.range (paddedWidth x))) : Launch :=
  ⟨rows x,columns x,paddedWidth x,h.width.1,padded_covers x,tree,cover⟩

def input (x : Tensor) (laws : TensorLaws x) (before : State) : Input (rows x.metadata) (columns x.metadata) where
  object := x.object
  logical := fun r c => VeriTile.Torch.FP32.logicalValue (load32 before.heap) x laws (r*columns x.metadata+c)

def view (x : Tensor) (laws : TensorLaws x) (h : InputContract x.metadata) :=
  VeriTile.Torch.FP32.dataView x laws h.contiguous

/-- Ordinary input allocation liveness, independent of the kernel or output. -/
def Resident (x : Tensor) (laws : TensorLaws x) (h : InputContract x.metadata) (before : State) : Prop :=
  ∀ a,InRange (view x laws h).base (4*x.metadata.numel) a → before.live a

def binding (x : Tensor) (laws : TensorLaws x) (h : InputContract x.metadata)
    (before : State) (resident : Resident x laws h before) :
    Normalized (rows x.metadata) (columns x.metadata) (input x laws before) before where
  after := before
  base := (view x laws h).base
  extension := ⟨fun _ live => live,fun _ _ => rfl⟩
  resident := by simpa [Resident,numel_eq x.metadata h] using resident
  coordinate := by
    intro r hr c hc
    have hi : r*columns x.metadata+c<x.metadata.numel := by
      rw [numel_eq x.metadata h]
      exact linear_bound _ _ r c hr hc
    exact VeriTile.Torch.FP32.direct_read_eq_logical (load32 before.heap) x laws h.contiguous h.noNeg _ hi

theorem binding_identity (x : Tensor) (laws : TensorLaws x) (h : InputContract x.metadata)
    (before : State) (resident : Resident x laws h before) :
    (binding x laws h before resident).after=before := rfl

theorem actual_pointer (x : Tensor) (laws : TensorLaws x) (h : InputContract x.metadata)
    (positive : 0<x.metadata.numel) : BitVec.ofNat 64 (view x laws h).base=x.dataPtr :=
  VeriTile.Torch.FP32.actual_pointer x laws h.contiguous positive

def prepare (allocate : Allocator) (p : Launch) (inp : Input p.rows p.columns) (before : State)
    (direct : Normalized p.rows p.columns inp before) : Option (Prepared p inp before) := do
  let quantized ← allocate .int8 [p.rows,p.columns] direct.after
  let maxima ← allocate .float16 [p.rows] quantized.after
  return ⟨direct,quantized,maxima⟩

def forward (allocate : Allocator) (ops : Operations Word) (p : Launch) (schedule : Schedule p)
    (inp : Input p.rows p.columns) (before : State) (direct : Normalized p.rows p.columns inp before) :
    Option (RowwiseQuantizationTriton.Tensor × RowwiseQuantizationTriton.Tensor × Heap) := do
  let prepared ← prepare allocate p inp before direct
  return (prepared.quantized.tensor,prepared.maxima.tensor,
    runGrid ops p prepared.layout schedule.order prepared.maxima.after.heap)

theorem returned (allocate : Allocator) (ops : Operations Word) (p : Launch) (schedule : Schedule p)
    (inp : Input p.rows p.columns) (before : State) (direct : Normalized p.rows p.columns inp before)
    (q m : RowwiseQuantizationTriton.Tensor) (after : Heap)
    (run : forward allocate ops p schedule inp before direct=some (q,m,after)) :
    ResultCorrect ops p inp before q m after := by
  unfold forward at run
  cases hp : prepare allocate p inp before direct with
  | none => simp [hp] at run
  | some prepared =>
    simp only [hp] at run
    rcases run with ⟨rfl,rfl,rfl⟩
    obtain ⟨hq,hm⟩ := runGrid_readback ops p prepared.layout prepared.separation schedule prepared.maxima.after.heap
    refine ⟨prepared.quantized.type_eq,prepared.quantized.shape_eq,
      prepared.maxima.type_eq,prepared.maxima.shape_eq,?_,?_,?_⟩
    · intro row hr column hc
      have hmax := rowMaximum_congr ops p _ (inp.logical row) (fun c hcc => prepared.coordinate row c hr hcc)
      have result := hq row hr column hc
      unfold qValue at result
      rw [prepared.coordinate row column hr hc,hmax] at result
      exact result
    · intro row hr byte
      have hmax := rowMaximum_congr ops p _ (inp.logical row) (fun c hc => prepared.coordinate row c hr hc)
      have result := hm row hr byte
      unfold mValue at result
      rw [hmax] at result
      exact result
    · intro address live
      rw [runGrid_frame ops p prepared.layout schedule.order
        (fun r hr => List.mem_range.mp (schedule.complete.mem_iff.mp hr)) _ address
        (prepared.caller_outside address live)]
      exact (prepared.normalized.extension.trans
        (prepared.quantized.extension.trans prepared.maxima.extension)).frame address live

/-- Both public entries use this same guard and identity binding before the
ordinary two allocations and complete kernel execution. -/
def checkedForward (allocate : Allocator) (ops : Operations Word)
    (x : Tensor) (laws : TensorLaws x) (h : InputContract x.metadata)
    (before : State) (resident : Resident x laws h before) (tree : Tree)
    (cover : tree.leaves.Perm (List.range (paddedWidth x.metadata)))
    (schedule : Schedule (launch x.metadata h tree cover)) :
    Except InputError (Option (RowwiseQuantizationTriton.Tensor × RowwiseQuantizationTriton.Tensor × Heap)) := do
  checkTensorInputs x.metadata
  return forward allocate ops (launch x.metadata h tree cover) schedule
    (input x laws before) before (binding x laws h before resident)

theorem checked_correct (allocate : Allocator) (ops : Operations Word)
    (x : Tensor) (laws : TensorLaws x) (h : InputContract x.metadata)
    (before : State) (resident : Resident x laws h before) (tree : Tree)
    (cover : tree.leaves.Perm (List.range (paddedWidth x.metadata)))
    (schedule : Schedule (launch x.metadata h tree cover))
    (q m : RowwiseQuantizationTriton.Tensor) (after : Heap)
    (success : checkedForward allocate ops x laws h before resident tree cover schedule=.ok (some (q,m,after))) :
    ResultCorrect ops (launch x.metadata h tree cover) (input x laws before) before q m after := by
  have accepted := (checkTensorInputs_iff x.metadata).mpr h
  simp only [checkedForward,accepted,Except.bind,bind,pure,Except.pure,Except.ok.injEq] at success
  exact returned allocate ops _ schedule _ before _ q m after success

theorem checked_devices (allocate : Allocator)
    (ops : Operations Word)
    (x : Tensor) (device : OnDevice allocate x.metadata.device) (laws : TensorLaws x) (h : InputContract x.metadata)
    (before : State) (resident : Resident x laws h before) (tree : Tree)
    (cover : tree.leaves.Perm (List.range (paddedWidth x.metadata)))
    (schedule : Schedule (launch x.metadata h tree cover))
    (q m : RowwiseQuantizationTriton.Tensor) (after : Heap)
    (success : checkedForward allocate ops x laws h before resident tree cover schedule=.ok (some (q,m,after))) :
    q.device=x.metadata.device ∧ m.device=x.metadata.device := by
  have accepted := (checkTensorInputs_iff x.metadata).mpr h
  simp only [checkedForward,accepted,Except.bind,bind,pure,Except.pure,Except.ok.injEq] at success
  let p := launch x.metadata h tree cover
  let direct := binding x laws h before resident
  change forward allocate ops p schedule (input x laws before) before direct=some (q,m,after) at success
  cases hq : allocate .int8 [p.rows,p.columns] direct.after with
  | none => simp [forward,prepare,hq,p,direct] at success
  | some qa =>
    cases hm : allocate .float16 [p.rows] qa.after with
    | none => simp [forward,prepare,hq,hm,p,direct] at success
    | some ma =>
      have success' : some (qa.tensor,ma.tensor,
          runGrid ops p (Prepared.layout (⟨direct,qa,ma⟩ : Prepared p (input x laws before) before)) schedule.order ma.after.heap)=some (q,m,after) := by
        simpa [forward,prepare,hq,hm,bind,Option.bind,pure] using success
      simp only [Option.some.injEq,Prod.mk.injEq] at success'
      rcases success' with ⟨rfl,rfl,rfl⟩
      exact ⟨device _ _ _ _ hq,device _ _ _ _ hm⟩

#axiomsClean binding
#axiomsClean actual_pointer
#axiomsClean checked_correct
#axiomsClean checked_devices
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RowwiseQuantizationTriton.Direct
