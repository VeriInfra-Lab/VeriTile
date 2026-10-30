import bench.repairs.tritonbench_g.quantize_kv_copy.Proofs.KernelLowering

namespace VeriTile.Bench.Repairs.QuantizeKvCopy.KernelPublic
open VeriTile.Triton FP.Structural Kernel KernelModels KernelWhole KernelTensor
noncomputable section
open Classical
local instance : Inhabited Word := ⟨KernelModels.zero⟩
set_option maxHeartbeats 2500000

structure NativeOutput where
  tensors : Tensors
  final : State Word
  heap : Heap
  returned : Option Unit

/-- Same checked arguments, same two caller Tensor objects, actual signed
INT32 backing words, and the complete two-dimensional canonical source grid.
Native scalar instructions and reduction layout are explicit compiler/ISA
boundaries supplied by a selected execution profile. -/
def nativeCall (a : NativeInstructions) (tree : ReductionTree) (t : Tensors) (before : Heap) : Option NativeOutput :=
  let m := metadata t
  let w := words t before
  if checkInputs m w=.ok () then
    (KernelGrid.run (nativeOps a tree) (parameters m w) (List.range (request m w).programs) (initial m before)).map
      (fun final => ⟨t,final,KernelPhysical.post m w before final,none⟩)
  else none

/-- Caller-visible outputs, actual Tensor logical values, exact source-final
FP32 scale bits and signed-byte codes, plus the entire exterior physical frame. -/
def NativeResult (a : NativeInstructions) (tree : ReductionTree) (t : Tensors) (before : Heap) (out : NativeOutput) : Prop :=
  let m := metadata t
  let w := words t before
  let x := values m w before
  out.tensors=t ∧ out.returned=none ∧
  KernelWhole.Result m w (fun pid g c => (nativeCode a tree (x pid g) c).toInt)
    (fun pid g => nativeScale a tree (x pid g)) (initial m before) out.final ∧
  KernelPhysical.Result m w before out.heap (fun pid g c => nativeCode a tree (x pid g) c)
    (fun pid g => nativeScale a tree (x pid g)) ∧
  (∀ pid g c,pid<(request m w).programs → g<(request m w).groups → c<8 →
    logicalInt8 out.heap (t .output) (codeOffset (parameters m w) ((request m w).destination pid) g c)=nativeCode a tree (x pid g) c) ∧
  (∀ pid g,pid<(request m w).programs → g<(request m w).groups →
    logicalFP32 out.heap (t .scale) (scaleOffset (parameters m w) ((request m w).destination pid) g)=nativeScale a tree (x pid g)) ∧
  (∀ i,i<(m .output).numel → out.heap (base m .output+i)=KernelPhysical.byteOfInt ((out.final.mem "O" i).read .int)) ∧
  (∀ i (b : Fin 4),i<(m .scale).numel → out.heap (base m .scale+4*i+b.val)=scaleByte ((out.final.mem "S" i).read .real) b)

theorem native (a : NativeInstructions) (tree : ReductionTree) (t : Tensors) (laws : ∀ r,Laws (t r))
    (before : Heap) (h : InputContract (metadata t) (words t before)) :
    ∃ out,nativeCall a tree t before=some out ∧ NativeResult a tree t before out := by
  let m := metadata t
  let w := words t before
  let x := values m w before
  obtain ⟨final,run,result⟩ := KernelWhole.native a tree m w h x (initial m before) (initial_reads t before)
  have hp := KernelPhysical.of_source m w h before (initial m before) final
    (fun pid g c => nativeCode a tree (x pid g) c) (fun pid g => nativeScale a tree (x pid g)) result
  have hl := KernelPhysical.logical_outputs t laws w h before _ _ _ hp
  let out : NativeOutput := ⟨t,final,KernelPhysical.post m w before final,none⟩
  refine ⟨out,?_,rfl,rfl,result,hp,hl.1,hl.2,?_,?_⟩
  · dsimp only [m,w] at run
    simp [nativeCall,(checkInputs_iff _ _).mpr h,run,out,m,w]
  · exact KernelLowering.output_reifies m w h before final _ _ result
  · exact KernelLowering.scale_reifies m w h before final _ _ result

/-- No destination token means no source programs, no physical writes and
None, even when caller storage contains arbitrary words. -/
theorem empty_native (a : NativeInstructions) (tree : ReductionTree) (t : Tensors) (before : Heap)
    (h : InputContract (metadata t) (words t before))
    (empty : (request (metadata t) (words t before)).tokens=0) :
    ∃ out,nativeCall a tree t before=some out ∧ out.tensors=t ∧ out.returned=none ∧ out.heap=before := by
  let m := metadata t
  let w := words t before
  have hp : (request m w).programs=0 := by simp [Request.programs,empty,m,w]
  have frame : KernelPhysical.post m w before (initial m before)=before := by
    funext address
    apply KernelPhysical.post_frame
    simp [KernelPhysical.Touched,KernelPhysical.CodeTouched,KernelPhysical.ScaleTouched,hp]
  refine ⟨⟨t,initial m before,before,none⟩,?_,rfl,rfl,rfl⟩
  simp [nativeCall,(checkInputs_iff m w).mpr h,hp,KernelGrid.run,frame,m,w] at *

/-- Arbitrary Real input and scale memory at the caller Tensor addresses.
This independent interpretation does not encode Real values as FP32 words. -/
def realInitial (m : Inputs) (w : Nat → MetadataWord) (realMemory : Nat → ℝ) (codeMemory : Nat → BitVec 8) : State ℝ where
  mem := fun r i =>
    if r="K" then .mk .real (realMemory (base m .input+4*i))
    else if r="D" then .mk .int (w i).toInt
    else if r="O" then .mk .int (codeMemory (base m .output+i)).toInt
    else if r="S" then .mk .real (realMemory (base m .scale+4*i))
    else .mk .int 0
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 0
  undef := fun d _ _ => defaultValue d

def realValues (m : Inputs) (w : Nat → MetadataWord) (realMemory : Nat → ℝ) (pid g c : Nat) : ℝ :=
  realMemory (base m .input+4*inputOffset (parameters m w) pid g c)

theorem mathematical (t : Tensors) (before : Heap) (h : InputContract (metadata t) (words t before))
    (realMemory : Nat → ℝ) (codeMemory : Nat → BitVec 8) :
    let m := metadata t
    let w := words t before
    let initial := realInitial m w realMemory codeMemory
    let x := realValues m w realMemory
    ∃ final,KernelGrid.run KernelModels.realOps (parameters m w) (List.range (request m w).programs) initial=some final ∧
      KernelWhole.Result m w (fun pid g c => (realCode (x pid g) c).toInt)
        (fun pid g => realScale (x pid g)) initial final := by
  apply KernelWhole.mathematical _ _ h
  constructor
  · intro pid g c _ _ _
    simp [realInitial,realValues]
  · intro token _
    simp [realInitial]

#axiomsClean native
#axiomsClean empty_native
#axiomsClean mathematical
#specNonCircular NativeResult avoiding [nativeCall,Kernel.kernel,FP.Structural.exec,KernelGrid.run,KernelPhysical.post]
#specNonCircular realValues avoiding [realInitial,Kernel.kernel,FP.Structural.exec,KernelGrid.run]
end
end VeriTile.Bench.Repairs.QuantizeKvCopy.KernelPublic
