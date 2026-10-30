import bench.tritonbench_g.attention_fwd_triton3.Proofs.Grid
import bench.tritonbench_g.attention_fwd_triton3.Proofs.NativeEval
import bench.tritonbench_g.attention_fwd_triton3.Proofs.Bytes

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Native
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.AttentionFwdTriton3
open Host Ownership
open VeriTile.Bench.Repairs.LightningAttention.KernelOnly (encode decode load32)
noncomputable section

def initialState (A : FlatAlloc) (before : Bytes.Memory) : State Expr where
  mem := fun rg off => .mk .real (.input rg (load32 before (4*A.addr rg off)))
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 0
  undef := fun d _ _ => defaultValue d

theorem shape_of_admission (c : Call) (A : FlatAlloc) (p : Checked.Pointers A)
    (r : TensorBinding.Request) (admission : TensorBinding.Admission c A p r) : Shape c := by
  have safety := Contract.safety admission.checked
  refine ⟨safety.valid.blockM,safety.valid.blockN,?_⟩
  have metadata := admission.checked.metadata
  simp only [Checked.metadataChecks,decide_eq_true_eq] at metadata
  obtain ⟨_,_,_,_,_,_,_,_,_,_,_,_,_,dim,_,_,_,_⟩ := metadata
  rcases dim with h|h|h|h <;> omega

/-- M/L permit owner selection separately for every byte. O uses the selected
matrix-result owner. Values always come from the completed source memory. -/
def projectLane (c : Call) (shape : Shape c) (schedule : Raw.Schedule c)
    (ops : Raw.Ops) (scale : Word) (memory : RegionName → Nat → Cell Expr)
    (idx : GridIndex c.grid) (lane : Lane c.BM c.D) : Word :=
  let location := Ownership.address c idx lane
  let value := (memory location.1 location.2).read .real
  match lane with
  | .center row => decode (fun byte => encode
      (evaluate c shape schedule ops scale value row.val 0 (schedule.centerStoreOwner idx row byte)) byte)
  | .denominator row => decode (fun byte => encode
      (evaluate c shape schedule ops scale value row.val 0 (schedule.denominatorStoreOwner idx row byte)) byte)
  | .output row dim => evaluate c shape schedule ops scale value row.val dim.val (schedule.outputStoreOwner row dim)

def wordMemory (c : Call) (shape : Shape c) (schedule : Raw.Schedule c)
    (ops : Raw.Ops) (scale : Word) (memory : RegionName → Nat → Cell Expr) : RegionName → Nat → Word := by
  classical
  exact fun rg off => if hit : ∃ idx lane,active c lane ∧ Ownership.address c idx lane=(rg,off) then
    projectLane c shape schedule ops scale memory (Classical.choose hit) (Classical.choose (Classical.choose_spec hit))
  else evaluate c shape schedule ops scale ((memory rg off).read .real) 0 0 0

theorem wordMemory_readback (c : Call) (valid : Valid c) (shape : Shape c) (schedule : Raw.Schedule c)
    (ops : Raw.Ops) (scale : Word) (memory : RegionName → Nat → Cell Expr)
    (idx : GridIndex c.grid) (lane : Lane c.BM c.D) (live : active c lane) :
    wordMemory c shape schedule ops scale memory (Ownership.address c idx lane).1 (Ownership.address c idx lane).2=
      projectLane c shape schedule ops scale memory idx lane := by
  have hit : ∃ jdx j,active c j ∧ Ownership.address c jdx j=
      ((Ownership.address c idx lane).1,(Ownership.address c idx lane).2) := ⟨idx,lane,live,rfl⟩
  rw [wordMemory,dif_pos hit]
  have witness := Classical.choose_spec (Classical.choose_spec hit)
  obtain ⟨hidx,hlane⟩ := Ownership.unique c valid _ idx _ lane witness.2
  exact congrArg₂ (projectLane c shape schedule ops scale memory) hidx hlane

def targetState (c : Call) (A : FlatAlloc) (before : Bytes.Memory) : State Expr :=
  Reference.gridResult (ScaleBinding.template c) backend (initialState A before)

def targetBytes (c : Call) (A : FlatAlloc) (shape : Shape c) (schedule : Raw.Schedule c)
    (ops : Raw.Ops) (before : Bytes.Memory) (scale : Word) : Bytes.Memory :=
  Bytes.publish c A (wordMemory c shape schedule ops scale (targetState c A before).mem) before

theorem six_inputs (c : Call) (A : FlatAlloc) (p : Checked.Pointers A)
    (r : TensorBinding.Request) (admission : TensorBinding.Admission c A p r)
    (shape : Shape c) (schedule : Raw.Schedule c) (ops : Raw.Ops) (scale : Word) (before : Bytes.Memory)
    (role : TensorBinding.Role) (i : Nat) (bound : i<(r role).tensor.metadata.numel)
    (row dim owner : Nat) :
    evaluate c shape schedule ops scale
      (((initialState A before).mem (TensorBinding.region role)
        (TensorBinding.offset (r role).tensor.metadata.shape (r role).strides i)).read .real) row dim owner=
      (Torch.FP32.logicalValue (fun addr => ⟨load32 before addr⟩)
        (r role).tensor (admission.laws role).tensor i).bits := by
  rw [TensorBinding.all_six_inputs c A p r admission _ role i bound]
  simp only [initialState,Cell.read_same,evaluate]

/-- Complete candidate grid execution in the explicit native interpretation,
independent all-six-input target, and exact byte publication. -/
theorem complete_native (c : Call) (A : FlatAlloc) (p : Checked.Pointers A)
    (r : TensorBinding.Request) (admission : TensorBinding.Admission c A p r)
    (closed : ∀ rg,rg∉A.regions → A.extent rg=0)
    (schedule : Raw.Schedule c) (ops : Raw.Ops) (before : Bytes.Memory) (scale : Word) :
    let shape := shape_of_admission c A p r admission
    ∃ final,Reference.Composition.schedule backend.algebra (Reference.launched (ScaleBinding.template c)) c.grid
      (Reference.Composition.allPrograms c.grid) (initialState A before)=some final ∧
      Bytes.publish c A (wordMemory c shape schedule ops scale final.mem) before=
        targetBytes c A shape schedule ops before scale ∧
      (∀ idx lane,active c lane → load32 (targetBytes c A shape schedule ops before scale)
        (4*(Flat.address c A idx lane).2)=projectLane c shape schedule ops scale (targetState c A before).mem idx lane) ∧
      (∀ addr,(¬∃ idx lane b,active c lane ∧ Physical.address c A idx lane b=addr) →
        targetBytes c A shape schedule ops before scale addr=before addr) := by
  dsimp only
  let shape := shape_of_admission c A p r admission
  obtain ⟨final,execution,meaning⟩ := Reference.complete_grid (ScaleBinding.template c) backend
    shape.blockN (initialState A before)
  refine ⟨final,execution,?_,?_,?_⟩
  · unfold targetBytes targetState
    rw [meaning]
  · intro idx lane live
    rw [targetBytes,Bytes.word_readback c A p r admission closed]
    exact wordMemory_readback c (Contract.safety admission.checked).valid shape schedule ops scale _ idx lane live
    exact live
  · intro addr outside
    exact Bytes.frame c A _ before addr outside

theorem native_equiv (c : Call) (A : FlatAlloc) (shape : Shape c) (schedule : Raw.Schedule c)
    (actual reference : Raw.Ops) (laws : Raw.LocalLaws actual reference) (before : Bytes.Memory) (scale : Word) :
    targetBytes c A shape schedule actual before scale=targetBytes c A shape schedule reference before scale := by
  rw [Raw.operations_eq actual reference laws]

#axiomsClean six_inputs
#axiomsClean complete_native
#axiomsClean native_equiv
#specNonCircular targetBytes avoiding [complete_native,FP.Structural.exec,FP.Structural.run,Kernels.source]
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Native
