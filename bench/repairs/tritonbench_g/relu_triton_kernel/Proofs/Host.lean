import bench.repairs.tritonbench_g.relu_triton_kernel.Proofs.Structural
import bench.repairs.tritonbench_g.relu_triton_kernel.Proofs.Contract
import VeriTile.Torch.FP32

/-! Direct host binding with no copied input. Storage cells are FP32 words
indexed by their byte start address. The Real interpretation uses the same
addressing with ℝ cells; it does not require a Real-to-byte encoding.
EmptyLikeResult supplies only ordinary allocation laws, never kernel results. -/
namespace VeriTile.Bench.Repairs.ReluTritonKernel.Host
open VeriTile.Triton FP.Structural
export VeriTile.Torch.FP32 (Tensor TensorLaws Storage DataView dataView)
noncomputable section
set_option maxHeartbeats 4000000

abbrev base (x : Tensor) (laws : TensorLaws x) (hc : x.metadata.isContiguous = Bool.true) :=
  (dataView x laws hc).base

def logical {α : Type} [Inhabited α] (negate : α → α) (s : State α)
    (x : Tensor) (_laws : TensorLaws x) (i : Nat) : α :=
  let raw := (s.mem "memory" (x.storage.address (x.physicalIndex i))).read .real
  if x.metadata.isNeg then negate raw else raw

def OutsideStorage (s : Storage) (r : RegionName) (a : Nat) : Prop :=
  r≠"memory" ∨ a<s.base ∨ s.base+4*s.elements≤a

def SeparateStorage (x y : Storage) : Prop :=
  x.base+4*x.elements≤y.base ∨ y.base+4*y.elements≤x.base

structure EmptyLikeResult {α : Type} [Inhabited α]
    (x : Tensor) (h : InputContract x.metadata) (before : State α) where
  output : Tensor
  laws : TensorLaws output
  allocated : State α
  metadata : output.metadata = {x.metadata with isNeg:=Bool.false,dataPtr:=output.dataPtr}
  alignment : 0<output.metadata.numel → output.dataPtr.toNat%4=0
  fresh : SeparateStorage x.storage output.storage
  freshObject : output.object≠x.object
  frame : ∀ r a, OutsideStorage output.storage r a → allocated.mem r a=before.mem r a

theorem EmptyLikeResult.contiguous {α : Type} [Inhabited α]
    {x : Tensor} {h : InputContract x.metadata} {before : State α}
    (a : EmptyLikeResult x h before) : a.output.metadata.isContiguous=Bool.true := by
  rw [a.metadata]; exact h.contiguous

theorem EmptyLikeResult.numel {α : Type} [Inhabited α]
    {x : Tensor} {h : InputContract x.metadata} {before : State α}
    (a : EmptyLikeResult x h before) : a.output.metadata.numel=x.metadata.numel := by
  simp [a.metadata,VeriTile.Torch.TensorMetadata.numel]

theorem address_eq (x : Tensor) (laws : TensorLaws x)
    (hc : x.metadata.isContiguous=Bool.true) (i : Nat) (hi : i<x.metadata.numel) :
    x.storage.address (x.physicalIndex i)=base x laws hc+4*i := by
  rw [laws.contiguous hc i hi]
  simp only [base,dataView,VeriTile.Torch.FP32.DataView.base,if_neg (by omega : x.metadata.numel≠0),VeriTile.Torch.FP32.Storage.address]
  omega

theorem direct_read {α : Type} [Inhabited α] (negate : α → α) (s : State α)
    (x : Tensor) (laws : TensorLaws x) (hc : x.metadata.isContiguous=Bool.true)
    (hn : x.metadata.isNeg=Bool.false) (i : Nat) (hi : i<x.metadata.numel) :
    (s.mem "memory" (base x laws hc+4*i)).read .real=logical negate s x laws i := by
  simp only [logical,hn,Bool.false_eq_true,↓reduceIte,address_eq x laws hc i hi]

theorem data_inside (x : Tensor) (laws : TensorLaws x)
    (hc : x.metadata.isContiguous=Bool.true) (i : Nat) (hi : i<x.metadata.numel) :
    x.storage.base≤base x laws hc+4*i ∧
      base x laws hc+4*i+4≤x.storage.base+4*x.storage.elements := by
  have cap:=VeriTile.Torch.FP32.contiguous_capacity x laws hc (by omega)
  simp only [base,dataView,VeriTile.Torch.FP32.DataView.base,if_neg (by omega : x.metadata.numel≠0),VeriTile.Torch.FP32.Storage.address]
  omega

theorem allocation_separate {α : Type} [Inhabited α] (x : Tensor) (laws : TensorLaws x)
    (h : InputContract x.metadata) (before : State α) (a : EmptyLikeResult x h before) :
    Structural.Separate "memory" "memory" (base x laws h.contiguous)
      (base a.output a.laws a.contiguous) 4 x.metadata.numel := by
  intro i hi j hj
  right
  have bx:=data_inside x laws h.contiguous i hi
  have bo:=data_inside a.output a.laws a.contiguous j (by rw [a.numel];exact hj)
  have sep:=a.fresh
  unfold SeparateStorage at sep
  omega

theorem input_outside {α : Type} [Inhabited α] (x : Tensor) (laws : TensorLaws x)
    (h : InputContract x.metadata) (before : State α) (a : EmptyLikeResult x h before)
    (i : Nat) (hi : i<x.metadata.numel) :
    OutsideStorage a.output.storage "memory" (base x laws h.contiguous+4*i) := by
  have bx:=data_inside x laws h.contiguous i hi
  have sep:=a.fresh
  unfold SeparateStorage at sep
  unfold OutsideStorage
  omega

/-- Successful ordinary allocation and launch interpretation. The exact
allocation object is returned. An empty Tensor launches no programs. -/
def runWrapper {α : Type} [Inhabited α] (M : Algebra α)
    (x : Tensor) (laws : TensorLaws x) (h : InputContract x.metadata)
    (before : State α) (a : EmptyLikeResult x h before) (schedule : Structural.Schedule x.metadata.numel) :
    Except InputError (Option (Tensor × State α)) := do
  checkInputs (.tensor x.metadata)
  if x.metadata.numel=0 then return some (a.output,a.allocated)
  return (Structural.runGrid M "memory" "memory" (base x laws h.contiguous)
    (base a.output a.laws a.contiguous) 4 x.metadata.numel schedule a.allocated).map (a.output,·)

/-- Kernel success, every returned value and the caller-owned frame are
conclusions. The source is interpreted directly against pointer-bound storage. -/
theorem wrapper_correct {α : Type} [Inhabited α] (M : Algebra α)
    (cmp : α → α → Bool) (hc : M.compareLe (some .fp32) .real=some cmp)
    (negate : α → α) (x : Tensor) (laws : TensorLaws x) (h : InputContract x.metadata)
    (before : State α) (a : EmptyLikeResult x h before) (schedule : Structural.Schedule x.metadata.numel) :
    ∃ final, runWrapper M x laws h before a schedule=.ok (some (a.output,final)) ∧
      (∀ i,i<x.metadata.numel → logical negate final a.output a.laws i=
        Structural.scalar M cmp (M.fp32Load (logical negate before x laws i))) ∧
      (∀ r address,OutsideStorage a.output.storage r address → final.mem r address=before.mem r address) ∧
      (∀ i,i<x.storage.elements → final.mem "memory" (x.storage.address i)=
        before.mem "memory" (x.storage.address i)) := by
  have accepted : checkInputs (.tensor x.metadata)=.ok () := (checkTensorInputs_iff _).mpr h
  have noNeg : a.output.metadata.isNeg=Bool.false := by rw [a.metadata]
  have input_frame (s : State α)
      (frame : ∀ r address,OutsideStorage a.output.storage r address → s.mem r address=before.mem r address) :
      ∀ i,i<x.storage.elements → s.mem "memory" (x.storage.address i)=before.mem "memory" (x.storage.address i) := by
    intro i hi
    apply frame
    have sep:=a.fresh
    unfold SeparateStorage at sep
    unfold OutsideStorage VeriTile.Torch.FP32.Storage.address
    omega
  by_cases empty : x.metadata.numel=0
  · exact ⟨a.allocated,by simp [runWrapper,accepted,empty],by simp [empty],a.frame,
      input_frame a.allocated a.frame⟩
  · obtain ⟨final,he,hv,hf⟩:=Structural.grid_complete M cmp hc "memory" "memory"
      (base x laws h.contiguous) (base a.output a.laws a.contiguous) 4 x.metadata.numel
      (by decide) (allocation_separate x laws h before a) schedule a.allocated
    have frame : ∀ r address,OutsideStorage a.output.storage r address → final.mem r address=before.mem r address := by
      intro r address outside
      rw [hf r address ?_,a.frame r address outside]
      rcases outside with hr | lo | high
      · exact Or.inl hr
      · right
        intro i hi
        have bound:=data_inside a.output a.laws a.contiguous i (by rw [a.numel];exact hi)
        omega
      · right
        intro i hi
        have bound:=data_inside a.output a.laws a.contiguous i (by rw [a.numel];exact hi)
        omega
    refine ⟨final,?_,?_,frame,input_frame final frame⟩
    · simp [runWrapper,accepted,empty,he]
    · intro i hi
      rw [←direct_read negate final a.output a.laws a.contiguous noNeg i (by rw [a.numel];exact hi)]
      rw [hv i hi]
      change Structural.scalar M cmp (M.fp32Load ((a.allocated.mem "memory" (base x laws h.contiguous+4*i)).read .real))=_
      rw [a.frame _ _ (input_outside x laws h before a i hi),direct_read negate before x laws h.contiguous h.noNeg i hi]

/-- Physical 64-bit pointer arithmetic and complete 4-byte access stay in the
valid storage span; inactive lanes need no padded backing allocation. -/
theorem native_pointer (x : Tensor) (laws : TensorLaws x) (h : InputContract x.metadata)
    (pid lane : Nat) (hp : pid<gridCount x.metadata.numel) (hl : lane<1024)
    (active : pid*1024+lane<x.metadata.numel) :
    let addr := (x.dataPtr +
      (VeriTile.Bench.TritonBenchG.ReluTritonKernel.FiniteAddress.offset32 pid lane).signExtend 64 *
        BitVec.ofNat 64 4).toNat
    addr=base x laws h.contiguous+4*(pid*1024+lane) ∧
    addr%4=0 ∧ addr+4≤2^64 := by
  have exactOffset := VeriTile.Bench.TritonBenchG.ReluTritonKernel.FiniteAddress.offset_sign_extend
    x.metadata.numel pid lane h.extent hp hl
  have pointer:=VeriTile.Torch.FP32.actual_pointer x laws h.contiguous (by omega)
  have bound:=(dataView x laws h.contiguous).spanFits
  have align:=VeriTile.Torch.FP32.dataView_aligned x laws h.contiguous h.alignment
  change base x laws h.contiguous+4*x.metadata.numel≤2^64 at bound
  change base x laws h.contiguous%4=0 at align
  simp only [VeriTile.Torch.FP32.DataView.address,Nat.mul_zero,Nat.add_zero] at pointer
  have eqAddr : (x.dataPtr +
      (VeriTile.Bench.TritonBenchG.ReluTritonKernel.FiniteAddress.offset32 pid lane).signExtend 64 *
        BitVec.ofNat 64 4).toNat=base x laws h.contiguous+4*(pid*1024+lane) := by
    rw [←pointer,exactOffset,BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat,BitVec.toNat_ofNat]
    change (base x laws h.contiguous+(pid*1024+lane)*4)%2^64=_
    rw [Nat.mod_eq_of_lt (by omega : base x laws h.contiguous+(pid*1024+lane)*4<2^64)]
    omega
  dsimp only
  rw [eqAddr]
  refine ⟨rfl,?_,by omega⟩
  simpa [Nat.add_mod,Nat.mul_mod] using align

#axiomsClean wrapper_correct
#axiomsClean native_pointer
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ReluTritonKernel.Host
