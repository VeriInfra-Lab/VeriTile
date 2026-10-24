import bench.repairs.tritonbench_g.rowwise_quantization_triton.Proofs.Direct
import bench.repairs.tritonbench_g.rowwise_quantization_triton.Proofs.Mathematics

/-! The source-row store graph is polymorphic in the loaded scalar. This
module instantiates it at arbitrary real values; the int8 and FP16 stores stay
byte-valued. No real input is encoded into an FP32 word. The fresh outputs
cannot change the source input, so its mathematical interpretation is fixed
through the grid just as the raw input is fixed in runGrid_frozen. -/
namespace VeriTile.Bench.Repairs.RowwiseQuantizationTriton.RealGrid
noncomputable section

def qValue (half : ℝ → Half) (p : Launch) (values : Nat → Nat → ℝ) (r c : Nat) : Byte :=
  rowQuantized (idealOperations half) (values r c) (rowMaximum (idealOperations half) p (values r))
def mValue (half : ℝ → Half) (p : Launch) (values : Nat → Nat → ℝ) (r : Nat) (b : Fin 2) : Byte :=
  halfByte (half (rowMaximum (idealOperations half) p (values r))) b

def source (half : ℝ → Half) (p : Launch) (v : Layout) (values : Nat → Nat → ℝ) (row : Nat) : List Write :=
  sourceRowValues (idealOperations half) p v (values row) row

def execute (half : ℝ → Half) (p : Launch) (v : Layout) (values : Nat → Nat → ℝ)
    (order : List Nat) (before : Heap) : Heap :=
  order.foldl (fun heap row => writeAll (source half p v values row) heap) before

theorem execute_stores (half : ℝ → Half) (p : Launch) (v : Layout) (values : Nat → Nat → ℝ)
    (order : List Nat) (before : Heap) :
    execute half p v values order before=writeAll (order.flatMap (source half p v values)) before := by
  induction order generalizing before with
  | nil => rfl
  | cons row rest ih =>
    simp only [execute,List.foldl_cons,List.flatMap_cons,writeAll_append]
    exact ih _

theorem source_member (half : ℝ → Half) (p : Launch) (v : Layout) (values : Nat → Nat → ℝ)
    (r : Nat) (event : Write) : event∈source half p v values r ↔
    (∃ c,c<p.columns ∧ event=⟨outputAddress p v r c,qValue half p values r c⟩) ∨
    (∃ b : Fin 2,event=⟨maximumAddress v r b,mValue half p values r b⟩) := by
  simp only [source,sourceRowValues,List.mem_append,List.mem_filterMap,List.mem_range,List.mem_map,List.mem_finRange,true_and]
  constructor
  · rintro (⟨c,hw,he⟩|⟨b,he⟩)
    · by_cases hc : c<p.columns
      · simp only [if_pos hc,Option.some.injEq] at he
        exact Or.inl ⟨c,hc,he.symm⟩
      · simp [hc] at he
    · exact Or.inr ⟨b,he.symm⟩
  · rintro (⟨c,hc,rfl⟩|⟨b,rfl⟩)
    · exact Or.inl ⟨c,lt_of_lt_of_le hc p.covers,by simp [hc,qValue]⟩
    · exact Or.inr ⟨b,rfl⟩

theorem readback (half : ℝ → Half) (p : Launch) (v : Layout)
    (separated : ∀ a,InRange v.output (p.rows*p.columns) a → ¬InRange v.maximum (2*p.rows) a)
    (values : Nat → Nat → ℝ) (schedule : Schedule p) (before : Heap) :
    (∀ r,r<p.rows → ∀ c,c<p.columns → execute half p v values schedule.order before (outputAddress p v r c)=
      mathematicalByte (values r) p.columns c) ∧
    (∀ r,r<p.rows → ∀ b : Fin 2,execute half p v values schedule.order before (maximumAddress v r b)=
      halfByte (half (logicalMaximum (values r) p.columns)) b) := by
  have valid : ∀ r∈schedule.order,r<p.rows := fun r hr => List.mem_range.mp (schedule.complete.mem_iff.mp hr)
  rw [execute_stores]
  constructor
  · intro r hr c hc
    have read : writeAll (schedule.order.flatMap (source half p v values)) before (outputAddress p v r c)=qValue half p values r c := by
      apply writeAll_readback _ before ⟨outputAddress p v r c,qValue half p values r c⟩
      · apply List.mem_flatMap.mpr
        exact ⟨r,schedule.complete.mem_iff.mpr (List.mem_range.mpr hr),(source_member half p v values r _).mpr (Or.inl ⟨c,hc,rfl⟩)⟩
      · intro event he same
        obtain ⟨other,ho,he⟩:=List.mem_flatMap.mp he
        rcases (source_member half p v values other event).mp he with ⟨d,hd,rfl⟩|⟨b,rfl⟩
        · obtain ⟨rfl,rfl⟩:=output_address_injective p v other r d c hd hc same
          rfl
        · change maximumAddress v other b=outputAddress p v r c at same
          exact False.elim (separated _ (output_bound p v r c hr hc) (same ▸ maximum_bound p v other b (valid other ho)))
    exact read.trans (source_quantization_is_logical half p (values r) c hc)
  · intro r hr b
    have read : writeAll (schedule.order.flatMap (source half p v values)) before (maximumAddress v r b)=mValue half p values r b := by
      apply writeAll_readback _ before ⟨maximumAddress v r b,mValue half p values r b⟩
      · apply List.mem_flatMap.mpr
        exact ⟨r,schedule.complete.mem_iff.mpr (List.mem_range.mpr hr),(source_member half p v values r _).mpr (Or.inr ⟨b,rfl⟩)⟩
      · intro event he same
        obtain ⟨other,ho,he⟩:=List.mem_flatMap.mp he
        rcases (source_member half p v values other event).mp he with ⟨d,hd,rfl⟩|⟨e,rfl⟩
        · change outputAddress p v other d=maximumAddress v r b at same
          exact False.elim (separated _ (output_bound p v other d (valid other ho) hd) (by rw [same];exact maximum_bound p v r b hr))
        · obtain ⟨rfl,rfl⟩:=maximum_address_injective v other r e b same
          rfl
    simpa [mValue,source_maximum_is_logical] using read

theorem frame (half : ℝ → Half) (p : Launch) (v : Layout) (values : Nat → Nat → ℝ)
    (schedule : Schedule p) (before : Heap) (address : Nat) (outside : ¬OutputCell p v address) :
    execute half p v values schedule.order before address=before address := by
  rw [execute_stores]
  apply writeAll_frame
  intro event he same
  obtain ⟨r,hr,he⟩:=List.mem_flatMap.mp he
  have rb:=List.mem_range.mp (schedule.complete.mem_iff.mp hr)
  rcases (source_member half p v values r event).mp he with ⟨c,hc,rfl⟩|⟨b,rfl⟩
  · exact outside (same ▸ Or.inl (output_bound p v r c rb hc))
  · exact outside (same ▸ Or.inr (maximum_bound p v r b rb))

/-- Two actual allocation calls, followed by all source rows. The real source
values may be arbitrary; the separately supplied half cast retains the explicit
FP16 scale result rather than pretending it is an exact real storage format. -/
def forward (allocate : Direct.Allocator) (half : ℝ → Half) (p : Launch) (schedule : Schedule p)
    (values : Nat → Nat → ℝ) (before : State) : Option (Tensor × Tensor × Heap) := do
  let q ← allocate .int8 [p.rows,p.columns] before
  let m ← allocate .float16 [p.rows] q.after
  return (q.tensor,m.tensor,execute half p ⟨0,q.tensor.base,m.tensor.base⟩ values schedule.order m.after.heap)

def Result (half : ℝ → Half) (p : Launch) (values : Nat → Nat → ℝ) (before : State)
    (q m : Tensor) (after : Heap) : Prop :=
  q.dtype=.int8 ∧ q.shape=[p.rows,p.columns] ∧ m.dtype=.float16 ∧ m.shape=[p.rows] ∧
  (∀ r,r<p.rows → ∀ c,c<p.columns → after (q.base+r*p.columns+c)=mathematicalByte (values r) p.columns c) ∧
  (∀ r,r<p.rows → ∀ b : Fin 2,after (m.base+2*r+b.val)=halfByte (half (logicalMaximum (values r) p.columns)) b) ∧
  (∀ a,before.live a → after a=before.heap a)

theorem returned (allocate : Direct.Allocator) (half : ℝ → Half) (p : Launch) (schedule : Schedule p)
    (values : Nat → Nat → ℝ) (before : State) (q m : Tensor) (after : Heap)
    (success : forward allocate half p schedule values before=some (q,m,after)) : Result half p values before q m after := by
  cases hq : allocate .int8 [p.rows,p.columns] before with
  | none => simp [forward,hq] at success
  | some qa =>
    cases hm : allocate .float16 [p.rows] qa.after with
    | none => simp [forward,hq,hm] at success
    | some ma =>
      simp [forward,hq,hm,bind,Option.bind,pure] at success
      rcases success with ⟨rfl,rfl,rfl⟩
      let v : Layout := ⟨0,qa.tensor.base,ma.tensor.base⟩
      have sep : ∀ a,InRange v.output (p.rows*p.columns) a → ¬InRange v.maximum (2*p.rows) a := by
        intro a hq hm
        exact ma.fresh a (by simpa [DType.bytes,v] using hm) (qa.resident a (by simpa [DType.bytes,v] using hq))
      obtain ⟨qr,mr⟩:=readback half p v sep values schedule ma.after.heap
      refine ⟨qa.type_eq,qa.shape_eq,ma.type_eq,ma.shape_eq,qr,mr,?_⟩
      intro a live
      rw [frame half p v values schedule ma.after.heap a]
      · exact (qa.extension.trans ma.extension).frame a live
      · rintro (hq|hm)
        · exact qa.fresh a (by simpa [DType.bytes,v] using hq) live
        · exact ma.fresh a (by simpa [DType.bytes,v] using hm) (qa.extension.live a live)


/-- Arbitrary real interpretation of the original Tensor storage. -/
def inputValues (storage : Nat → ℝ) (x : Direct.Tensor) (r c : Nat) : ℝ :=
  let value := storage (x.storage.address (x.physicalIndex (r*columns x.metadata+c)))
  if x.metadata.isNeg then -value else value

theorem input_pointer (storage : Nat → ℝ) (x : Direct.Tensor) (laws : Direct.TensorLaws x)
    (h : InputContract x.metadata) (r c : Nat) (hr : r<rows x.metadata) (hc : c<columns x.metadata) :
    storage ((Direct.view x laws h).base+4*(r*columns x.metadata+c))=inputValues storage x r c := by
  have hi : r*columns x.metadata+c<x.metadata.numel := by
    rw [numel_eq x.metadata h]
    exact linear_bound _ _ r c hr hc
  simp only [inputValues,h.noNeg,Bool.false_eq_true,↓reduceIte,laws.contiguous h.contiguous _ hi]
  congr 1
  simp only [Direct.view,VeriTile.Torch.FP32.dataView,VeriTile.Torch.FP32.DataView.base,
    if_neg (by omega : x.metadata.numel≠0),VeriTile.Torch.FP32.Storage.address]
  omega

def checkedForward (allocate : Direct.Allocator) (half : ℝ → Half)
    (x : Direct.Tensor) (h : InputContract x.metadata) (storage : Nat → ℝ) (before : State)
    (tree : Tree) (cover : tree.leaves.Perm (List.range (paddedWidth x.metadata)))
    (schedule : Schedule (Direct.launch x.metadata h tree cover)) :
    Except InputError (Option (Tensor × Tensor × Heap)) := do
  checkTensorInputs x.metadata
  return forward allocate half (Direct.launch x.metadata h tree cover) schedule (inputValues storage x) before

theorem checked_correct (allocate : Direct.Allocator) (half : ℝ → Half)
    (x : Direct.Tensor) (h : InputContract x.metadata) (storage : Nat → ℝ) (before : State)
    (tree : Tree) (cover : tree.leaves.Perm (List.range (paddedWidth x.metadata)))
    (schedule : Schedule (Direct.launch x.metadata h tree cover))
    (q m : Tensor) (after : Heap)
    (success : checkedForward allocate half x h storage before tree cover schedule=.ok (some (q,m,after))) :
    Result half (Direct.launch x.metadata h tree cover) (inputValues storage x) before q m after := by
  have accepted := (checkTensorInputs_iff x.metadata).mpr h
  simp only [checkedForward,accepted,Except.bind,bind,pure,Except.pure,Except.ok.injEq] at success
  exact returned allocate half _ schedule _ before q m after success


theorem checked_devices (allocate : Direct.Allocator) (half : ℝ → Half)
    (x : Direct.Tensor) (h : InputContract x.metadata) (storage : Nat → ℝ) (before : State)
    (tree : Tree) (cover : tree.leaves.Perm (List.range (paddedWidth x.metadata)))
    (schedule : Schedule (Direct.launch x.metadata h tree cover))
    (device : Direct.OnDevice allocate x.metadata.device) (q m : Tensor) (after : Heap)
    (success : checkedForward allocate half x h storage before tree cover schedule=.ok (some (q,m,after))) :
    q.device=x.metadata.device ∧ m.device=x.metadata.device := by
  have accepted := (checkTensorInputs_iff x.metadata).mpr h
  simp only [checkedForward,accepted,Except.bind,bind,pure,Except.pure,Except.ok.injEq] at success
  let p := Direct.launch x.metadata h tree cover
  change forward allocate half p schedule (inputValues storage x) before=some (q,m,after) at success
  cases hq : allocate .int8 [p.rows,p.columns] before with
  | none => simp [forward,hq] at success
  | some qa =>
    cases hm : allocate .float16 [p.rows] qa.after with
    | none => simp [forward,hq,hm] at success
    | some ma =>
      simp [forward,hq,hm,bind,Option.bind,pure] at success
      rcases success with ⟨rfl,rfl,rfl⟩
      exact ⟨device _ _ _ _ hq,device _ _ _ _ hm⟩

#axiomsClean checked_devices

#axiomsClean input_pointer
#axiomsClean checked_correct

#specNonCircular Result avoiding [forward,execute,source,sourceRowValues]
#axiomsClean readback
#axiomsClean returned
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RowwiseQuantizationTriton.RealGrid
