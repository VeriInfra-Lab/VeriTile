import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.Proofs.Backing
import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.Proofs.NativeSource

namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.Backing
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.AttentionLlama HostGrid NativeSchedule
open Source
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

structure Objects (B : Nat) (c : Config) (p : RegionName→Pointer) where
  query : TensorObject
  key : TensorObject
  value : TensorObject
  output : TensorObject
  queryLayout : TensorLayout query (p c.Q) B c.NH c.H c.D [c.sqz,c.sqh,c.sqm,c.sqk] .half
  keyLayout : TensorLayout key (p c.K) B c.NH c.N c.D [c.skz,c.skh,c.skn,c.skk]
    (if c.fp8 then .int8 else .half)
  valueLayout : TensorLayout value (p c.V) B c.NH c.N c.D [c.svz,c.svh,c.svk,c.svn]
    (if c.fp8 then .int8 else .half)
  outputLayout : TensorLayout output (p c.Out) B c.NH c.H c.D [c.soz,c.soh,c.som,c.son] .half
  deviceQuery : key.device=query.device
  deviceValue : value.device=query.device
  deviceOutput : output.device=query.device
  fresh : Fresh [query,key,value] output

theorem layout_bounds (t : TensorObject) (p : Pointer) (strides : List Nat)
    (dtype : Checked.DType) (h : TensorLayout t p B NH H D strides dtype)
    (i : Index B NH H D) :
    t.allocationBase≤tensorAddress t i ∧ tensorAddress t i<t.allocationBase+t.allocationBytes := by
  have hl:=h.lower
  have hb:=h.backed i
  have hw : 0<p.width := by rw [←h.width]; unfold elementWidth; split <;> decide
  have ha : t.dataPointer≤tensorAddress t i := by unfold tensorAddress; omega
  omega

theorem output_address (c : Config) (p : RegionName→Pointer) (o : Objects B c p)
    (i : Index B c.NH c.H c.D) :
    tensorAddress o.output i=(p c.Out).address (Tensor.offset c i) := by
  simp only [tensorAddress,elementOffset,o.outputLayout.pointer,o.outputLayout.stride,
    o.outputLayout.width,Pointer.address,Tensor.offset,List.getElem?_cons_zero,
    List.getElem?_cons_succ,Option.getD_some]

theorem output_injective (c : Config) (p : RegionName→Pointer) (o : Objects B c p)
    (h : Function.Injective (Tensor.offset (B:=B) c)) :
    Function.Injective (tensorAddress (B:=B) (NH:=c.NH) (H:=c.H) (D:=c.D) o.output) := by
  intro i j he
  rw [output_address,output_address] at he
  have hw : (p c.Out).width=2 := by
    rw [←o.outputLayout.width,elementWidth,o.outputLayout.kind]; decide
  apply h
  simp only [Pointer.address,hw] at he
  omega

/-- Only the internal typed read view normalizes cells outside the int8
representation. The actual caller heap is never changed by this projection. -/
def projectCell (cell : Cell (Term α)) : Cell (Term α) :=
  if Float8E5Bits.isInt8 (cell.read .int) then cell else .mk .fp16 (.literal 0)

theorem project_half (cell : Cell (Term α)) :
    (projectCell cell).read .fp16=cell.read .fp16 := by
  unfold projectCell
  split
  · rfl
  · rename_i h
    cases cell with
    | mk dtype value =>
      cases dtype <;> simp [Cell.read,defaultValue,Float8E5Bits.isInt8] at h ⊢
      rfl

theorem project_byte (cell : Cell (Term α))
    (h : Float8E5Bits.isInt8 (cell.read .int)=Bool.true) : projectCell cell=cell := by
  simp [projectCell,h]

theorem project_range (cell : Cell (Term α)) :
    Float8E5Bits.isInt8 ((projectCell cell).read .int)=Bool.true := by
  unfold projectCell
  split
  · assumption
  · change Float8E5Bits.isInt8 (0 : Int)=Bool.true
    decide

def sourceState (p : RegionName→Pointer) (before : Heap (Cell (Term α)))
    (seed : State (Term α)) : State (Term α) :=
  {seed with mem:=fun r offset=>before ((p r).address offset)}

def typedState (p : RegionName→Pointer) (before : Heap (Cell (Term α)))
    (seed : State (Term α)) : State (Term α) :=
  {seed with mem:=fun r offset=>projectCell (before ((p r).address offset))}

/-- Signed-byte representation is required only at reached logical K/V cells
and only on the E5M2 path. Exterior caller storage is unrestricted. -/
structure InputBytes (B : Nat) (c : Config) (s : State (Term α)) : Prop where
  key : c.fp8=Bool.true → ∀ b : Fin B, ∀ h : Fin c.NH, ∀ k : Fin c.N, ∀ d : Fin c.D,
    Float8E5Bits.isInt8 ((s.mem c.K
      (b.val*c.skz+h.val*c.skh+k.val*c.skn+d.val*c.skk)).read .int)=Bool.true
  value : c.fp8=Bool.true → ∀ b : Fin B, ∀ h : Fin c.NH, ∀ k : Fin c.N, ∀ d : Fin c.D,
    Float8E5Bits.isInt8 ((s.mem c.V
      (b.val*c.svz+h.val*c.svh+k.val*c.svk+d.val*c.svn)).read .int)=Bool.true

theorem typed_range (p : RegionName→Pointer) (before : Heap (Cell (Term α)))
    (seed : State (Term α)) : ByteRange (typedState p before seed) :=
  fun r offset=>project_range _

theorem typed_binding (c : Config) (p : RegionName→Pointer) (before : Heap (Cell (Term α)))
    (seed : State (Term α)) (x : Tensor.Inputs α B c.NH c.H c.D)
    (h : Tensor.Binding c (sourceState p before seed) x)
    (bytes : InputBytes B c (sourceState p before seed)) :
    Tensor.Binding c (typedState p before seed) x := by
  constructor
  · intro b h' q d
    change (projectCell _).read .fp16=_
    rw [project_half]
    exact h.query b h' q d
  · intro hf b h' k d hk
    change (projectCell _).read .fp16=_
    rw [project_half]
    exact h.halfKey hf b h' k d hk
  · intro hf b h' k d hk
    change (projectCell _).read .fp16=_
    rw [project_half]
    exact h.halfValue hf b h' k d hk
  · intro hf b h' k d hk
    change UInt8.ofFin (Float8E5Bits.ofInt ((projectCell _).read .int))=_
    have hb:=bytes.key hf b h' ⟨k,hk⟩ d
    dsimp only [sourceState] at hb
    rw [project_byte _ hb]
    exact h.byteKey hf b h' k d hk
  · intro hf b h' k d hk
    change UInt8.ofFin (Float8E5Bits.ofInt ((projectCell _).read .int))=_
    have hb:=bytes.value hf b h' ⟨k,hk⟩ d
    dsimp only [sourceState] at hb
    rw [project_byte _ hb]
    exact h.byteValue hf b h' k d hk

def nativePost (c : Config) (o : Objects B c p) (before : Heap (Cell (Term α)))
    (states : Program B c.NH c.H→State (Term α)) : Heap (Cell (Term α)) :=
  merge before (tensorAddress o.output) owner
    (fun program i=>(states program).mem c.Out (Tensor.offset c i))

structure Returned (Cell : Type) where
  tensor : TensorObject
  backing : Heap Cell

def Returned.read {Value : Type} (r : Returned Value) (i : Index B NH H D) : Value :=
  tensorCell r.backing r.tensor i

/-- One source result per physical program; complete footprints are joined
into one caller heap and the allocated output object reads that same heap. -/
theorem native_backing (A : Target.Ops α) (C : ScalarABI.Conversions α)
    (raw : RawScalar) (admitted : scalarABICheck Bool.true raw=Bool.true)
    (c : Config) (_supported : FPEquivalence.supportedHead c.D)
    (hD : 0<c.D) (hN : 0<c.N) (p : RegionName→Pointer) (o : Objects B c p)
    (before : Heap (Cell (Term α))) (seed : State (Term α))
    (x : Tensor.Inputs α B c.NH c.H c.D)
    (binding : Tensor.Binding c (sourceState p before seed) x)
    (bytes : InputBytes B c (sourceState p before seed))
    (hinj : Function.Injective (Tensor.offset (B:=B) c)) :
    ∃ arg, ScalarABI.specialize raw=some arg ∧
      ∃ states : Program B c.NH c.H→State (Term α),
        (∀ program, exec algebra (kernel c)
          (bindScale (Tensor.initialFor c (typedState p before seed) (anchor hD program))
            (Native.argument C arg))=some (states program)) ∧
        (∀ program region offset, (region≠c.Out ∨ ∀ j : TileIndex [64,c.D],
          rows (Tensor.initialFor c (typedState p before seed) (anchor hD program)) j.1<c.H →
            offset≠address c (Tensor.initialFor c (typedState p before seed) (anchor hD program)) j) →
          (states program).mem region offset=(typedState p before seed).mem region offset) ∧
        let returned : Returned (Cell (Term α)) := ⟨o.output,nativePost c o before states⟩
        (∀ i : Index B c.NH c.H c.D,
          Native.lower A c.fp8 c.D (lane i.channel.val) ((returned.read i).read .fp16)=
            Tensor.expected A C arg c x i) ∧
        (∀ location, (∀ i : Index B c.NH c.H c.D, location≠tensorAddress o.output i) →
          returned.backing location=before location) ∧
        (∀ t∈[o.query,o.key,o.value], ∀ location,
          t.allocationBase≤location → location<t.allocationBase+t.allocationBytes →
            returned.backing location=before location) ∧
        (∀ program j, rows (Tensor.initialFor c (typedState p before seed) (anchor hD program)) j.1<c.H →
          ∃ i : Index B c.NH c.H c.D, owner i=program ∧
            address c (Tensor.initialFor c (typedState p before seed) (anchor hD program)) j=Tensor.offset c i) ∧
        (∀ i j : Index B c.NH c.H c.D, tensorAddress o.output i=tensorAddress o.output j → owner i=owner j) := by
  obtain ⟨arg,harg⟩ := (ScalarABI.succeeds_iff raw).mpr admitted
  let s:=typedState p before seed
  have sb : Tensor.Binding c s x := typed_binding c p before seed x binding bytes
  have runs := fun program : Program B c.NH c.H =>
    values_and_frame c (Tensor.initialFor c s (anchor hD program)) (Native.argument C arg)
      (typed_range p before seed) (Tensor.block_injective c s hinj (anchor hD program))
  choose states he hv hf using runs
  refine ⟨arg,harg,states,he,hf,?_,?_,?_,?_,?_⟩
  · intro i
    change Native.lower A c.fp8 c.D (lane i.channel.val)
      ((nativePost c o before states (tensorAddress o.output i)).read .fp16)=_
    rw [nativePost,merge_read _ _ _ _ (output_injective c p o hinj)]
    have hr : rows (Tensor.initialFor c s i) (Tensor.laneIndex i).1<c.H := by
      rw [Tensor.row_eq]; exact i.query.isLt
    have vi:=hv (owner i) (Tensor.laneIndex i)
    rw [initial_owner c s hD i] at vi
    rw [←Tensor.address_eq c s i,vi hr,Cell.read_same]
    exact Native.output_binding A C arg c (Tensor.initialFor c s i) (Tensor.laneIndex i).1
      (x.at i) (Tensor.bindings c s x sb i) hr hN i.channel
  · intro location miss
    change merge before (tensorAddress o.output) owner
      (fun program i=>(states program).mem c.Out (Tensor.offset c i)) location=before location
    exact merge_frame _ _ _ _ location miss
  · intro t ht location hlo hhi
    change merge before (tensorAddress o.output) owner
      (fun program i=>(states program).mem c.Out (Tensor.offset c i)) location=before location
    apply merge_frame
    intro i
    have ho:=layout_bounds o.output (p c.Out) _ _ o.outputLayout i
    have hfr:=o.fresh t ht
    omega
  · intro program j hj
    have h0:=congrArg (fun q=>q.1.val) (owner_anchor hD program)
    change (anchor hD program).query.val/64=program.1.val at h0
    have hj' : program.1.val*64+j.1.val<c.H := by
      simpa only [rows,Tensor.initialFor,h0,↓reduceIte] using hj
    exact ⟨laneIndex c hD program j hj',owner_lane c hD program j hj',lane_address c s hD program j hj'⟩
  · intro i j hij
    exact congrArg owner (output_injective c p o hinj hij)

theorem empty_backing (raw : RawScalar) (c : Config) (o : Objects B c p)
    (before : Heap (Cell (Term α))) (states : Program B c.NH c.H→State (Term α))
    (empty : B*c.NH*c.H*c.D=0) :
    scalarABICheck Bool.false raw=Bool.true ∧ nativePost c o before states=before := by
  refine ⟨rfl,?_⟩
  funext location
  apply merge_frame
  intro i
  have hp:=Nat.mul_pos (Nat.mul_pos (Nat.mul_pos
    (Nat.zero_lt_of_lt i.batch.isLt) (Nat.zero_lt_of_lt i.head.isLt))
    (Nat.zero_lt_of_lt i.query.isLt)) (Nat.zero_lt_of_lt i.channel.isLt)
  omega

#axiomsClean project_half
#axiomsClean typed_binding
#axiomsClean native_backing
#axiomsClean empty_backing
#specNonCircular Tensor.expected avoiding [nativePost,merge,FP.Structural.exec,FP.Structural.run]
end
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.Backing
