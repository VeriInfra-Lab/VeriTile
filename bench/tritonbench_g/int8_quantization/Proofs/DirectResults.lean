import bench.tritonbench_g.int8_quantization.Proofs.Direct

namespace VeriTile.Bench.Repairs.Int8Quantization.Direct
open VeriTile.Triton FP.Structural Revision002 Words
noncomputable section
set_option maxHeartbeats 200000

/-- Original logical input bytes. The checked absent negative flag makes the
physical FP32 word its logical value; storage offsets are retained. -/
def inputValue (c : Request) (m : Memory) (i : Nat) : Word :=
  Bytes.read32 m (c.x.storage.address (c.x.physicalIndex i))

theorem input_pointer (c : Request) (m : Memory) (i : Nat) (hi : i < c.x.metadata.numel) :
    Bytes.read32 m ((view c).base+4*i) = inputValue c m i := by
  unfold inputValue
  rw [c.laws.contiguous c.contract.contiguous i hi]
  apply congrArg (Bytes.read32 m)
  simp only [view,VeriTile.Torch.FP32.DataView.base,VeriTile.Torch.FP32.dataView,
    if_neg (by omega : c.x.metadata.numel ≠ 0),VeriTile.Torch.FP32.Storage.address]
  omega

def InputLive (c : Request) (h : Heap) : Prop :=
  ∀ i, i < c.x.metadata.numel → ∀ b : Fin 4, h.live ((view c).base+4*i+b.val)

theorem input_preserved (c : Request) (before after : Heap) (ext : Extension before after)
    (live : InputLive c before) (i : Nat) (hi : i < c.x.metadata.numel) :
    inputValue c after.memory i = inputValue c before.memory i := by
  rw [←input_pointer c after.memory i hi,←input_pointer c before.memory i hi]
  funext b
  exact ext.frame _ (live i hi b)

theorem output_bytes (c : Request) (M : Algebra Word) (ops : Model.Primitives Word)
    (before : Heap) (out : Allocation c before) (w : Schedule.Program)
    (member : w∈Schedule.programs 0 (count c)) (i : Lane (p c)) (active : Grid.active (p c) w i) :
    (after c M ops before out).memory (out.output.base+Grid.offset (p c) w i) =
    BitVec.ofInt 8 (quantized M ops (Grid.parameters (p c) w) (width_positive c.x.metadata c.block)
      (Grid.atProgram (Bytes.bind (layout c out) before.memory) w) i) := by
  have bound := offset_bound c w member i active
  rw [numel_eq _ _ c.contract] at bound
  change Bytes.store (p c) (layout c out) _ _ ((layout c out).outputBase+_) = _
  rw [Bytes.store_output _ _ _ _ _ bound,typed_output c M ops _ w member i active]
  rfl

theorem scale_bytes (c : Request) (M : Algebra Word) (ops : Model.Primitives Word)
    (before : Heap) (out : Allocation c before) (w : Schedule.Program)
    (member : w∈Schedule.programs 0 (count c)) :
    Bytes.read32 (after c M ops before out).memory (out.scales.base+4*w.index) =
      if channels c.x.metadata=0 then M.literal (some .fp32) .real 0 else
      savedScale M ops (Grid.parameters (p c) w) (width_positive c.x.metadata c.block)
        (Grid.atProgram (Bytes.bind (layout c out) before.memory) w) := by
  have bound : w.index < (p c).batches*(p c).blocks := by
    simpa using (Schedule.program_bounds 0 (count c) w member).2.2
  change Bytes.read32 (Bytes.store (p c) (layout c out) _ _) ((layout c out).scaleBase+_) = _
  rw [Bytes.store_scale _ _ (allocationBounds c out) _ _ _ bound,typed_scale c M ops _ w member]
  rfl

theorem native_pointers (c : Request) (before : Heap) (out : Allocation c before)
    (positive : 0 < channels c.x.metadata) (w : Schedule.Program)
    (member : w∈Schedule.programs 0 (count c)) (i : Lane (p c)) :
    Bytes.nativeAccess (Grid.parameters (p c) w) (layout c out) w.pid i.1.val =
      Bytes.exactAccess (Grid.parameters (p c) w) (layout c out) w.pid i.1.val := by
  have wb := Schedule.program_bounds 0 (count c) w member
  have admitted : Native.Admitted (Grid.parameters (p c) w) := by
    have h := native_admitted c.x.metadata c.block c.contract c.query c.factor positive
    exact ⟨h.blocks,h.blockPositive,h.channelsPositive,h.covers,h.compiledWidth,h.inputAllocation,h.scaleAllocation⟩
  have bounds : Bytes.Bounds (Grid.parameters (p c) w) (layout c out) := by
    have h := allocationBounds c out
    exact ⟨h.input,h.output,h.scale,h.separated⟩
  exact Bytes.access_exact _ _ admitted bounds _ _ wb.1
    (by simpa [Schedule.Program.index] using wb.2.2) i.1.isLt

/-- Every caller coordinate is covered, independently of a numerical backend. -/
theorem coordinate_covered (c : Request) (batch row channel : Nat)
    (hb : batch < batches c.x.metadata) (hr : row < length c.x.metadata) (hc : channel < channels c.x.metadata) :
    ∃ w∈Schedule.programs 0 (count c), ∃ i : Lane (p c), Grid.active (p c) w i ∧
      Grid.offset (p c) w i = batch*length c.x.metadata*channels c.x.metadata+row*channels c.x.metadata+channel :=
  Grid.coordinate_covered (p c) c.contract.blockPositive (show 0 < channels c.x.metadata by omega) rfl
    (width_covers c.x.metadata c.block) batch row channel hb hr hc

structure PairReturned where
  q : Tensor
  qScale : Tensor
  k : Tensor
  kScale : Tensor
  heap : Heap

def pair (q k : Request) (M : Algebra Word) (ops : Model.Primitives Word) (before : Heap)
    (qa : Allocation q before) (ka : Allocation k (after q M ops before qa)) : Except InputError PairReturned := do
  checkTensorInputs q.x.metadata q.block
  let middle := after q M ops before qa
  checkTensorInputs k.x.metadata k.block
  return ⟨qa.output,qa.scales,ka.output,ka.scales,after k M ops middle ka⟩

#axiomsClean output_bytes
#axiomsClean scale_bytes
#axiomsClean native_pointers
end
end VeriTile.Bench.Repairs.Int8Quantization.Direct
