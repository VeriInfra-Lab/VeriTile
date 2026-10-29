import bench.repairs.tritonbench_g.vector_addition_custom.Proofs.TensorBinding

/-! Successful ordinary empty_like + checked direct launch + returned Tensor.
Allocation laws describe only fresh aligned storage, metadata, and frame.
They contain no addition result or kernel execution assumption. -/
namespace VeriTile.Bench.Repairs.VectorAdditionCustom.Contiguous.Binding
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.AddExample.Finite
open Native
noncomputable section
set_option maxHeartbeats 4000000

def Separate (a b : Storage) : Prop :=
  a.base + 4*a.elements ≤ b.base ∨ b.base + 4*b.elements ≤ a.base

def SeparateViews (a b : DataView) : Prop :=
  a.base + 4*a.elements ≤ b.base ∨ b.base + 4*b.elements ≤ a.base

theorem separate_data (x out : Tensor) (lx : TensorLaws x) (lo : TensorLaws out)
    (cx : x.metadata.isContiguous = Bool.true) (co : out.metadata.isContiguous = Bool.true)
    (sep : Separate x.storage out.storage) :
    SeparateViews (dataView x lx cx) (dataView out lo co) := by
  by_cases hx0 : x.metadata.numel = 0
  · simp [SeparateViews, dataView, VeriTile.Torch.FP32.DataView.base, hx0]
  by_cases ho0 : out.metadata.numel = 0
  · simp [SeparateViews, dataView, VeriTile.Torch.FP32.DataView.base, ho0]
  have hx := VeriTile.Torch.FP32.contiguous_capacity x lx cx (Nat.pos_of_ne_zero hx0)
  have ho := VeriTile.Torch.FP32.contiguous_capacity out lo co (Nat.pos_of_ne_zero ho0)
  simp only [Separate, SeparateViews, dataView, VeriTile.Torch.FP32.DataView.base,
    VeriTile.Torch.FP32.Storage.address, if_neg hx0, if_neg ho0] at *
  omega

/-- The actual successful empty_like outcome. All laws are ordinary Tensor API
facts about that call, not a premise about the future kernel output.
[torch.empty_like](https://docs.pytorch.org/docs/stable/generated/torch.empty_like.html)
documents inherited shape/dtype/device and the default
`preserve_format` choice. Freshness, alignment, pointer binding and the byte
frame are explicit trusted allocator contracts; the API citation alone does
not prove those implementation properties. See add_example/LOAD_STORE_AUDIT.md. -/
structure EmptyLikeResult {M : Type} (ops : MemoryPrimitives M Word)
    (x y : Tensor) (h : InputContract x.metadata y.metadata) (before : M) where
  output : Tensor
  laws : TensorLaws output
  allocated : M
  metadata : output.metadata =
    {x.metadata with isNeg := Bool.false, dataPtr := output.dataPtr}
  alignment : 0 < output.metadata.numel → output.dataPtr.toNat % 4 = 0
  offset : output.storageOffset = 0
  freshX : Separate x.storage output.storage
  freshY : Separate y.storage output.storage
  freshObject : output.object ≠ x.object ∧ output.object ≠ y.object
  frame : Preserved ops (view (dataView output laws
    (by rw [metadata]; simpa only using h.contiguous.1)) 0) allocated before

theorem EmptyLikeResult.contiguous {M : Type} {ops : MemoryPrimitives M Word}
    {x y : Tensor} {h : InputContract x.metadata y.metadata} {before : M}
    (a : EmptyLikeResult ops x y h before) : a.output.metadata.isContiguous = Bool.true := by
  rw [a.metadata]; simpa only using h.contiguous.1

/-- Successful with torch.cuda.device(x.device) uses that device's pointers and
stable ordinary memory permissions. This says nothing about the kernel result. -/
structure Runtime {M : Type} (ops : MemoryPrimitives M Word) (x y : Tensor)
    (lx : TensorLaws x) (ly : TensorLaws y) (h : InputContract x.metadata y.metadata)
    (before : M) (a : EmptyLikeResult ops x y h before) : Prop where
  currentDevice : ops.launchDevice = x.metadata.device.index

def layout {M : Type} (ops : MemoryPrimitives M Word) (x y : Tensor)
    (lx : TensorLaws x) (ly : TensorLaws y) (h : InputContract x.metadata y.metadata)
    (before : M) (a : EmptyLikeResult ops x y h before) : Layout x.metadata.numel where
  window
    | .x => dataView x lx h.contiguous.1
    | .y => dataView y ly h.contiguous.2
    | .out => dataView a.output a.laws a.contiguous
  capacity := by
    intro r
    cases r
    · exact Nat.le_refl _
    · exact Nat.le_of_eq (same_numel x.metadata y.metadata h)
    · simp only [dataView, a.metadata, VeriTile.Torch.TensorMetadata.numel, le_refl]
  aligned := by
    intro r
    cases r
    · exact VeriTile.Torch.FP32.dataView_aligned x lx h.contiguous.1
        (fun nonempty => (h.alignment nonempty).1)
    · apply VeriTile.Torch.FP32.dataView_aligned y ly h.contiguous.2
      intro nonempty
      have hx : 0 < x.metadata.numel := by
        rw [same_numel x.metadata y.metadata h]
        exact nonempty
      exact (h.alignment hx).2
    · exact VeriTile.Torch.FP32.dataView_aligned a.output a.laws a.contiguous a.alignment
  freshX := separate_data x a.output lx a.laws h.contiguous.1 a.contiguous a.freshX
  freshY := separate_data y a.output ly a.laws h.contiguous.2 a.contiguous a.freshY

/-- Source host order on a successful empty_like call. Empty input bypasses the
launch. The returned object is exactly the allocated Tensor, never a copy. -/
def runWrapper {M : Type} (ops : MemoryPrimitives M Word) (compute : Word → Word → Word)
    (x y : Tensor) (lx : TensorLaws x) (ly : TensorLaws y)
    (h : InputContract x.metadata y.metadata) (before : M)
    (a : EmptyLikeResult ops x y h before) (programs : List Nat) :
    Except InputError (Option (Tensor × M)) := do
  checkInputs x.metadata y.metadata
  if x.metadata.numel = 0 then return some (a.output, a.allocated)
  return (runJobs ops compute (FP.Value.zero .fp32)
    (layout ops x y lx ly h before a) programs a.allocated).map (a.output, ·)

/-- Whole tensor result linked to entry-state logical values and the actual
returned object. Runtime facts are stated separately from InputContract. -/
theorem wrapper_correct {M : Type} (ops : MemoryPrimitives M Word)
    (compute : Word → Word → Word) (x y : Tensor)
    (lx : TensorLaws x) (ly : TensorLaws y)
    (h : InputContract x.metadata y.metadata) (before : M)
    (a : EmptyLikeResult ops x y h before)
    (memory : LocalMemory ops (layout ops x y lx ly h before a))
    (schedule : Index.Schedule x.metadata.numel) :
    ∃ final,
      runWrapper ops compute x y lx ly h before a schedule.programs = .ok (some (a.output, final)) ∧
      (∀ i, i < x.metadata.numel → logicalValue ops final a.output a.laws i =
        compute (logicalValue ops before x lx i) (logicalValue ops before y ly i)) ∧
      Preserved ops (view (dataView a.output a.laws a.contiguous) 0) final before := by
  let L := layout ops x y lx ly h before a
  have accepted := (check_tensor_arguments_iff x.metadata y.metadata).mpr h
  have outputContiguous : a.output.metadata.isContiguous = Bool.true := by
    rw [a.metadata]; exact h.contiguous.1
  have outputNeg : a.output.metadata.isNeg = Bool.false := by rw [a.metadata]
  have outputNumel : a.output.metadata.numel = x.metadata.numel := by
    simp [a.metadata, VeriTile.Torch.TensorMetadata.numel]
  by_cases empty : x.metadata.numel = 0
  · exact ⟨a.allocated, by simp [runWrapper, accepted, empty], by simp [empty], a.frame⟩
  · obtain ⟨final, execution, values, frame, _⟩ :=
      launch_correct ops compute (FP.Value.zero .fp32) L memory schedule a.allocated
    have fx := output_disjoint_x L 0
    have fy := output_disjoint_y L 0
    refine ⟨final, ?_, ?_, preserved_trans ops _ final a.allocated before frame a.frame⟩
    · simpa [runWrapper, accepted, empty, L, Index.jobs] using
        congrArg (fun r => (Except.ok (r.map (a.output, ·)) :
          Except InputError (Option (Tensor × M)))) execution
    · intro i hi
      rw [←direct_read_eq_logical ops a.output a.laws final outputContiguous outputNeg i
        (by rw [outputNumel]; exact hi)]
      change read ops L final .out i = _
      rw [values i hi]
      have px := input_preserved ops (view (L.window .x) 0)
        (view (L.window .out) 0) a.allocated before a.frame fx _ (inside L .x i hi 0)
      have py := input_preserved ops (view (L.window .y) 0)
        (view (L.window .out) 0) a.allocated before a.frame fy _ (inside L .y i hi 0)
      simp only [view] at px py
      unfold result Native.read
      rw [px, py]
      exact congrArg₂ compute (checked_read_x ops x y lx h before i hi)
        (checked_read_y ops x y ly h before i hi)

#axiomsClean wrapper_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.VectorAdditionCustom.Contiguous.Binding
