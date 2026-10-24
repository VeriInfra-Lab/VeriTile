import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.Proofs.NativeBacking

namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.NativeGridResult
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.AttentionLlama HostGrid NativeSchedule
open Source Backing FPEquivalence
noncomputable section
set_option maxHeartbeats 4000000

/-- All physical source programs, native outputs and typed frames over raw
arrays and regional state. `Tensor.Inputs` and `Tensor.Binding` expand solely
to five input arrays and raw coordinate-read equalities, with no Tensor object,
API metadata, admission checks, or allocation result. -/
theorem attention_native_grid (B : Nat) (A : Target.Ops α) (C : ScalarABI.Conversions α)
    (arg : ScalarABI.Argument) (c : Config) (s : State (Term α))
    (_supported : VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.FPEquivalence.supportedHead c.D)
    (hD : 0<c.D) (hN : 0<c.N)
    (bytes : ByteRange s) (x : Tensor.Inputs α B c.NH c.H c.D)
    (binding : Tensor.Binding c s x) (hinj : Function.Injective (Tensor.offset (B:=B) c)) :
    ∃ states : Program B c.NH c.H→State (Term α),
      (∀ program, exec algebra (kernel c)
        (bindScale (Tensor.initialFor c s (anchor hD program)) (Native.argument C arg))=some (states program)) ∧
      (∀ i : Index B c.NH c.H c.D,
        Native.lower A c.fp8 c.D (lane i.channel.val)
          (((states (owner i)).mem c.Out (Tensor.offset c i)).read .fp16)=
          Tensor.expected A C arg c x i) ∧
      (∀ program region offset, (region≠c.Out ∨ ∀ j : TileIndex [64,c.D],
        rows (Tensor.initialFor c s (anchor hD program)) j.1<c.H →
          offset≠address c (Tensor.initialFor c s (anchor hD program)) j) →
        (states program).mem region offset=s.mem region offset) ∧
      (∀ program j, rows (Tensor.initialFor c s (anchor hD program)) j.1<c.H →
        ∃ i : Index B c.NH c.H c.D, owner i=program ∧
          address c (Tensor.initialFor c s (anchor hD program)) j=Tensor.offset c i) := by
  have runs := fun program : Program B c.NH c.H =>
    values_and_frame c (Tensor.initialFor c s (anchor hD program)) (Native.argument C arg)
      bytes (Tensor.block_injective c s hinj (anchor hD program))
  choose states he hv hf using runs
  refine ⟨states,he,?_,hf,?_⟩
  · intro i
    have hr : rows (Tensor.initialFor c s i) (Tensor.laneIndex i).1<c.H := by
      rw [Tensor.row_eq]; exact i.query.isLt
    have vi:=hv (owner i) (Tensor.laneIndex i)
    rw [initial_owner c s hD i] at vi
    rw [←Tensor.address_eq c s i,vi hr,Cell.read_same]
    exact Native.output_binding A C arg c (Tensor.initialFor c s i) (Tensor.laneIndex i).1
      (x.at i) (Tensor.bindings c s x binding i) hr hN i.channel
  · intro program j hj
    have h0:=congrArg (fun q=>q.1.val) (owner_anchor hD program)
    change (anchor hD program).query.val/64=program.1.val at h0
    have hj' : program.1.val*64+j.1.val<c.H := by
      simpa only [rows,Tensor.initialFor,h0,↓reduceIte] using hj
    exact ⟨laneIndex c hD program j hj',owner_lane c hD program j hj',lane_address c s hD program j hj'⟩

#axiomsClean attention_native_source
#axiomsClean attention_native_grid
#specNonCircular Tensor.expected avoiding [FP.Structural.exec,FP.Structural.run,Source.accumulated]
end
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.NativeGridResult
