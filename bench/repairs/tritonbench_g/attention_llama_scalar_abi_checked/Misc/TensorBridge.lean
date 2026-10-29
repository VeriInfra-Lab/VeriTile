import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.Proofs.RealGridResult
import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.Misc.NativeGridResult
import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.Proofs.RealBacking
import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.Proofs.TensorObservation
import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.Contract
import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.Proofs.ScalarABI

/-! Optional checked FP16/E5M2 Tensor storage, access and address applications. -/
namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.TensorBridge
open VeriTile Triton
open VeriTile.Bench.Repairs.AttentionLlama
open VeriTile.Bench.TritonBenchG.AttentionLlama
open VeriTile.Bench.TritonBenchG.AttentionLlama.RealKernel
open LoopInvariant HostGrid HostMemory HostResult Correctness Checked
set_option maxHeartbeats 1000000

/-- Checked Real result, physical stores, and complete complementary frame.
Tensor bindings and the fresh output layout are ordinary observations. -/
theorem attention_fp16 (B : Nat) (c : Context) (Out : RegionName) (o : OutputLayout B c)
    (q k v : TensorMetadata) (a : ScalarMetadata) (p : Pointers B c o)
    (raw : RawScalar) (_scaleBinding : ScalarABI.realValue raw=some c.scale)
    (hinput : accepted B c Bool.false q k v a raw=Bool.true) (hl : launchChecks B c o=Bool.true)
    (hp : pointerChecks B c o Bool.false p=Bool.true)
    (bounds : RegionBounds) (storage : Storage B c Out o bounds)
    (x : Correctness.TensorInputs B c.NH c.H c.D) (binding : Correctness.TensorBinding c x)
    (hu : ∀ rg n, c.initial.undef rg n=0) {Cell : Type} (encode : ℝ→Cell) (memory : Nat→Cell) :
    ∃ result : HostResult.Results (B:=B) (heads:=c.NH) c Out o.soz o.soh o.som o.son,
      let address := outputAddress (B:=B) c o.base o.soz o.soh o.som o.son
      (∀ i, (returnTensor address (storeAll address (fun i=>encode (HostResult.word result i))
        memory (schedule B c.NH c.H c.D))).read i=encode (attention c x i)) ∧
      (∀ location, (∀ i, location≠address i) →
        storeAll address (fun i=>encode (HostResult.word result i)) memory
          (schedule B c.NH c.H c.D) location=memory location) ∧
      (∀ i, 2*address i<2^64) ∧
      (∀ i : Index B c.NH c.H c.D, (HostResult.kernel c Out o.soz o.soh o.som o.son).eraseDType.toAlgKernel.TraceSafe
        bounds (contextFor c i).initial) ∧
      (∀ n : Nat, n<envelope B c o → (BitVec.ofInt 32 (n : Int)).toInt=(n : Int)) := by
  have hm:=input_metadata B c Bool.false q k v a (accepted_original B c Bool.false q k v a raw hinput)
  have hd:=admitted_metadata c hm
  obtain ⟨result,hr,hf⟩:=attention_fp16_storage B c Out o x binding hd.1 hd.2.1 hd.2.2 hu encode memory
  exact ⟨result,hr,hf,returned_address B c o Bool.false p hp,
    access_safe B c Out o bounds storage Bool.false hm hu,formed_signed32 B c o hl⟩

/-- The E5M2 numerical domain is derived from the actual all-content check.
The physical frame retains raw input bytes, including signed zero. -/
theorem attention_e5m2 (B : Nat) (c : Context) (Out : RegionName) (o : OutputLayout B c)
    (q k v : TensorMetadata) (a : ScalarMetadata) (p : Pointers B c o)
    (raw : RawScalar) (_scaleBinding : ScalarABI.realValue raw=some c.scale)
    (hinput : accepted B c Bool.true q k v a raw=Bool.true) (hl : launchChecks B c o=Bool.true)
    (hp : pointerChecks B c o Bool.true p=Bool.true)
    (bounds : RegionBounds) (storage : Storage B c Out o bounds)
    (bytes : Bytes B c) (observed : ByteBinding B c bytes) (hb : byteChecks B c bytes=Bool.true)
    (x : Correctness.TensorInputs B c.NH c.H c.D) (binding : Correctness.TensorBinding (FP8Kernel.view c) x)
    (hQK : c.Q≠c.K) (hQV : c.Q≠c.V) (hOK : Out≠c.K) (hOV : Out≠c.V)
    (hu : ∀ rg n, c.initial.undef rg n=0) {Cell : Type} (encode : ℝ→Cell) (memory : Nat→Cell) :
    ∃ result : ByteHostResult.Results (B:=B) (heads:=c.NH) c Out o.soz o.soh o.som o.son,
      let address := outputAddress (B:=B) c o.base o.soz o.soh o.som o.son
      (∀ i, (returnTensor address (storeAll address (fun i=>encode (ByteHostResult.word result i))
        memory (schedule B c.NH c.H c.D))).read i=encode (attention c x i)) ∧
      (∀ location, (∀ i, location≠address i) →
        storeAll address (fun i=>encode (ByteHostResult.word result i)) memory
          (schedule B c.NH c.H c.D) location=memory location) ∧
      (∀ i, 2*address i<2^64) ∧
      (∀ i : Index B c.NH c.H c.D, (ByteHostResult.kernel c Out o.soz o.soh o.som o.son).eraseDType.toAlgKernel.TraceSafe
        bounds (contextFor c i).initial) ∧
      (∀ n : Nat, n<envelope B c o → (BitVec.ofInt 32 (n : Int)).toInt=(n : Int)) := by
  have hm:=input_metadata B c Bool.true q k v a (accepted_original B c Bool.true q k v a raw hinput)
  have hd:=admitted_metadata c hm
  obtain ⟨result,hr,hf⟩:=attention_e5m2_storage B c Out o x binding
    (checked_bytes B c bytes observed hb) hd.1 hd.2.1 hd.2.2 hQK hQV hOK hOV hu encode memory
  exact ⟨result,hr,hf,returned_address B c o Bool.true p hp,
    access_safe B c Out o bounds storage Bool.true hm hu,formed_signed32 B c o hl⟩

#axiomsClean attention_fp16
#axiomsClean attention_e5m2
open FP.Structural Source Backing NativeSchedule
noncomputable section

theorem attention_tensor_backing (raw : RawScalar) (c : Context) (Out : RegionName)
    (soz soh som son : Nat) (fp8 : Bool)
    (_scale : ScalarABI.realValue raw=some c.scale)
    (hD : 0<c.D) (hN : 0<c.N) (hSP : c.causal=Bool.true → c.SP=0)
    (hu : ∀ r offset,c.initial.undef r offset=0)
    (p : RegionName→Pointer) (o : Objects B (realConfig c Out soz soh som son fp8) p)
    (before : Heap MemCell)
    (x : HostResult.Inputs B c.NH c.H c.D)
    (binding : HostResult.Binding (realView (realContext c p before) fp8) x)
    (domain : fp8=Bool.true → ByteHostResult.Domain (B:=B) (heads:=c.NH) (realContext c p before))
    (sep : fp8=Bool.true → c.Q≠c.K ∧ c.Q≠c.V ∧ Out≠c.K ∧ Out≠c.V)
    (hinj : Function.Injective (Tensor.offset (B:=B) (realConfig c Out soz soh som son fp8))) :
    ∃ states : Program B c.NH c.H→BlockState,
      (∀ program, exec (realSource c Out soz soh som son fp8).eraseDType.toAlgKernel
        (contextFor (realContext c p before) (anchor hD program)).initial=some (states program)) ∧
      (∀ program region offset, (region≠Out ∨ ∀ j : TileIndex [64,c.D],
        alActive (contextFor (realContext c p before) (anchor hD program)).initial c.H 64 j →
          offset≠alOutOffset (contextFor (realContext c p before) (anchor hD program)).initial
            c.NH soz soh som son 64 j) →
        (states program).mem region offset=before ((p region).address offset)) ∧
      let returned : Returned MemCell := ⟨o.output,realPost c Out soz soh som son fp8 o before states⟩
      (∀ i : Index B c.NH c.H c.D, realRead (returned.read i)=HostResult.expected c x i) ∧
      (∀ location, (∀ i : Index B c.NH c.H c.D,location≠tensorAddress o.output i) →
        returned.backing location=before location) ∧
      (∀ t∈[o.query,o.key,o.value], ∀ location,
        t.allocationBase≤location → location<t.allocationBase+t.allocationBytes →
          returned.backing location=before location) ∧
      (∀ program (j : TileIndex [64,c.D]), alActive (contextFor (realContext c p before) (anchor hD program)).initial c.H 64 j →
        ∃ i : Index B c.NH c.H c.D, owner i=program ∧
          alOutOffset (contextFor (realContext c p before) (anchor hD program)).initial c.NH soz soh som son 64 j=
            Tensor.offset (realConfig c Out soz soh som son fp8) i) ∧
      (∀ i j : Index B c.NH c.H c.D,tensorAddress o.output i=tensorAddress o.output j → owner i=owner j) := by
  exact Backing.real_backing raw c Out soz soh som son fp8 _scale hD hN hSP hu p o before x binding domain sep hinj

theorem attention_native_tensor (A : Target.Ops α) (C : ScalarABI.Conversions α)
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
  exact Backing.native_backing A C raw admitted c _supported hD hN p o before seed x binding bytes hinj

theorem attention_empty (raw : RawScalar) (c : Config) (o : Objects B c p)
    (before : Heap (Cell (Term α))) (states : Program B c.NH c.H→State (Term α))
    (empty : B*c.NH*c.H*c.D=0) :
    scalarABICheck Bool.false raw=Bool.true ∧ nativePost c o before states=before := by
  exact Backing.empty_backing raw c o before states empty

#axiomsClean attention_tensor_backing
#axiomsClean attention_native_tensor
#axiomsClean attention_empty
end
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.TensorBridge
