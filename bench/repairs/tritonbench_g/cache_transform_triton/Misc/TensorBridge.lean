import bench.repairs.tritonbench_g.cache_transform_triton.Correct
import bench.repairs.tritonbench_g.cache_transform_triton.Misc.NativeCorrect
import bench.repairs.tritonbench_g.cache_transform_triton.Proofs.Single

/-! Optional paired Tensor applications and Python-check correspondence. -/
namespace VeriTile.Bench.Repairs.CacheTransform.TensorBridge
open VeriTile.Triton FP.Structural KernelBinding KernelResult
noncomputable section
set_option maxRecDepth 10000

/-- One actual gather, with its own heap, row table and output. Allocation
freshness concerns only this call's live cache and table; it imposes no
cross-call lifetime or allocator-reuse requirement. -/
theorem cache_gather_kernel_real (M : Algebra ℝ) (side : Single.Side)
    (r : Request) (h : InputContract r) (bytes : Bytes)
    (x : Cache (side.metadata r)) (table : Table r bytes) (z : Output r)
    (allocated : Single.Allocation side x h table z) (payload : Nat → ℝ) (s : State ℝ) :
    VeriTile.Spec.Real (∃ t,
      KernelSource.launch M (side.parameters r) (Single.realInitial side x h table s) (blocks r.count)=some t ∧
      realTensor z (Single.realAfter side z payload t)=
        mathematical r.mode r.naturalLengths (fun row => fun col : Fin r.width =>
          x.real payload (row*r.width+col.val)) ∧
      (∀ (region : RegionName) o,region≠(side.parameters r).output ∨ r.count≤o →
        t.mem region o=(Single.realInitial side x h table s).mem region o) ∧
      (∀ address,¬z.contains address → Single.realAfter side z payload t address=payload address) ∧
      (∀ o,o<4*(side.metadata r).numel →
        Single.realAfter side z payload t ((x.view (side.contiguous h)).base+o)=
          payload ((x.view (side.contiguous h)).base+o)) ∧
      (0<r.count → ∀ o,o<8*r.outputRows →
        Single.realAfter side z payload t (table.metadata.dataPtr.toNat+o)=
          payload (table.metadata.dataPtr.toNat+o))) := by
  obtain ⟨t,he,hv,hf⟩ := Single.complete M side r h
    (fun byte => (x.view (side.contiguous h)).address (byte/4)) table.read s table.values
  refine ⟨t,he,?_,hf,?_,?_,?_⟩
  · rw [Single.real_output]
    have observed := tensor_from_bytes (side.parameters r) t r.mode r.naturalLengths
      (fun byte => (x.view (side.contiguous h)).address (byte/4)) r.width .fp32 (realObserve payload)
      (by simp [Request.count,←outputRows_eq,Storage.bytes,Nat.mul_assoc]) side.rowBytes
      (by simpa only [Single.Side.count] using hv)
    rw [show observedTensor t (side.parameters r).output r.outputRows r.width .fp32
        (realObserve payload)=_ from (by simpa only [←outputRows_eq] using observed)]
    apply mathematical_congr r.mode r.naturalLengths (side.capacity r) (side.valid h)
    intro row hr
    funext col
    exact real_cache x (side.contiguous h) (side.shape h) payload row hr col
  · exact fun address outside => Single.realize_frame z payload _ address outside
  · exact fun o ho => Single.realize_frame z payload _ _ (allocated.cache o ho)
  · exact fun nonempty o ho => Single.realize_frame z payload _ _ (allocated.table nonempty o ho)

/-- Exact native FP32 result for either actual gather, independently of any
other invocation's heap or released temporaries. Every uint8 payload and FP32
word is admitted, and freshness is only for this call's live reads. -/
theorem cache_gather_kernel_fp32 (M : Algebra Unit) (side : Single.Side)
    (r : Request) (h : InputContract r) (bytes : Bytes)
    (x : Cache (side.metadata r)) (table : Table r bytes) (z : Output r)
    (allocated : Single.Allocation side x h table z) (s : State Unit) :
    ∃ t,
      KernelSource.launch M (side.parameters r) (Single.nativeInitial side x h table s) (blocks r.count)=some t ∧
      wordTensor z (Single.nativeAfter side z bytes t)=
        mathematical r.mode r.naturalLengths (fun row => fun col : Fin r.width =>
          x.word bytes (row*r.width+col.val)) ∧
      (∀ (region : RegionName) o,region≠(side.parameters r).output ∨ r.count≤o →
        t.mem region o=(Single.nativeInitial side x h table s).mem region o) ∧
      (∀ address,¬z.contains address → Single.nativeAfter side z bytes t address=bytes address) ∧
      (∀ o,o<4*(side.metadata r).numel →
        Single.nativeAfter side z bytes t ((x.view (side.contiguous h)).base+o)=
          bytes ((x.view (side.contiguous h)).base+o)) ∧
      (0<r.count → ∀ o,o<8*r.outputRows →
        Single.nativeAfter side z bytes t (table.metadata.dataPtr.toNat+o)=
          bytes (table.metadata.dataPtr.toNat+o)) := by
  obtain ⟨t,he,hv,hf⟩ := Single.complete M side r h
    (fun o => (bytes ((x.view (side.contiguous h)).base+o)).toNat) table.read s table.values
  refine ⟨t,he,?_,hf,?_,?_,?_⟩
  · rw [Single.native_output]
    have observed := tensor_from_bytes (side.parameters r) t r.mode r.naturalLengths
      (fun o => (bytes ((x.view (side.contiguous h)).base+o)).toNat) r.width .fp32 nativeObserve
      (by simp [Request.count,←outputRows_eq,Storage.bytes,Nat.mul_assoc]) side.rowBytes
      (by simpa only [Single.Side.count] using hv)
    rw [show observedTensor t (side.parameters r).output r.outputRows r.width .fp32
        nativeObserve=_ from (by simpa only [←outputRows_eq] using observed)]
    apply mathematical_congr r.mode r.naturalLengths (side.capacity r) (side.valid h)
    intro row hr
    funext col
    exact native_cache x (side.contiguous h) (side.noNeg h) (side.shape h) bytes row hr col
  · exact fun address outside => Single.realize_frame z bytes _ address outside
  · exact fun o ho => Single.realize_frame z bytes _ _ (allocated.cache o ho)
  · exact fun nonempty o ho => Single.realize_frame z bytes _ _ (allocated.table nonempty o ho)


/-- Complete execution of the unchanged byte-copy source in its symbolic Real
interpretation, followed by actual output Tensor coordinate readback. Symbols
name arbitrary Real input words; they are not asserted to be native uint8
encodings. The source copies those symbols without inspecting their values.
Native FP32 byte storage is independently covered by FPEquiv. This paired
corollary has the stronger Allocation premise that both tables remain live
and protected across both calls; the primary per-gather theorem is the general
kernel interface and does not impose that cross-call requirement. -/
theorem cache_transform_kernel_real (M : Algebra ℝ) (r : Request) (h : InputContract r)
    (bytes : Bytes) (x : Inputs r bytes) (cosTable sinTable : Table r bytes) (z : Outputs r)
    (allocated : Allocation x h cosTable sinTable z) (payload : Nat → ℝ) (s : State ℝ) :
    VeriTile.Spec.Real (∃ t,KernelSource.pair M r (realInitial x h cosTable sinTable s)=some t ∧
      realTensor z.cos (realAfter z payload t)=
        mathematical r.mode r.naturalLengths (fun row => fun col : Fin r.width =>
          x.cos.real payload (row*r.width+col.val)) ∧
      realTensor z.sin (realAfter z payload t)=
        mathematical r.mode r.naturalLengths (fun row => fun col : Fin r.width =>
          x.sin.real payload (row*r.width+col.val)) ∧
      (∀ (region : RegionName) o,(region≠"cos_output" ∨ r.count ≤ o) →
        (region≠"sin_output" ∨ r.count ≤ o) → t.mem region o=(realInitial x h cosTable sinTable s).mem region o) ∧
      (∀ address,¬z.cos.contains address → ¬z.sin.contains address →
        realAfter z payload t address=payload address) ∧
      (∀ o,o<4*r.cos.numel → realAfter z payload t ((x.cos.view (cos_contiguous h)).base+o)=
        payload ((x.cos.view (cos_contiguous h)).base+o)) ∧
      (∀ o,o<4*r.sin.numel → realAfter z payload t ((x.sin.view (sin_contiguous h)).base+o)=
        payload ((x.sin.view (sin_contiguous h)).base+o))) := by
  obtain ⟨t,he,hc,hs,hf⟩ := KernelSource.complete M r h
    (fun byte => (x.cos.view (cos_contiguous h)).address (byte/4))
    (fun byte => (x.sin.view (sin_contiguous h)).address (byte/4)) cosTable.read sinTable.read s cosTable.values sinTable.values
  have valid := admitted_lengths r h
  refine ⟨t,he,?_,?_,hf,?_,?_,?_⟩
  · rw [real_output_cos]
    have observed := tensor_from_bytes (KernelSource.cosine r) t r.mode r.naturalLengths
      (fun byte => (x.cos.view (cos_contiguous h)).address (byte/4)) r.width .fp32 (realObserve payload)
      (by simp [KernelSource.cosine,Request.count,←outputRows_eq,Storage.bytes,Nat.mul_assoc]) rfl hc
    rw [show observedTensor t "cos_output" r.outputRows r.width .fp32
        (realObserve payload)=_ from
      (by simpa only [←outputRows_eq] using observed)]
    apply mathematical_congr r.mode r.naturalLengths r.cosRows valid.1
    intro row hr
    funext col
    exact real_cache x.cos (cos_contiguous h) (cos_shape h) payload row hr col
  · rw [real_output_sin]
    have observed := tensor_from_bytes (KernelSource.sine r) t r.mode r.naturalLengths
      (fun byte => (x.sin.view (sin_contiguous h)).address (byte/4)) r.width .fp32 (realObserve payload)
      (by simp [KernelSource.sine,Request.count,←outputRows_eq,Storage.bytes,Nat.mul_assoc]) rfl hs
    rw [show observedTensor t "sin_output" r.outputRows r.width .fp32
        (realObserve payload)=_ from
      (by simpa only [←outputRows_eq] using observed)]
    apply mathematical_congr r.mode r.naturalLengths r.sinRows valid.2
    intro row hr
    funext col
    exact real_cache x.sin (sin_contiguous h) (sin_shape h) payload row hr col
  · exact fun address hc hs => realize_frame z payload _ _ address hc hs
  · exact fun o ho => realize_caller z payload _ _ _ (allocated.cos o ho)
  · exact fun o ho => realize_caller z payload _ _ _ (allocated.sin o ho)

theorem python_checks_iff_preconditions (r : Request) : checkInputs r=Bool.true ↔ InputContract r :=
  checkInputs_iff r

/-- The actual source succeeds on the actual input byte windows and created
row table, and the two actual output Tensors have exactly the independent
FP32 word-copy result. All bit patterns are included. No source variant,
floating arithmetic law, numerical experiment or whole-wrapper premise is
introduced. Exterior physical bytes and every exterior tagged cell are framed.
This is a conditional paired corollary: Allocation protects both row tables
through both calls. The primary per-gather entry permits independent
heaps and release/reuse of another invocation's temporary table. -/
theorem cache_transform_kernel_fp32 (M : Algebra Unit) (r : Request) (h : InputContract r)
    (bytes : Bytes) (x : Inputs r bytes) (cosTable sinTable : Table r bytes) (z : Outputs r)
    (allocated : Allocation x h cosTable sinTable z) (s : State Unit) :
    ∃ t,KernelSource.pair M r (nativeInitial x h cosTable sinTable s)=some t ∧
      wordTensor z.cos (nativeAfter z bytes t)=
        mathematical r.mode r.naturalLengths (fun row => fun col : Fin r.width =>
          x.cos.word bytes (row*r.width+col.val)) ∧
      wordTensor z.sin (nativeAfter z bytes t)=
        mathematical r.mode r.naturalLengths (fun row => fun col : Fin r.width =>
          x.sin.word bytes (row*r.width+col.val)) ∧
      (∀ (region : RegionName) o,(region≠"cos_output" ∨ r.count ≤ o) →
        (region≠"sin_output" ∨ r.count ≤ o) → t.mem region o=(nativeInitial x h cosTable sinTable s).mem region o) ∧
      (∀ address,¬z.cos.contains address → ¬z.sin.contains address →
        nativeAfter z bytes t address=bytes address) ∧
      (∀ o,o<4*r.cos.numel → nativeAfter z bytes t ((x.cos.view (cos_contiguous h)).base+o)=
        bytes ((x.cos.view (cos_contiguous h)).base+o)) ∧
      (∀ o,o<4*r.sin.numel → nativeAfter z bytes t ((x.sin.view (sin_contiguous h)).base+o)=
        bytes ((x.sin.view (sin_contiguous h)).base+o)) := by
  obtain ⟨t,he,hc,hs,hf⟩ := KernelSource.complete M r h (x.cosBytes h) (x.sinBytes h) cosTable.read sinTable.read s cosTable.values sinTable.values
  have valid := admitted_lengths r h
  refine ⟨t,he,?_,?_,hf,?_,?_,?_⟩
  · rw [native_output_cos]
    have observed := tensor_from_bytes (KernelSource.cosine r) t r.mode r.naturalLengths
      (x.cosBytes h) r.width .fp32 nativeObserve
      (by simp [KernelSource.cosine,Request.count,←outputRows_eq,Storage.bytes,Nat.mul_assoc]) rfl hc
    rw [show observedTensor t "cos_output" r.outputRows r.width .fp32 nativeObserve=_ from
      (by simpa only [←outputRows_eq] using observed)]
    apply mathematical_congr r.mode r.naturalLengths r.cosRows valid.1
    intro row hr
    funext col
    exact native_cache x.cos (cos_contiguous h) (cos_noNeg h) (cos_shape h) bytes row hr col
  · rw [native_output_sin]
    have observed := tensor_from_bytes (KernelSource.sine r) t r.mode r.naturalLengths
      (x.sinBytes h) r.width .fp32 nativeObserve
      (by simp [KernelSource.sine,Request.count,←outputRows_eq,Storage.bytes,Nat.mul_assoc]) rfl hs
    rw [show observedTensor t "sin_output" r.outputRows r.width .fp32 nativeObserve=_ from
      (by simpa only [←outputRows_eq] using observed)]
    apply mathematical_congr r.mode r.naturalLengths r.sinRows valid.2
    intro row hr
    funext col
    exact native_cache x.sin (sin_contiguous h) (sin_noNeg h) (sin_shape h) bytes row hr col
  · exact fun address hc hs => realize_frame z bytes _ _ address hc hs
  · exact fun o ho => realize_caller z bytes _ _ _ (allocated.cos o ho)
  · exact fun o ho => realize_caller z bytes _ _ _ (allocated.sin o ho)

#axiomsClean cache_gather_kernel_real
#axiomsClean cache_gather_kernel_fp32
#axiomsClean cache_transform_kernel_real
#axiomsClean cache_transform_kernel_fp32
#axiomsClean python_checks_iff_preconditions
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CacheTransform.TensorBridge
