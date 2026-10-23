import bench.repairs.tritonbench_g.attention_llama.Contract

namespace VeriTile.Bench.Repairs.AttentionLlama.Checked
open VeriTile Triton
open VeriTile.Bench.TritonBenchG.AttentionLlama
open VeriTile.Bench.TritonBenchG.AttentionLlama.RealKernel
open LoopInvariant HostGrid HostMemory HostResult Correctness
set_option maxHeartbeats 1000000

/-- Every actual source access is bounded. Byte contents and all floating-point
correspondence hypotheses are absent from this structural theorem. -/
theorem access_safe (B : Nat) (c : Context) (Out : RegionName) (o : OutputLayout B c)
    (bounds : RegionBounds) (storage : Storage B c Out o bounds) (fp8 : Bool)
    (hmeta : metadataChecks c=Bool.true) (hu : ∀ rg n, c.initial.undef rg n=0)
    (i : Index B c.NH c.H c.D) :
    (if fp8 then ByteHostResult.kernel c Out o.soz o.soh o.som o.son
     else HostResult.kernel c Out o.soz o.soh o.som o.son).eraseDType.toAlgKernel.TraceSafe
        bounds (contextFor c i).initial := by
  have hm:=admitted_metadata c hmeta
  cases fp8
  · exact attention_fp16_access_safe bounds (contextFor c i) Out o.soz o.soh o.som o.son
      hm.1 hm.2.1 (by intro hc; simp only [contextFor] at hc ⊢; rw [hm.2.2 hc]; omega) hu
      (query_access B c Out o bounds storage i) (key_access B c Out o bounds storage i)
      (value_access B c Out o bounds storage i) (output_access B c Out o bounds storage i)
  · exact attention_e5m2_access_safe c.Q c.K c.V Out c.scale
      c.sqz c.sqh c.sqm c.sqk c.skz c.skh c.skn c.skk c.svz c.svh c.svk c.svn
      o.soz o.soh o.som o.son c.NH c.H c.N c.SP 64 64 c.D c.causal bounds
      (contextFor c i).initial hu (query_access B c Out o bounds storage i)
      (key_access B c Out o bounds storage i) (value_access B c Out o bounds storage i)
      (output_access B c Out o bounds storage i)

/-- The executed pointer and envelope checks apply to the actual strided
source expressions, including padded query/key lanes and final advances. -/
theorem formed_addresses (B : Nat) (c : Context) (o : OutputLayout B c)
    (fp8 : Bool) (p : Pointers B c o) (hl : launchChecks B c o=Bool.true)
    (hp : pointerChecks B c o fp8 p=Bool.true) (which : Buffer)
    (b h r d : Nat) (hb : b≤B) (hh : h≤c.NH) (hr : r≤bufferRows c which) (hd : d≤c.D) :
    let off := (bufferStrides o which).at b h r d
    (BitVec.ofInt 32 (off : Int)).toInt=(off : Int) ∧
      p.base which%width fp8 which=0 ∧ p.base which+width fp8 which*off<2^64 := by
  have hbound:=source_offset_bound B c o which b h r d hb hh hr hd
  exact ⟨formed_signed32 B c o hl _ hbound,pointer_formed B c o fp8 p hp _ _ hbound⟩

/-- Physical half-word output address is bound to the actual data pointer. -/
theorem returned_address (B : Nat) (c : Context) (o : OutputLayout B c)
    (fp8 : Bool) (p : Pointers B c o) (hp : pointerChecks B c o fp8 p=Bool.true)
    (i : Index B c.NH c.H c.D) :
    2*outputAddress c o.base o.soz o.soh o.som o.son i<2^64 := by
  have hbound:=source_offset_bound B c o .output i.batch.val i.head.val i.query.val i.channel.val
    (Nat.le_of_lt i.batch.isLt) (Nat.le_of_lt i.head.isLt)
    (by simp only [bufferRows]; omega) (Nat.le_of_lt i.channel.isLt)
  have hp':=pointer_formed B c o fp8 p hp .output _ hbound
  simp only [width] at hp'
  rw [outputAddress_eq,p.outputBinding,Nat.mul_add]
  have hdiv:=Nat.mod_add_div (p.base .output) 2
  have he : 2*(p.base .output/2)=p.base .output := by omega
  rw [he]
  exact hp'.2

/-- Checked Real result, physical stores, and complete complementary frame.
Tensor bindings and the fresh output layout are ordinary observations. -/
theorem attention_fp16 (B : Nat) (c : Context) (Out : RegionName) (o : OutputLayout B c)
    (q k v : TensorMetadata) (a : ScalarMetadata) (p : Pointers B c o)
    (hinput : inputChecks B c Bool.false q k v a=Bool.true) (hl : launchChecks B c o=Bool.true)
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
  have hm:=input_metadata B c Bool.false q k v a hinput
  have hd:=admitted_metadata c hm
  obtain ⟨result,hr,hf⟩:=attention_fp16_storage B c Out o x binding hd.1 hd.2.1 hd.2.2 hu encode memory
  exact ⟨result,hr,hf,returned_address B c o Bool.false p hp,
    access_safe B c Out o bounds storage Bool.false hm hu,formed_signed32 B c o hl⟩

/-- The E5M2 numerical domain is derived from the actual all-content check.
The physical frame retains raw input bytes, including signed zero. -/
theorem attention_e5m2 (B : Nat) (c : Context) (Out : RegionName) (o : OutputLayout B c)
    (q k v : TensorMetadata) (a : ScalarMetadata) (p : Pointers B c o)
    (hinput : inputChecks B c Bool.true q k v a=Bool.true) (hl : launchChecks B c o=Bool.true)
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
  have hm:=input_metadata B c Bool.true q k v a hinput
  have hd:=admitted_metadata c hm
  obtain ⟨result,hr,hf⟩:=attention_e5m2_storage B c Out o x binding
    (checked_bytes B c bytes observed hb) hd.1 hd.2.1 hd.2.2 hQK hQV hOK hOV hu encode memory
  exact ⟨result,hr,hf,returned_address B c o Bool.true p hp,
    access_safe B c Out o bounds storage Bool.true hm hu,formed_signed32 B c o hl⟩

#axiomsClean access_safe
#axiomsClean formed_addresses
#axiomsClean returned_address
#axiomsClean attention_fp16
#axiomsClean attention_e5m2
end VeriTile.Bench.Repairs.AttentionLlama.Checked
