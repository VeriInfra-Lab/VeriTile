import bench.repairs.tritonbench_g.attention_llama.Kernels
import bench.repairs.tritonbench_g.attention_llama.Proofs.ByteHostResult
import bench.repairs.tritonbench_g.attention_llama.Proofs.AccessSafety
import bench.repairs.tritonbench_g.attention_llama.Proofs.ByteSafety
import VeriTile.Meta.StatementAudit

/-! Repaired attention_llama. Q and Out are FP16; K/V are FP16 or signed
int8 E5M2 encodings. The mathematical interpretation erases arithmetic rounding,
but retains E5M2 decoding. Native precision is specified independently in FPEquiv.

Production BLOCK_M = BLOCK_N = 64; head dimensions are 16, 32, 64, 128.
N need not be divisible by 64. Logical keys are k < N, additionally
k + start_position ≤ query in causal mode. A nonempty complete causal grid
has start_position = 0 under this source convention; prefix-cache semantics
are not substituted. N > 0 and nonempty visible rows are the real domain.

The source theorems derive successful execution, actual output stores, and
all complementary memory cells. Tensor interfaces bind coordinates only;
fresh output layout, launch, compiler correctness and readback are ordinary
library/runtime contracts. No attention-output equation is an interface premise.
-/
namespace VeriTile.Bench.Repairs.AttentionLlama.Correctness
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.AttentionLlama
open VeriTile.Bench.TritonBenchG.AttentionLlama.RealKernel
open FP8RealLoop (BytesFrom)
open LoopInvariant HostGrid HostMemory MaskCoverage HostResult
set_option maxHeartbeats 1000000

abbrev TensorInputs := HostResult.Inputs
abbrev TensorBinding {B heads : Nat} (c : Context) (x : TensorInputs B heads c.H c.D) :=
  HostResult.Binding c x
noncomputable abbrev attention {B heads : Nat} (c : Context) (x : TensorInputs B heads c.H c.D) :=
  HostResult.expected c x

theorem attention_fp16_source (c : Context) (Out : RegionName) (soz soh som son : Nat)
    (hD : 0 < c.D) (hN : 0 < c.N)
    (hSP : c.causal = Bool.true → c.SP ≤ c.initial.pids 0 * 64)
    (hout : Postamble.ActiveInjective (c.initial.pids 0 * 64) c.H 64 c.D som son)
    (hundef : ∀ rg o, c.initial.undef rg o = 0) :
    ∃ sf, exec (TypedSource.repaired c.Q c.K c.V Out c.scale
      c.sqz c.sqh c.sqm c.sqk c.skz c.skh c.skn c.skk c.svz c.svh c.svk c.svn
      soz soh som son c.NH c.H c.N c.SP 64 64 c.D c.causal).eraseDType.toAlgKernel c.initial = some sf ∧
      (∀ idx : TileIndex [64,c.D], alActive c.initial c.H 64 idx →
        sf.readMem Out (alOutOffset c.initial c.NH soz soh som son 64 idx) = RealKernel.target c idx) ∧
      (∀ r o, (r ≠ Out ∨ ∀ idx : TileIndex [64,c.D], alActive c.initial c.H 64 idx →
        o ≠ alOutOffset c.initial c.NH soz soh som son 64 idx) → sf.mem r o = c.initial.mem r o) := by
  exact RealKernel.execution c Out soz soh som son hD hN hSP hout hundef

theorem attention_e5m2_source (c : Context) (Out : RegionName) (soz soh som son : Nat)
    (hD : 0 < c.D) (hN : 0 < c.N)
    (hSP : c.causal = Bool.true → c.SP ≤ c.initial.pids 0 * 64)
    (hout : Postamble.ActiveInjective (c.initial.pids 0 * 64) c.H 64 c.D som son)
    (hQK : c.Q ≠ c.K) (hQV : c.Q ≠ c.V) (hOK : Out ≠ c.K) (hOV : Out ≠ c.V)
    (hundef : ∀ rg o, c.initial.undef rg o = 0)
    (hK : BytesFrom c.K c.initial (alBase c.initial c.NH c.skz c.skh) c.skn c.skk 0
      (FP8RealSetup.stop c.initial c.N c.SP 64 c.causal) c.N 64 c.D)
    (hV : BytesFrom c.V c.initial (alBase c.initial c.NH c.svz c.svh) c.svk c.svn 0
      (FP8RealSetup.stop c.initial c.N c.SP 64 c.causal) c.N 64 c.D) :
    ∃ sf, exec (TypedSource.repairedFP8 c.Q c.K c.V Out c.scale
      c.sqz c.sqh c.sqm c.sqk c.skz c.skh c.skn c.skk c.svz c.svh c.svk c.svn
      soz soh som son c.NH c.H c.N c.SP 64 64 c.D c.causal).eraseDType.toAlgKernel c.initial = some sf ∧
      (∀ idx : TileIndex [64,c.D], alActive c.initial c.H 64 idx →
        sf.readMem Out (alOutOffset c.initial c.NH soz soh som son 64 idx) =
          VeriTile.Bench.Repairs.AttentionLlama.RealKernel.target (FP8Kernel.view c) idx) ∧
      (∀ r o, (r ≠ Out ∨ ∀ idx : TileIndex [64,c.D], alActive c.initial c.H 64 idx →
        o ≠ alOutOffset c.initial c.NH soz soh som son 64 idx) →
          sf.mem r o = c.initial.mem r o) := by
  exact FP8Kernel.execution c Out soz soh som son hD hN hSP hout hQK hQV hOK hOV hundef hK hV

def outputAddress {B : Nat} (c : Context) (base soz soh som son : Nat)
    (i : Index B c.NH c.H c.D) : Nat :=
  base + alOutOffset (contextFor c i).initial c.NH soz soh som son 64
    (lane i,i.channel,PUnit.unit)

/-- The fresh returned tensor supplies its actual strides and allocation base.
Injectivity is an allocator/layout property; it contains no numeric result. -/
structure OutputLayout (B : Nat) (c : Context) where
  base : Nat
  soz : Nat
  soh : Nat
  som : Nat
  son : Nat
  injective : Function.Injective (outputAddress (B := B) c base soz soh som son)


/-- Source rematerialization gives exactly the returned tensor coordinate. -/
theorem outputAddress_eq {B : Nat} (c : Context) (base soz soh som son : Nat)
    (i : Index B c.NH c.H c.D) :
    outputAddress c base soz soh som son i =
      base + (i.batch.val*soz + i.head.val*soh + i.query.val*som + i.channel.val*son) := by
  have hh := MaskCoverage.batch_head_roundtrip c.NH i.batch.val i.head.val i.head.isLt
  have hq : i.query.val/64*64+i.query.val%64 = i.query.val := by omega
  simp only [outputAddress,contextFor,lane,alOutOffset,alBase,alRow,Nat.one_ne_zero,↓reduceIte,hh.1,hh.2,hq]

/-- Active block injectivity follows from the ordinary logical output layout.
No condition is imposed on addresses of masked-off padded lanes. -/
theorem OutputLayout.blockInjective {B : Nat} {c : Context} (layout : OutputLayout B c)
    (i : Index B c.NH c.H c.D) :
    Postamble.ActiveInjective ((contextFor c i).initial.pids 0 * 64) c.H 64 c.D
      layout.som layout.son := by
  intro a b ha hb he
  let ia : Index B c.NH c.H c.D :=
    ⟨i.batch,i.head,⟨(contextFor c i).initial.pids 0*64+a.1.val,ha⟩,a.2.1⟩
  let ib : Index B c.NH c.H c.D :=
    ⟨i.batch,i.head,⟨(contextFor c i).initial.pids 0*64+b.1.val,hb⟩,b.2.1⟩
  have hi : ia = ib := layout.injective (by
    rw [outputAddress_eq,outputAddress_eq]
    change layout.base + (i.batch.val*layout.soz+i.head.val*layout.soh+
      ((contextFor c i).initial.pids 0*64+a.1.val)*layout.som+a.2.1.val*layout.son) = _
    simp only [Nat.add_assoc] at he ⊢
    rw [he])
  have hq := congrArg (fun j => j.query.val) hi
  have hd := congrArg (fun j => j.channel) hi
  apply Prod.ext
  · apply Fin.ext
    exact Nat.add_left_cancel hq
  · exact Prod.ext hd (Subsingleton.elim _ _)

/-- The complete grid has source executions and returns the independently
specified logical tensor. The frame covers every cell outside its allocation. -/
theorem attention_fp16_tensor (B : Nat) (c : Context) (Out : RegionName)
    (layout : OutputLayout B c) (x : TensorInputs B c.NH c.H c.D)
    (binding : TensorBinding c x)
    (hD : 0 < c.D) (hN : 0 < c.N) (hSP : c.causal = Bool.true → c.SP = 0)
    (hu : ∀ rg o, c.initial.undef rg o = 0) (memory : Nat → ℝ) :
    ∃ result : HostResult.Results (B := B) (heads := c.NH) c Out
        layout.soz layout.soh layout.som layout.son,
      let address := outputAddress (B := B) c layout.base layout.soz layout.soh layout.som layout.son
      (∀ i, (returnTensor address (storeAll address (HostResult.word result) memory
        (schedule B c.NH c.H c.D))).read i = attention c x i) ∧
      (∀ location, (∀ i, location ≠ address i) →
        storeAll address (HostResult.word result) memory (schedule B c.NH c.H c.D) location =
          memory location) := by
  obtain ⟨result⟩ := HostResult.results_exist c Out layout.soz layout.soh layout.som layout.son
    hD hN hSP layout.blockInjective hu
  exact ⟨result,HostResult.returned_values_and_frame c Out layout.soz layout.soh layout.som layout.son
    x binding result hD hN hSP layout.blockInjective hu _ layout.injective memory⟩

/-- E5M2 uses the raw-byte source executions. Coordinate bindings decode K/V
for the mathematical target; original byte cells are protected by the source
frame in attention_e5m2_source, including signed zero. -/
theorem attention_e5m2_tensor (B : Nat) (c : Context) (Out : RegionName)
    (layout : OutputLayout B c) (x : TensorInputs B c.NH c.H c.D)
    (binding : TensorBinding (FP8Kernel.view c) x)
    (domain : ByteHostResult.Domain (B := B) (heads := c.NH) c)
    (hD : 0 < c.D) (hN : 0 < c.N) (hSP : c.causal = Bool.true → c.SP = 0)
    (hQK : c.Q ≠ c.K) (hQV : c.Q ≠ c.V) (hOK : Out ≠ c.K) (hOV : Out ≠ c.V)
    (hu : ∀ rg o, c.initial.undef rg o = 0) (memory : Nat → ℝ) :
    ∃ result : ByteHostResult.Results (B := B) (heads := c.NH) c Out
        layout.soz layout.soh layout.som layout.son,
      let address := outputAddress (B := B) c layout.base layout.soz layout.soh layout.som layout.son
      (∀ i, (returnTensor address (storeAll address (ByteHostResult.word result) memory
        (schedule B c.NH c.H c.D))).read i = attention c x i) ∧
      (∀ location, (∀ i, location ≠ address i) →
        storeAll address (ByteHostResult.word result) memory (schedule B c.NH c.H c.D) location =
          memory location) := by
  obtain ⟨result⟩ := ByteHostResult.results_exist c Out layout.soz layout.soh layout.som layout.son
    domain hD hN hSP layout.blockInjective hQK hQV hOK hOV hu
  exact ⟨result,ByteHostResult.returned_values_and_frame c Out layout.soz layout.soh layout.som layout.son
    x binding domain result hD hN hSP layout.blockInjective hQK hQV hOK hOV hu _ layout.injective memory⟩

theorem attention_fp16_access_safe (bounds : RegionBounds) (c : Context) (Out : RegionName) (soz soh som son : Nat)
    (hD : 0 < c.D) (hN : 0 < c.N)
    (hSP : c.causal = Bool.true → c.SP ≤ c.initial.pids 0 * 64)
    (hundef : ∀ rg o, c.initial.undef rg o = 0)
    (hQ : ∀ idx : TileIndex [64,c.D], alActive c.initial c.H 64 idx →
      alBase c.initial c.NH c.sqz c.sqh+(c.initial.pids 0*64+idx.1.val)*c.sqm+idx.2.1.val*c.sqk < bounds c.Q)
    (hK : ∀ k, k < c.N → ∀ d : Fin c.D,
      alBase c.initial c.NH c.skz c.skh+k*c.skn+d.val*c.skk < bounds c.K)
    (hV : ∀ k, k < c.N → ∀ d : Fin c.D,
      alBase c.initial c.NH c.svz c.svh+k*c.svk+d.val*c.svn < bounds c.V)
    (hO : ∀ idx : TileIndex [64,c.D], alActive c.initial c.H 64 idx →
      alOutOffset c.initial c.NH soz soh som son 64 idx < bounds Out) :
    (TypedSource.repaired c.Q c.K c.V Out c.scale c.sqz c.sqh c.sqm c.sqk
      c.skz c.skh c.skn c.skk c.svz c.svh c.svk c.svn soz soh som son
      c.NH c.H c.N c.SP 64 64 c.D c.causal).eraseDType.toAlgKernel.TraceSafe bounds c.initial := by
  exact AccessSafety.complete bounds c Out soz soh som son hD hN hSP hundef hQ hK hV hO

/-- Byte-source address safety has no finite-byte or numerical premise. -/
theorem attention_e5m2_access_safe
    (Q K V Out : RegionName) (scale : ℝ)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD : Nat) (causal : Bool) (bounds : RegionBounds) (s : BlockState)
    (hu : ∀ r o, s.undef r o = 0)
    (hQ : ∀ i : TileIndex [BM,BD], alActive s H BM i →
      alBase s NH sqz sqh+alRow s BM i.1*sqm+i.2.1.val*sqk < bounds Q)
    (hK : ∀ k, k < N → ∀ d : Fin BD, alBase s NH skz skh+k*skn+d.val*skk < bounds K)
    (hV : ∀ k, k < N → ∀ d : Fin BD, alBase s NH svz svh+k*svk+d.val*svn < bounds V)
    (hO : ∀ i : TileIndex [BM,BD], alActive s H BM i →
      alOutOffset s NH soz soh som son BM i < bounds Out) :
    (TypedSource.repairedFP8 Q K V Out scale sqz sqh sqm sqk skz skh skn skk
      svz svh svk svn soz soh som son NH H N SP BM BN BD causal).eraseDType.toAlgKernel.TraceSafe bounds s := by
  exact ByteSafety.source_safe Q K V Out scale sqz sqh sqm sqk skz skh skn skk
    svz svh svk svn soz soh som son NH H N SP BM BN BD causal bounds s hu hQ
    (by intro k hk j d hj; exact hK _ hj d)
    (by intro k hk j d hj; exact hV _ hj d) hO

/-- The output codec is an ordinary representation contract. The frame is
on opaque storage cells, so untouched raw bytes retain their exact encoding. -/
theorem attention_fp16_storage (B : Nat) (c : Context) (Out : RegionName)
    (layout : OutputLayout B c) (x : TensorInputs B c.NH c.H c.D)
    (binding : TensorBinding c x)
    (hD : 0 < c.D) (hN : 0 < c.N) (hSP : c.causal = Bool.true → c.SP = 0)
    (hu : ∀ rg o, c.initial.undef rg o = 0)  {Cell : Type} (encode : ℝ → Cell) (memory : Nat → Cell) :
    ∃ result : HostResult.Results (B := B) (heads := c.NH) c Out
        layout.soz layout.soh layout.som layout.son,
      let address := outputAddress (B := B) c layout.base layout.soz layout.soh layout.som layout.son
      (∀ i, (returnTensor address (storeAll address (fun i => encode (HostResult.word result i)) memory
        (schedule B c.NH c.H c.D))).read i = encode (attention c x i)) ∧
      (∀ location, (∀ i, location ≠ address i) →
        storeAll address (fun i => encode (HostResult.word result i)) memory (schedule B c.NH c.H c.D) location =
          memory location) := by
  obtain ⟨result⟩ := HostResult.results_exist c Out layout.soz layout.soh layout.som layout.son
    hD hN hSP layout.blockInjective hu
  let address := outputAddress (B := B) c layout.base layout.soz layout.soh layout.som layout.son
  obtain ⟨hv,hf⟩ := logical_result_and_frame address
    (fun i => encode (HostResult.word result i)) layout.injective memory
  refine ⟨result,?_,hf⟩
  intro i
  rw [hv]
  exact congrArg encode (HostResult.word_correct c Out layout.soz layout.soh layout.som layout.son
    x binding result hD hN hSP layout.blockInjective hu i)

/-- The output codec is an ordinary representation contract. The frame is
on opaque storage cells, so untouched raw bytes retain their exact encoding. -/
theorem attention_e5m2_storage (B : Nat) (c : Context) (Out : RegionName)
    (layout : OutputLayout B c) (x : TensorInputs B c.NH c.H c.D)
    (binding : TensorBinding (FP8Kernel.view c) x)
    (domain : ByteHostResult.Domain (B := B) (heads := c.NH) c)
    (hD : 0 < c.D) (hN : 0 < c.N) (hSP : c.causal = Bool.true → c.SP = 0)
    (hQK : c.Q ≠ c.K) (hQV : c.Q ≠ c.V) (hOK : Out ≠ c.K) (hOV : Out ≠ c.V)
    (hu : ∀ rg o, c.initial.undef rg o = 0)  {Cell : Type} (encode : ℝ → Cell) (memory : Nat → Cell) :
    ∃ result : ByteHostResult.Results (B := B) (heads := c.NH) c Out
        layout.soz layout.soh layout.som layout.son,
      let address := outputAddress (B := B) c layout.base layout.soz layout.soh layout.som layout.son
      (∀ i, (returnTensor address (storeAll address (fun i => encode (ByteHostResult.word result i)) memory
        (schedule B c.NH c.H c.D))).read i = encode (attention c x i)) ∧
      (∀ location, (∀ i, location ≠ address i) →
        storeAll address (fun i => encode (ByteHostResult.word result i)) memory (schedule B c.NH c.H c.D) location =
          memory location) := by
  obtain ⟨result⟩ := ByteHostResult.results_exist c Out layout.soz layout.soh layout.som layout.son
    domain hD hN hSP layout.blockInjective hQK hQV hOK hOV hu
  let address := outputAddress (B := B) c layout.base layout.soz layout.soh layout.som layout.son
  obtain ⟨hv,hf⟩ := logical_result_and_frame address
    (fun i => encode (ByteHostResult.word result i)) layout.injective memory
  refine ⟨result,?_,hf⟩
  intro i
  rw [hv]
  have hn := ByteHostResult.returned_values_and_frame c Out layout.soz layout.soh layout.som layout.son
    x binding domain result hD hN hSP layout.blockInjective hQK hQV hOK hOV hu
    address layout.injective (fun _ => 0)
  have hs := logical_result_and_frame address (ByteHostResult.word result) layout.injective (fun _ => 0)
  exact congrArg encode ((hs.1 i).symm.trans (hn.1 i))

#print axioms attention_fp16_access_safe
#print axioms attention_e5m2_access_safe
#print axioms attention_fp16_storage
#print axioms attention_e5m2_storage
#print axioms attention_fp16_source
#print axioms attention_e5m2_source
#print axioms attention_fp16_tensor
#print axioms attention_e5m2_tensor
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.AttentionLlama.Correctness
