import bench.repairs.tritonbench_g.fused_layernorm_triton.Proofs.Logical
import bench.repairs.tritonbench_g.fused_layernorm_triton.Proofs.KernelRaw
import bench.repairs.tritonbench_g.fused_layernorm_triton.Proofs.Physical
import bench.repairs.tritonbench_g.fused_layernorm_triton.Correct
import bench.repairs.tritonbench_g.fused_layernorm_triton.Misc.NativeCorrect

/-! Optional Tensor checks, pointer bindings and physical-byte application. -/
namespace VeriTile.Bench.Repairs.FusedLayernormTriton.TensorBridge
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.FusedLayernormTriton FPStages Whole
open Binding
open Physical (observeCell)
open VeriTile.Bench.Repairs.FusedLayernormTriton.Equivalence
noncomputable section

theorem fused_layernorm_real (rest : Algebra ℝ) (s : State ℝ)
    (read : Nat → KernelBinding.Word) (decode : KernelBinding.Word → ℝ)
    (w b x : Contract.Metadata) (W B X : Torch.Tensor)
    (Y Mean Rstd : RegionName) (h : Contract.InputContract w b x)
    (api : Inputs s (fun a => decode (read a)) w b x W B X)
    (sep : Separation (region X) (region W) (region B) Y Mean Rstd)
    (c : Contract.Configuration) (complete : c.tree.Complete) :
    ∃ final,
      KernelProof.launch (ideal rest) realEqual (region X) (region W) (region B) Y Mean Rstd
        (Contract.rows x) (Contract.width x) c s = some final ∧
      (∀ row, row < Contract.rows x → final.mem Mean row = .mk .real
        (Contract.mean (Contract.width x) (fun col => decode
          (KernelBinding.logical read X api.inputAPI (col+Contract.width x*row))))) ∧
      (∀ row, row < Contract.rows x → final.mem Rstd row = .mk .real
        (1 / Real.sqrt (Contract.variance (Contract.width x) (fun col => decode
          (KernelBinding.logical read X api.inputAPI (col+Contract.width x*row))) + 1e-5))) ∧
      (∀ row, row < Contract.rows x → ∀ col, col < Contract.width x →
        final.mem Y (col+Contract.width x*row) = .mk .bf16
          (Contract.output (Contract.width x)
            (fun j => decode (KernelBinding.logical read X api.inputAPI (j+Contract.width x*row)))
            (fun j => decode (KernelBinding.logical read W api.weightAPI j))
            (fun j => decode (KernelBinding.logical read B api.biasAPI j)) col)) ∧
      (∀ q off, (∀ row, row < Contract.rows x → rowMiss Y Mean Rstd (Contract.width x) row q off) →
        final.mem q off = s.mem q off) := by
  obtain ⟨final,run,mean,inverse,output,frame⟩ := KernelRaw.raw_real rest s
    (region X) (region W) (region B) Y Mean Rstd (Contract.rows x) (Contract.width x)
    h.positive sep c complete
  refine ⟨final,run,?_,?_,?_,frame⟩
  · intro row hr
    simpa only [(Logical.moments s (fun a => decode (read a)) w b x W B X api h row hr).1,
      KernelBinding.decoded_logical read decode x X api.inputAPI api.inputMetadata h.input] using mean row hr
  · intro row hr
    simpa only [logicalInverse,
      (Logical.moments s (fun a => decode (read a)) w b x W B X api h row hr).2,
      KernelBinding.decoded_logical read decode x X api.inputAPI api.inputMetadata h.input] using inverse row hr
  · intro row hr col hc
    have wvalues := funext (KernelBinding.decoded_logical read decode w W api.weightAPI api.weightMetadata h.weight)
    have bvalues := funext (KernelBinding.decoded_logical read decode b B api.biasAPI api.biasMetadata h.bias)
    simpa only [Logical.output s (fun a => decode (read a)) w b x W B X api h row col hr hc,
      KernelBinding.decoded_logical read decode x X api.inputAPI api.inputMetadata h.input,
      wvalues, bvalues] using output row hr col hc

theorem fused_layernorm_source_fp32_bf16 (I : Native.Instructions) (P : Native.Profile)
    (s : State Native.Term) (read : Nat → KernelBinding.Word)
    (w b x : Contract.Metadata) (W B X : Torch.Tensor) (Y Mean Rstd : RegionName)
    (h : Contract.InputContract w b x) (api : Inputs s (KernelBinding.graphRead read) w b x W B X)
    (sep : Separation (region X) (region W) (region B) Y Mean Rstd) (c : Contract.Configuration) :
    ∃ final,
      KernelProof.launch Native.algebra (Native.equal I P) (region X) (region W) (region B)
        Y Mean Rstd (Contract.rows x) (Contract.width x) c s = some final ∧
      (∀ row, row < Contract.rows x →
        let ref := Native.reference I P
          (fun j => KernelBinding.logicalGraph read X api.inputAPI (j+Contract.width x*row))
          (KernelBinding.logicalGraph read W api.weightAPI) (KernelBinding.logicalGraph read B api.biasAPI)
          (Contract.width x) c.block c.tree
        observeCell I P (final.mem Mean row) = .mk .real ref.mean ∧
        observeCell I P (final.mem Rstd row) = .mk .real ref.rstd ∧
        ∀ col, col < Contract.width x →
          observeCell I P (final.mem Y (col+Contract.width x*row)) = .mk .bf16 (ref.output col)) ∧
      (∀ q off, (∀ row, row < Contract.rows x → rowMiss Y Mean Rstd (Contract.width x) row q off) →
        final.mem q off = s.mem q off) := by
  obtain ⟨hx,hw,hb⟩ := KernelBinding.graph_cells read s w b x W B X h api
  exact KernelValues.native I P s (region X) (region W) (region B) Y Mean Rstd
    (Contract.rows x) (Contract.width x)
    (fun row j => KernelBinding.logicalGraph read X api.inputAPI (j+Contract.width x*row))
    (KernelBinding.logicalGraph read W api.weightAPI) (KernelBinding.logicalGraph read B api.biasAPI)
    hx hw hb sep c

#axiomsClean fused_layernorm_real
#axiomsClean fused_layernorm_source_fp32_bf16

theorem python_checks_iff_preconditions (w b x : Contract.Argument) :
    Contract.checkArguments w b x = .ok () ↔ Contract.AdmittedArguments w b x :=
  Contract.checkArguments_iff w b x

theorem actual_pointers_and_indices {α : Type} (s : State α) (read : Nat → α)
    (w b x : Contract.Metadata) (W B X : Torch.Tensor) (h : Contract.InputContract w b x)
    (api : Inputs s read w b x W B X) :
    (∀ m t, (m = w ∧ t = W) ∨ (m = b ∧ t = B) ∨ (m = x ∧ t = X) →
      0 < m.numel → m.dataPtr = BitVec.ofNat 64 t.storage.base) ∧
    (∀ (c : Contract.Configuration) row k lane, row < Contract.rows x →
      k*c.block < Contract.width x → lane < c.block →
      NativeIndex.Bounds (Contract.rows x) (Contract.width x) c.block row k lane) :=
  KernelBinding.pointers_and_indices s read w b x W B X h api

/-- The same successful native kernel result at its actual launch pointers.
Inputs are decoded directly from the caller's byte heap; output byte widths,
source store compatibility and all untouched bytes are explicit. -/
theorem fused_layernorm_physical (I : Native.Instructions) (P : Native.Profile)
    (s : State Native.Term) (heap : Physical.Bytes)
    (w b x : Contract.Metadata) (W B X : Torch.Tensor) (Y Mean Rstd : RegionName)
    (h : Contract.InputContract w b x)
    (api : Inputs s (KernelBinding.graphRead (Physical.read32 heap)) w b x W B X)
    (sep : Separation (region X) (region W) (region B) Y Mean Rstd) (c : Contract.Configuration)
    (p : Physical.Outputs (Contract.rows x) (Contract.width x)) (fresh : Physical.FreshInputs p W B X) :
    ∃ final,
      KernelProof.launch Native.algebra (Native.equal I P) (region X) (region W) (region B)
        Y Mean Rstd (Contract.rows x) (Contract.width x) c s = some final ∧
      (∀ row, row < Contract.rows x →
        let ref := Native.reference I P
          (fun j => KernelBinding.logicalGraph (Physical.read32 heap) X api.inputAPI (j+Contract.width x*row))
          (KernelBinding.logicalGraph (Physical.read32 heap) W api.weightAPI)
          (KernelBinding.logicalGraph (Physical.read32 heap) B api.biasAPI)
          (Contract.width x) c.block c.tree
        (∀ byte, byte < 4 → Physical.outputBytes I P p Y Mean Rstd heap final (p.mean.toNat+4*row+byte) =
          Physical.byteOf ref.mean byte) ∧
        (∀ byte, byte < 4 → Physical.outputBytes I P p Y Mean Rstd heap final (p.rstd.toNat+4*row+byte) =
          Physical.byteOf ref.rstd byte) ∧
        (∀ col, col < Contract.width x → ∀ byte, byte < 2 →
          Physical.outputBytes I P p Y Mean Rstd heap final (p.y.toNat+2*(col+Contract.width x*row)+byte) =
            Physical.byteOf (ref.output col) byte)) ∧
      (∀ address, Physical.Outside p address →
        Physical.outputBytes I P p Y Mean Rstd heap final address = heap address) ∧
      (∀ t, t = W ∨ t = B ∨ t = X → ∀ i, i < t.storage.elements → ∀ byte, byte < 4 →
        Physical.outputBytes I P p Y Mean Rstd heap final (t.storage.address i+byte) =
          heap (t.storage.address i+byte)) ∧
      (∀ q off, (∀ row, row < Contract.rows x → rowMiss Y Mean Rstd (Contract.width x) row q off) →
        final.mem q off = s.mem q off) := by
  obtain ⟨final,run,result,frame⟩ := fused_layernorm_source_fp32_bf16 I P s (Physical.read32 heap)
    w b x W B X Y Mean Rstd h api sep c
  refine ⟨final,run,?_,Physical.frame I P p Y Mean Rstd heap final,?_,frame⟩
  · intro row hr
    have hrow := result row hr
    have hm := Physical.observed_real I P _ _ hrow.1
    have hs := Physical.observed_real I P _ _ hrow.2.1
    refine ⟨?_,?_,?_⟩
    · intro byte hb
      rw [Physical.mean_bytes I P p Y Mean Rstd heap final row byte hr hb]
      exact congrArg (fun v => Physical.byteOf v byte) hm
    · intro byte hb
      rw [Physical.rstd_bytes I P p Y Mean Rstd heap final row byte hr hb]
      exact congrArg (fun v => Physical.byteOf v byte) hs
    · intro col hc byte hb
      have hy := Physical.observed_bf16 I P _ _ (hrow.2.2 col hc)
      rw [Physical.output_bytes I P p Y Mean Rstd heap final _ byte (output_bounds _ _ _ _ hr hc) hb]
      exact congrArg (fun v => Physical.byteOf v byte) hy
  · intro t ht i hi byte hb
    exact Physical.frame I P p Y Mean Rstd heap final _ (fresh t ht i hi byte hb)

#axiomsClean python_checks_iff_preconditions
#axiomsClean actual_pointers_and_indices
#axiomsClean fused_layernorm_physical
end
end VeriTile.Bench.Repairs.FusedLayernormTriton.TensorBridge
