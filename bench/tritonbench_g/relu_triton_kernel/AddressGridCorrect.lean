import bench.tritonbench_g.relu_triton_kernel.Proofs.ByteAddress
import bench.tritonbench_g.relu_triton_kernel.WordGridCorrect

/-! Compose the existing finite-input-word logical execution with a separate
bounded index and byte-address certificate. No native memory relation is a
conclusion: mathematical base intervals do not establish storage, alignment,
provenance, allocation, compiler or instruction-interleaving correctness.
N=0 keeps the logical empty-grid result; native zero-grid behavior stays open.
N>=2^31 and non-1024 launch parameters remain outside this subtheorem. -/
namespace VeriTile.Bench.TritonBenchG.ReluTritonKernel.AddressGridCorrect
open VeriTile.Triton FP.Structural GatedGrid TypedValue IntegerWords FiniteAddress
noncomputable section

/-- Execute the first four statements of the existing typed source itself.
The logical registers agree with the bounded bit-vector expressions. The
entire prefix succeeds independently of memory or floating callbacks. -/
theorem source_prefix_registers {α : Type} [Inhabited α]
    (M : Algebra α) (cmp16 cmp32 : α → α → Bool) (encoding : Encoding)
    (x y : RegionName) (N pid : Nat) (initial : State α)
    (indexDomain : IndexDomain N) (launched : pid < gridCount N blockSize) :
    ∃ t, run (backend M cmp16 cmp32)
        ((source encoding.profile x y N blockSize).surfaceBody.take 4) (atPid initial pid) = some t ∧
      t.regs .nat [blockSize] "offsets" =
        some (fun i : TileIndex [blockSize] => (offset32 pid i.1.val).toNat) ∧
      t.regs .bool [blockSize] "mask" =
        some (fun i : TileIndex [blockSize] => mask32 N pid i.1.val) ∧
      t.mem = initial.mem := by
  have offsets : (fun i : TileIndex [blockSize] => (offset32 pid i.1.val).toNat) =
      fun i : TileIndex [blockSize] => offset pid i.1.val := by
    funext i
    exact (lane_arithmetic N pid i.1.val indexDomain launched i.1.isLt).2.2
  have masks : (fun i : TileIndex [blockSize] => mask32 N pid i.1.val) =
      fun i : TileIndex [blockSize] => decide (offset pid i.1.val < N) := by
    funext i
    exact mask_exact N pid i.1.val indexDomain launched i.1.isLt
  rw [offsets, masks]
  simp [source, materialized, ComputeKernel.surfaceBody, run, step, evalExpr,
    evalOp_unfold, numeric, numericLt, natLt, backend,
    atPid, offset, State.setReg]
  exact ⟨rfl, rfl⟩

def AccessCertificate (base bytes N pid lane : Nat) : Prop :=
  (pointer64 base bytes pid lane).toNat = base + offset pid lane * bytes ∧
  ∀ byte, byte < bytes →
    (bytePointer64 base bytes pid lane byte).toNat = base + offset pid lane * bytes + byte ∧
    base ≤ base + offset pid lane * bytes + byte ∧
    base + offset pid lane * bytes + byte < base + N * bytes ∧
    base + offset pid lane * bytes + byte < 2^64

def AddressCertificate (e : Encoding) (N inputBase outputBase : Nat) : Prop :=
  ∀ pid, pid < gridCount N blockSize → ∀ lane, lane < blockSize →
    (start32 pid).toInt = (pid * blockSize : Nat) ∧
    (offset32 pid lane).toInt = (offset pid lane : Nat) ∧
    (offset32 pid lane).toNat = offset pid lane ∧
    mask32 N pid lane = decide (offset pid lane < N) ∧
    stores32 pid = decide (pid = 0) ∧
    (mask32 N pid lane = Bool.true →
      AccessCertificate inputBase (elementBytes e) N pid lane ∧
      (stores32 pid = Bool.true → AccessCertificate outputBase 4 N pid lane))

theorem access_certificate (base bytes N pid lane : Nat)
    (indexDomain : IndexDomain N) (domain : ByteDomain base bytes N)
    (positive : 0 < bytes) (launched : pid < gridCount N blockSize)
    (inBlock : lane < blockSize) (active : offset pid lane < N) :
    AccessCertificate base bytes N pid lane := by
  refine ⟨pointer_exact base bytes N pid lane indexDomain domain positive launched inBlock active, ?_⟩
  intro byte inElement
  exact ⟨byte_pointer_exact base bytes N pid lane byte indexDomain domain launched inBlock active inElement,
    byte_span base bytes N (offset pid lane) byte domain active inElement⟩

theorem address_certificate (e : Encoding) (N inputBase outputBase : Nat)
    (indexDomain : IndexDomain N)
    (inputDomain : ByteDomain inputBase (elementBytes e) N)
    (outputDomain : ByteDomain outputBase 4 N) : AddressCertificate e N inputBase outputBase := by
  intro pid launched lane inBlock
  have index := lane_arithmetic N pid lane indexDomain launched inBlock
  have mask := mask_exact N pid lane indexDomain launched inBlock
  refine ⟨index.1, index.2.1, index.2.2, mask, stores_exact N pid indexDomain launched, ?_⟩
  intro activeMask
  have active : offset pid lane < N := by simpa only [mask, decide_eq_true_eq] using activeMask
  refine ⟨access_certificate inputBase (elementBytes e) N pid lane indexDomain inputDomain
    (element_bytes_positive e) launched inBlock active, ?_⟩
  intro _
  exact access_certificate outputBase 4 N pid lane indexDomain outputDomain (by decide) launched inBlock active

/-- Successful logical execution and its exact frame are proved from the
input-word relation. The additional bounds yield an independent arithmetic
certificate; no successful native execution or desired result is assumed. -/
specification relu_word_grid_finite_address_correctness {α : Type} [Inhabited α]
    (M : Algebra α) (cmp16 cmp32 : α → α → Bool)
    (encoding : Encoding) (words : Nat → Word encoding)
    (x y : RegionName) (N inputBase outputBase : Nat) (initial : State α)
    (inputWords : InputWords encoding words x N initial.mem)
    (indexDomain : IndexDomain N)
    (inputDomain : ByteDomain inputBase (elementBytes encoding) N)
    (outputDomain : ByteDomain outputBase 4 N) :
    ∃ final,
      runGrid (backend M cmp16 cmp32) (source encoding.profile x y N blockSize)
        N blockSize initial = some final ∧
      final.mem = wordSpec M encoding words y N blockSize initial.mem ∧
      (∀ r o, r ≠ y ∨ blockSize ≤ o ∨ N ≤ o → final.mem r o = initial.mem r o) ∧
      AddressCertificate encoding N inputBase outputBase := by
  obtain ⟨final, he, hm⟩ := WordGridCorrect.relu_word_grid_correctness
    M cmp16 cmp32 encoding words x y N blockSize initial inputWords
  refine ⟨final, he, hm, ?_, address_certificate encoding N inputBase outputBase
    indexDomain inputDomain outputDomain⟩
  intro r o outside
  rw [hm]
  exact wordSpec_frame M encoding words y N blockSize initial.mem r o outside

#specNonCircular AddressCertificate avoiding [TypedValue.source, runPrefix, runGrid, FP.Structural.exec]
#axiomsClean source_prefix_registers
#axiomsClean access_certificate
#axiomsClean address_certificate
#axiomsClean relu_word_grid_finite_address_correctness
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ReluTritonKernel.AddressGridCorrect
