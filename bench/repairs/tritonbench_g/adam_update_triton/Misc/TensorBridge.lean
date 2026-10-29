import bench.repairs.tritonbench_g.adam_update_triton.Misc.NativeCorrect
import bench.repairs.tritonbench_g.adam_update_triton.Correct
import bench.repairs.tritonbench_g.adam_update_triton.FPEquiv
import bench.repairs.tritonbench_g.adam_update_triton.Proofs.InputBinding

/-! Optional checked-Tensor applications and input-check correspondence. -/
namespace VeriTile.Bench.Repairs.AdamUpdateTriton.TensorBridge
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SwigluBackward.Observed (Term)
open Correct NativeExecution
noncomputable section

private theorem target_eq (params : Scalars ℝ) (output : Output) (p g m : ℝ) :
    Math.target params output p g m = formula params output p g m := by
  cases output <;> rfl

open VeriTile.Torch.FP32 in
/-- Checked physical Tensor reads compose with complete kernel correctness.
The region snapshot is constructed from actual storage views, not from an
assumed result. n is the local mask extent and may be a shorter prefix. -/
theorem lion_checked_views_correctness
    (read : Nat → Word) (decode : Word → ℝ) (p g m : Tensor)
    (lp : TensorLaws p) (lg : TensorLaws g) (lm : TensorLaws m)
    (h : InputContract p.metadata g.metadata m.metadata)
    (params : Scalars ℝ) (total n B pid : Nat) (hn : n ≤ p.metadata.numel)
    (scratch : State ℝ) :
    let s := InputBinding.bindReadViews read decode p g m lp lg lm h scratch
    Spec.Real (∃ final,
      run Real.semantics.backend (source total n B).surfaceBody
        (Tile.programState params pid s) = some final ∧
      (∀ output (j : TileIndex [B]), pid * B + j.1.val < n →
        final.mem (Tile.outputRegion output) (pid * B + j.1.val) =
          Cell.mk .real (formula params output
            (decode (logicalValue read p lp (pid * B + j.1.val)))
            (decode (logicalValue read g lg (pid * B + j.1.val)))
            (decode (logicalValue read m lm (pid * B + j.1.val))))) ∧
      (∀ r o, Tile.Outside n B pid r o → final.mem r o = s.mem r o)) := by
  obtain ⟨final, execution, values, frame⟩ := Real.kernel_execution params total n B pid
    (InputBinding.bindReadViews read decode p g m lp lg lm h scratch)
  obtain ⟨hp, hg, hm⟩ := InputBinding.bindReadViews_logical read decode p g m lp lg lm h scratch
  refine ⟨final, execution, ?_, frame⟩
  intro output j active
  simpa only [target_eq, hp _ (lt_of_lt_of_le active hn), hg _ (lt_of_lt_of_le active hn),
    hm _ (lt_of_lt_of_le active hn)] using values output j active

open VeriTile.Torch.FP32 in
theorem checked_tensor_pointers (p g m : Tensor) (lp : TensorLaws p) (lg : TensorLaws g) (lm : TensorLaws m)
    (h : InputContract p.metadata g.metadata m.metadata) (hn : 0 < p.metadata.numel) :
    BitVec.ofNat 64 ((dataView p lp h.contiguous.1).address 0) = p.dataPtr ∧
    BitVec.ofNat 64 ((dataView g lg h.contiguous.2.1).address 0) = g.dataPtr ∧
    BitVec.ofNat 64 ((dataView m lm h.contiguous.2.2).address 0) = m.dataPtr :=
  InputBinding.actual_pointers p g m lp lg lm h hn

open VeriTile.Torch.FP32 in
theorem checked_native_address (t : Tensor) (laws : TensorLaws t)
    (hc : t.metadata.isContiguous = Bool.true) (n : Nat)
    (config : VeriTile.Bench.TritonBenchG.AdamUpdateTriton.HostAutotune.Config)
    (pid lane : Nat) (extent : n ≤ t.metadata.numel) (bounded : n ≤ 8388480)
    (hp : pid < Index.grid n config) (hl : lane < config.block)
    (active : pid * config.block + lane < n) :
    (Index.pointer (dataView t laws hc).base 0 pid config.block lane).toNat =
      (dataView t laws hc).address (pid * config.block + lane) :=
  InputBinding.native_address t laws hc n config pid lane extent bounded hp hl active

open Index VeriTile.Bench.TritonBenchG.AdamUpdateTriton.HostAutotune in
theorem native_index_exact (count : Nat) (config : Config) (pid lane : Nat)
    (bounded : count≤8388480) (hp : pid<grid count config) (hl : lane<config.block) :
    (BitVec.ofNat 32 pid).toInt=(pid:Int) ∧
    (BitVec.ofNat 32 config.block).toInt=(config.block:Int) ∧
    (BitVec.ofNat 32 lane).toInt=(lane:Int) ∧
    (BitVec.ofNat 32 pid*BitVec.ofNat 32 config.block).toInt=(pid*config.block:Nat) ∧
    (offset pid config.block lane).toInt=(pid*config.block+lane:Nat) :=
  Index.i32_exact count config pid lane bounded hp hl

open Index VeriTile.Bench.TritonBenchG.AdamUpdateTriton.HostAutotune in
theorem native_mask_exact (count : Nat) (config : Config) (pid lane : Nat)
    (bounded : count≤8388480) (hp : pid<grid count config) (hl : lane<config.block) :
    (offset pid config.block lane).slt (BitVec.ofNat 32 count)=decide (pid*config.block+lane<count) :=
  Index.mask_exact count config pid lane bounded hp hl


theorem python_checks_iff (p g m : ArgumentMetadata) :
    checkInputs p g m = .ok () ↔ AdmittedArguments p g m := checkInputs_iff p g m

open VeriTile.Torch.FP32 in
/-- Physical FP32 Tensor views supply the word leaves in the public native
theorem. No finiteness or primitive software-conformance premise is added. -/
theorem lion_checked_views_native_equiv
    (read : Nat → Word) (p g m : Tensor)
    (lp : TensorLaws p) (lg : TensorLaws g) (lm : TensorLaws m)
    (h : InputContract p.metadata g.metadata m.metadata)
    (native : NativeWords.Instructions) (params : Scalars Native.Word)
    (total n B pid : Nat) (hn : n ≤ p.metadata.numel) (scratch : State (Term Native.Word)) :
    let s := InputBinding.bindReadViews read (fun w => Term.input w.bits) p g m lp lg lm h scratch
    ∃ final,
      run (Native.semantics (NativeWords.ports native)).backend (source total n B).surfaceBody
        (Tile.programState (Native.arguments params) pid s) = some final ∧
      (∀ output (j : TileIndex [B]), pid * B + j.1.val < n →
        observeCell (NativeWords.ports native) (final.mem (Tile.outputRegion output) (pid * B + j.1.val)) =
          Cell.mk .real (Native.reference (NativeWords.ports native) params output
            (logicalValue read p lp (pid * B + j.1.val)).bits
            (logicalValue read g lg (pid * B + j.1.val)).bits
            (logicalValue read m lm (pid * B + j.1.val)).bits)) ∧
      (∀ r o, Tile.Outside n B pid r o → final.mem r o = s.mem r o) := by
  obtain ⟨hp, hg, hm⟩ := InputBinding.bindReadViews_logical read (fun w => Term.input w.bits)
    p g m lp lg lm h scratch
  exact NativeExecution.kernel_execution native params total n B pid _
    (fun j => (logicalValue read p lp j).bits) (fun j => (logicalValue read g lg j).bits)
    (fun j => (logicalValue read m lm j).bits)
    (fun j hj => hp _ (lt_of_lt_of_le hj hn))
    (fun j hj => hg _ (lt_of_lt_of_le hj hn))
    (fun j hj => hm _ (lt_of_lt_of_le hj hn))

#axiomsClean lion_checked_views_correctness
#axiomsClean lion_checked_views_native_equiv
#axiomsClean checked_tensor_pointers
#axiomsClean checked_native_address
#axiomsClean native_index_exact
#axiomsClean native_mask_exact
#axiomsClean python_checks_iff
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AdamUpdateTriton.TensorBridge
