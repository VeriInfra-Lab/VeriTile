import bench.tritonbench_g.relu_strided_buffer.Proofs.Mathematical_StructuralTile
import bench.tritonbench_g.relu_triton_kernel.Proofs.Mathematical_SoftwareFP32

/-!
Raw FP32 specialization of the complete signed source perTile execution.
Comparison is strict ordered less-than, used as 0 < x by the source's x > 0.
Both zero signs and all NaNs select positive zero. Same-FP32 casts use the
existing software scalar backend. The source real-literal-to-FP32 zero path
has one explicit primitive law; no other mixed-format cast is erased.
Native scalar, memory, launch and finite-index conformance are not asserted.
-/
namespace VeriTile.Bench.TritonBenchG.ReluStridedBuffer.WordTile
open VeriTile.Triton FP.Structural SourceSignedBranch StructuralTile
noncomputable section
local instance : Inhabited (FP.Value .fp32) := ⟨FP.Value.zero .fp32⟩

/-- Strict order from the already defined bit-level ordered non-strict order.
The conjunction rejects unordered pairs and both orders of signed zero. -/
def orderedLt (x y : FP.Value .fp32) : Bool :=
  ReluTritonKernel.SoftwareFP32.orderedLe x y &&
    !(ReluTritonKernel.SoftwareFP32.orderedLe y x)

def wordRelu (x : FP.Value .fp32) : FP.Value .fp32 :=
  if orderedLt (FP.Value.zero .fp32) x then x else FP.Value.zero .fp32

/-- Only the exact comparison slot used by this source is supplied here. -/
def backend (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) : Algebra (FP.Value .fp32) :=
  { FP.Software.scalarFP32 rest cfg with
    compareLt := fun p d => if p = none ∧ d = .fp32 then some orderedLt else rest.compareLt p d }

/-- Primitive interface for the source's literal-zero conversion.
WholeFP32.zero_literal_exact discharges it from the explicit word semantics;
this is not a numerical-experiment obligation.
It speaks only about zero construction, not a desired ReLU result or loop. -/
def ZeroLiteralLaw (rest : Algebra (FP.Value .fp32)) : Prop :=
  rest.cast none .real .fp32 (rest.literal none .real 0) = FP.Value.zero .fp32

theorem backend_compare (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) :
    (backend rest cfg).compareLt none .fp32 = some orderedLt := by simp [backend]

theorem scalarResult_word (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (zeroLaw : ZeroLiteralLaw rest) (x : FP.Value .fp32) :
    scalarResult (backend rest cfg) orderedLt x = wordRelu x := by
  simp only [ZeroLiteralLaw] at zeroLaw
  simp [scalarResult, scalarZero, backend, FP.Software.scalarFP32,
    FP.Software.fp32Context, zeroLaw, wordRelu]

def wordOutput (c : SignedViews) (T : Nat) (mem : Memory (FP.Value .fp32)) :
    Memory (FP.Value .fp32) :=
  (TileShape.allIndices [c.B]).foldl (fun current i =>
    if T * c.B + i.1.val < c.n then
      writeCell current c.outputView.region (c.outputView.address (T * c.B + i.1.val))
        (.mk .fp32 (wordRelu
          ((mem c.inputView.region (c.inputView.address (T * c.B + i.1.val))).read .fp32)))
    else current) mem

theorem tileOutput_word (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (zeroLaw : ZeroLiteralLaw rest) (c : SignedViews) (T : Nat)
    (mem : Memory (FP.Value .fp32)) :
    tileOutput (backend rest cfg) orderedLt c T mem = wordOutput c T mem := by
  simp only [tileOutput, scalarResult_word rest cfg zeroLaw, wordOutput]

/-- The existing installed signed source statements execute on FP32 words.
Only ZeroLiteralLaw binds a still external scalar primitive; the complete
execution, input dependence, typed stores and controls are all derived. -/
theorem perTile_words (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (zeroLaw : ZeroLiteralLaw rest) (c : SignedViews) (T : Nat)
    (s : State (FP.Value .fp32)) (ht : s.regs .nat [] "tile_id" = some (fun _ => T)) :
    ∃ final, run (backend rest cfg) (c.perTile.map ComputeStmt.alg) s = some final ∧
      final.mem = wordOutput c T s.mem ∧ final.undef = s.undef ∧
      final.regs .nat [] "pid" = s.regs .nat [] "pid" ∧
      final.regs .nat [] "num_ctas" = s.regs .nat [] "num_ctas" ∧
      final.pids = s.pids ∧ final.numPids = s.numPids := by
  obtain ⟨final, hexec, hmem, hrest⟩ :=
    perTile_run (backend rest cfg) orderedLt (backend_compare rest cfg) c T s ht
  exact ⟨final, hexec, (hmem.trans (tileOutput_word rest cfg zeroLaw c T s.mem)), hrest⟩

theorem nan_selects_positive_zero (x : FP.Value .fp32) (nan : x.isNaN = Bool.true) :
    wordRelu x = FP.Value.zero .fp32 := by
  simp [wordRelu, orderedLt, ReluTritonKernel.SoftwareFP32.orderedLe, nan]

theorem negative_zero_becomes_positive_zero :
    (wordRelu (FP.Value.ofBits .fp32 0x80000000)).bits = 0 := by decide

theorem quiet_nan_becomes_positive_zero :
    (wordRelu (FP.Value.ofBits .fp32 0x7fc12345)).bits = 0 := by decide

theorem signaling_nan_becomes_positive_zero :
    (wordRelu (FP.Value.ofBits .fp32 0x7f800001)).bits = 0 := by decide

theorem positive_subnormal_is_preserved :
    (wordRelu (FP.Value.ofBits .fp32 1)).bits = 1 := by decide

theorem positive_infinity_is_preserved :
    (wordRelu (FP.Value.ofBits .fp32 0x7f800000)).bits = 0x7f800000 := by decide

theorem negative_infinity_becomes_positive_zero :
    (wordRelu (FP.Value.ofBits .fp32 0xff800000)).bits = 0 := by decide

#guard_msgs (drop info) in
#specNonCircular wordRelu avoiding [perTileSource, kernelSource, FP.Structural.exec]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ReluStridedBuffer.WordTile
