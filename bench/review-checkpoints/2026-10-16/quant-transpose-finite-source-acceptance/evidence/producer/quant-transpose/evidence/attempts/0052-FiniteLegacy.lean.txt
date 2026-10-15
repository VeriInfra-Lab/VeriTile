import bench.tritonbench_g.quant_transpose_kernel.Proofs.FiniteGrid
import bench.tritonbench_g.quant_transpose_kernel.IntegerCorrect

namespace QuantTransposeFinite
open VeriTile.Triton
set_option maxHeartbeats 4000000

/-- A total extension used only to compare with the older total callback
interpreter. The finite source interpreter still rejects undefined llrint. -/
def legacyRound (w : FP.Value .fp32) : BitVec 64 := (llrint .fp32 w.bits).getD 0

def Mul32Agrees (b : Numerics) (cfg : FP.Config) : Prop :=
  ∀ x y, b.mul32 x y=(FP.mul cfg (FP.Value.mk x : FP.Value .fp32) (FP.Value.mk y : FP.Value .fp32)).bits

theorem scaled_legacy (b : Numerics) (cfg : FP.Config) (h : Mul32Agrees b cfg)
    (x v : Word .fp32) :
    scaled b .fp32 x v=(QuantizeIntegerStage.FP32.scaled cfg (FP.Value.mk x) (FP.Value.mk v)).bits := by
  dsimp [scaled,mul,promoted,scale127,QuantizeIntegerStage.FP32.scaled]
  rw [h,h]
  rfl

/-- Pointwise bridge to the already proved fp32 llrint64/int8 source primitive;
the old callback's totalization is used only where exact conversion is defined. -/
theorem scalar_legacy (b : Numerics) (cfg : FP.Config) (h : Mul32Agrees b cfg)
    (x v : Word .fp32) (hd : LlrintDefined .fp32 (scaled b .fp32 x v)) :
    result b .fp32 x v=some (QuantizeIntegerStage.FP32.outputWord legacyRound cfg (FP.Value.mk x) (FP.Value.mk v)) := by
  have hh := (llrint_defined_iff .fp32 (scaled b .fp32 x v)).2 hd
  obtain ⟨z,hz⟩ := Option.isSome_iff_exists.mp hh
  simp only [result,hz,Option.map_some,QuantizeIntegerStage.FP32.outputWord,legacyRound]
  rw [←scaled_legacy b cfg h x v,hz]
  rfl

/-- The existing typed AST, interpreter, source-run proof and complete byte
store route remain directly available at every mapped 128x128 tile. -/
theorem existing_fp32_primitive
    (rest : FP.Structural.Algebra QuantizeIntegerStage.FP32.Word) (cfg : FP.Config)
    (s : FP.Structural.State QuantizeIntegerStage.FP32.Word) (p : Parameters) :
    ∃ t, QuantizeIntegerStage.exec (QuantizeIntegerStage.FP32.backend rest cfg) (some legacyRound)
      (QuantizeIntegerStage.Transpose.kernel p.x p.scale p.output p.sam p.san p.M 1 p.M p.N 128 128) s=some t ∧
      t.mem=(QuantizeIntegerStage.Transpose.written legacyRound cfg s p.x p.scale p.output p.sam p.san p.M 1 p.M p.N 128 128).mem := by
  exact QuantizeIntegerStage.Transpose.source_run rest legacyRound cfg s p.x p.scale p.output p.sam p.san p.M 1 p.M p.N 128 128

#axiomsClean scalar_legacy
#axiomsClean existing_fp32_primitive
#guard_msgs (drop info) in
#auditModuleAxioms
end QuantTransposeFinite
