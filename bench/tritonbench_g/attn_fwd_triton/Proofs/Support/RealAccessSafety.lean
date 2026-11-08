import VeriTile.Meta.StatementAudit
import VeriTile.Triton.DSL
import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.Semantics.EvalOp
import VeriTile.Triton.Source
import bench.tritonbench_g.attention_fwd_triton3.Proofs.Sources.Mathematical_RealAccess
import bench.tritonbench_g.attention_fwd_triton3.Proofs.Sources.Mathematical_RealFormula
import bench.tritonbench_g.attention_fwd_triton3.Proofs.Sources.Mathematical_RealKeys
import bench.tritonbench_g.attention_fwd_triton3.Proofs.Sources.Mathematical_RealLoopFormula
import bench.tritonbench_g.attention_fwd_triton3.Proofs.Sources.Mathematical_RealProbabilities
import bench.tritonbench_g.attention_fwd_triton3.Proofs.Sources.Mathematical_RealRegisterSafety
import bench.tritonbench_g.attention_fwd_triton3.Proofs.Sources.Mathematical_RealTiles
import bench.tritonbench_g.attention_score.Proofs.Mathematical_RealFrame
import bench.tritonbench_g.attn_fwd_triton.Proofs.Support.RealLoads
/-! Retained mathematical support for attn_fwd_triton. This module
provides no alternative kernel or finite-penalty source specification. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdTritonSupport.RealAccessSafety
open VeriTile.Triton RealLoads
open AttentionFwdTriton3.RealAccess
noncomputable section
set_option maxHeartbeats 800000

theorem masked_address_safe {shape : TileShape} (bounds : RegionBounds) (s : BlockState)
    (name : RegName) (ptrs : Tile .ptr shape) (mask : Op .bool shape) (bits : Tile .bool shape)
    (hp : s.regs .ptr shape name = some ptrs) (hm : evalOp mask s = some bits)
    (hb : ∀ i, bits.data i = Bool.true → (ptrs.data i).2 < bounds (ptrs.data i).1) :
    (MemAccess.ptr (d := .real) (.ref .ptr shape name)).ActiveAddressSafe bounds s
      ((MaskOpt.mask (dtype := .real) mask).Active s) := by
  unfold MemAccess.ActiveAddressSafe memAccessActiveAddressSafe Op.PointerAddressesSafeOn
  refine ⟨by simp [Op.MemorySafe], ?_⟩
  intro ptr hp' i hi
  rw [evalOp_ref, hp] at hp'
  cases hp'
  obtain ⟨ms, he, hi⟩ := hi
  rw [hm] at he
  cases he
  exact hb i hi

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdTritonSupport.RealAccessSafety