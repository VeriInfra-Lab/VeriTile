import bench.repairs.tritonbench_g.rope_embedding.Proofs.Public

/-! Retained optional host results. The current public scope is the kernel. -/
namespace VeriTile.Bench.Repairs.RopeEmbedding
open VeriTile.Triton FP.Structural Numerics

theorem rope_embedding_forward_correct (api : Memory.API ℝ Store Byte) (r : Request) (s : State ℝ)
    (call : Native.Preparation api id r s) (accepted : Checked.check api id r s call Bool.true Bool.true=Bool.true) :
    ∃ out,Checked.forward api id id ideal r s call=some out ∧
      Checked.ForwardResult api id r s call (Checked.RealOutput api r Bool.false s call) out :=
  Checked.forward_real api r s call accepted

theorem rope_embedding_backward_correct (api : Memory.API ℝ Store Byte) (r : Request) (s : State ℝ)
    (call : Native.Preparation api id r s) (groups block warps : Nat)
    (accepted : Checked.check api id r s call Bool.false Bool.false=Bool.true)
    (cached : Checked.cacheCheck r groups block warps=Bool.true) :
    ∃ out,Checked.backward api id id ideal r s call groups block warps=some out ∧
      Checked.RealOutput api r Bool.true s call out :=
  Checked.backward_real api r s call groups block warps accepted cached

theorem rope_embedding_forward_fp32 (api : Memory.API Word Store (BitVec 8)) (p : Primitives)
    (r : Request) (s : State Tagged) (call : Native.Preparation api leaf r s)
    (codec : ∀ w k,api.encode w k=wordByte w k)
    (accepted : Checked.check api leaf r s call Bool.true Bool.true=Bool.true) :
    ∃ out,Checked.forward api leaf Tagged.word (operations p) r s call=some out ∧
      Checked.ForwardResult api leaf r s call (Checked.FP32Output api p r Bool.false s call) out :=
  Checked.forward_fp32 api p r s call codec accepted

theorem rope_embedding_backward_fp32 (api : Memory.API Word Store (BitVec 8)) (p : Primitives)
    (r : Request) (s : State Tagged) (call : Native.Preparation api leaf r s) (groups block warps : Nat)
    (codec : ∀ w k,api.encode w k=wordByte w k)
    (accepted : Checked.check api leaf r s call Bool.false Bool.false=Bool.true)
    (cached : Checked.cacheCheck r groups block warps=Bool.true) :
    ∃ out,Checked.backward api leaf Tagged.word (operations p) r s call groups block warps=some out ∧
      Checked.FP32Output api p r Bool.true s call out :=
  Checked.backward_fp32 api p r s call groups block warps codec accepted cached

#axiomsClean rope_embedding_forward_correct
#axiomsClean rope_embedding_backward_correct
#axiomsClean rope_embedding_forward_fp32
#axiomsClean rope_embedding_backward_fp32
end VeriTile.Bench.Repairs.RopeEmbedding
