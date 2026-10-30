import bench.tritonbench_g.matmul_triton_autotune.Sources.Mathematical_Host

namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Whole
open VeriTile.Triton FP.Structural
open ParallelAttention.FPStages (foldRange)
noncomputable section

/-- Logical inputs only; the incoming carry enters the same primitive as the
actual source. Padded lanes remain in every BK-wide dot operation. -/
def logicalStep {α : Type} (Q : Numerics α) (r : Request)
    (A : Fin r.rows → Fin r.inner → α) (B : Fin r.inner → Fin r.columns → α)
    (i : Fin r.rows) (j : Fin r.columns) (block : Nat) (acc : α) : α :=
  Q.dotAcc r.config.bk
    (fun k => if hk : block*r.config.bk+k.val<r.inner then A i ⟨block*r.config.bk+k.val,hk⟩ else Q.zero)
    (fun k => if hk : block*r.config.bk+k.val<r.inner then B ⟨block*r.config.bk+k.val,hk⟩ j else Q.zero) acc

def logicalReference {α : Type} (Q : Numerics α) (r : Request)
    (A : Fin r.rows → Fin r.inner → α) (B : Fin r.inner → Fin r.columns → α)
    (i : Fin r.rows) (j : Fin r.columns) : α :=
  let value := foldRange (fun block x => logicalStep Q r A B i j block x)
    0 r.parameters.numKBlocks 1 Q.zero
  Q.castHalf (Q.activate (r.activation == "leaky_relu") value)

theorem reference_logical {α : Type} [Inhabited α]
    (Q : Numerics α) (r : Request) (initial : State α)
    (A : Fin r.rows → Fin r.inner → α) (B : Fin r.inner → Fin r.columns → α)
    (hA : ∀ i k,inputA r initial i.val k.val=A i k)
    (hB : ∀ k j,inputB r initial k.val j.val=B k j)
    (i : Fin r.rows) (j : Fin r.columns) :
    reference Q r initial i.val j.val=logicalReference Q r A B i j := by
  have hb (block : Nat) (acc : α) : blockStep Q r initial i.val j.val block acc=logicalStep Q r A B i j block acc := by
    unfold blockStep logicalStep
    apply congrArg₂ (fun a b => Q.dotAcc r.config.bk a b acc)
    · funext k
      by_cases hk : block*r.config.bk+k.val<r.inner
      · simp only [hk,if_true,dif_pos]
        exact hA i ⟨block*r.config.bk+k.val,hk⟩
      · simp [hk]
    · funext k
      by_cases hk : block*r.config.bk+k.val<r.inner
      · simp only [hk,if_true,dif_pos]
        exact hB ⟨block*r.config.bk+k.val,hk⟩ j
      · simp [hk]
  unfold reference accumulator logicalReference
  simp_rw [hb]
  rfl

theorem logicalReference_zero_inner {α : Type} (Q : Numerics α) (r : Request)
    (A : Fin r.rows → Fin r.inner → α) (B : Fin r.inner → Fin r.columns → α)
    (i : Fin r.rows) (j : Fin r.columns) (hk : r.inner=0) :
    logicalReference Q r A B i j=Q.castHalf (Q.activate (r.activation == "leaky_relu") Q.zero) := by
  cases hh : r.config <;> simp [logicalReference,hk,foldRange,Request.parameters,hh,Config.bk]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Whole
