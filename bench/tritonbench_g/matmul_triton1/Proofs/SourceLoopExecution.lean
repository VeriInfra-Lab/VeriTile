import bench.tritonbench_g.matmul_triton1.Proofs.SourceBodyInvariant
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.MatmulTriton1.SourceRuntime
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulTriton1.Revision001
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem loopStatement_eq (p : Core.Parameters) :
    loopStatement p = .forRangeDyn "k" (.constNat 0)
      (.div .nat .nil (.sub .nat .nil (.add .nat .nil (.ref .nat [] "k_size")
        (.constNat p.blockInner)) (.constNat 1)) (.constNat p.blockInner))
      (.constNat 1) (loopBody p) := by rfl

theorem dynamicLoop (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters)
    (initial current : State (Word α)) (block : Nat) (h : RuntimeInvariant p initial block current) :
    step (VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.algebra rest P) (loopStatement p) current =
      FP.Structural.range (VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.algebra rest P) "k" 0
        (Core.steps p) 1 (loopBody p) current := by
  have hk := h.metadata "k_size" p.innerSize (by simp [VeriTile.Bench.Repairs.MatmulTriton1.Kernel.scalarParameters])
  rw [loopStatement_eq]
  simp [step,evalOp_unfold,numeric,bop,Option.bind,Core.steps,Core.inner,hk]

theorem loopExecution (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters)
    (initial current : State (Word α)) (block : Nat) (hk : block ≤ Core.steps p)
    (h : RuntimeInvariant p initial block current) :
    ∃ final, FP.Structural.range (VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.algebra rest P) "k" block
        (Core.steps p) 1 (loopBody p) current = some final ∧
      RuntimeInvariant p initial (Core.steps p) final := by
  have aux : ∀ fuel block (current : State (Word α)), Core.steps p-block=fuel → block≤Core.steps p →
      RuntimeInvariant p initial block current →
      ∃ final, FP.Structural.range (VeriTile.Bench.Repairs.MatmulTriton1.DotProfile.algebra rest P) "k" block
          (Core.steps p) 1 (loopBody p) current=some final ∧
        RuntimeInvariant p initial (Core.steps p) final := by
    intro fuel
    induction fuel using Nat.strong_induction_on with
    | h fuel ih =>
      intro block current he hk h
      by_cases hb : block < Core.steps p
      · let next := bodyExecution rest P p initial current block h
        obtain ⟨final,hr,hf⟩ := ih (Core.steps p-(block+1)) (by omega) (block+1) next.val rfl
          (by omega) (bodyInvariant p initial current block h)
        refine ⟨final,?_,hf⟩
        rw [FP.Structural.range,if_neg (by decide),if_pos hb,next.property]
        exact hr
      · have hb : block=Core.steps p := by omega
        subst block
        refine ⟨current,?_,h⟩
        rw [FP.Structural.range,if_neg (by decide),if_neg (by omega)]
  exact aux _ block current rfl hk h

#axiomsClean loopStatement_eq
#axiomsClean dynamicLoop
#axiomsClean loopExecution
end
end VeriTile.Bench.Repairs.MatmulTriton1.SourceRuntime
