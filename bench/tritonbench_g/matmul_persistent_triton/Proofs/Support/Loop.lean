import bench.tritonbench_g.matmul_persistent_triton.Proofs.Support.Inner
import bench.tritonbench_g.matmul_persistent_triton.Proofs.Support.Registers

namespace VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.Core
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulPersistentTriton
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem bodyInvariant (P : Source.Arithmetic α α) (p : Parameters)
    (initial current : State α) (tile block : Nat) (h : InnerInvariant P p initial tile block current) :
    InnerInvariant P p initial tile (block+1) (bodyResult P p initial current tile block) := by
  constructor <;> (try apply Arguments.mk) <;> simp [bodyResult,h.memory,h.offCM,h.offCN,h.offAM,h.offBN,h.offK,h.args.M,h.args.N,h.args.K,h.args.stride_am,h.args.stride_ak,h.args.stride_bk,h.args.stride_bn,h.args.stride_cm,h.args.stride_cn]

theorem bodyRegisters (P : Source.Arithmetic α α) (p : Parameters)
    (initial current : State α) (pid tile block : Nat) (h : Registers p pid current) :
    Registers p pid (bodyResult P p initial current tile block) := by
  constructor <;> (try apply Arguments.mk) <;> simp [bodyResult,h.start,h.rows,h.columns,h.count,h.work,h.group,h.span,h.lanesM,h.lanesN,h.lanesK,h.args.M,h.args.N,h.args.K,h.args.stride_am,h.args.stride_ak,h.args.stride_bk,h.args.stride_bn,h.args.stride_cm,h.args.stride_cn]

theorem innerLoop (P : Source.Arithmetic α α) (p : Parameters)
    (initial current : State α) (pid tile block : Nat) (hb : block≤p.problem.kTiles)
    (h : InnerInvariant P p initial tile block current) (regs : Registers p pid current) :
    ∃ final,rangeInt (backend P) "ki" (block : Int) (p.problem.kTiles : Int) 1 (inner p) current=some final ∧
      InnerInvariant P p initial tile p.problem.kTiles final ∧ Registers p pid final := by
  have aux : ∀ fuel block (current : State α),p.problem.kTiles-block=fuel → block≤p.problem.kTiles →
      InnerInvariant P p initial tile block current → Registers p pid current →
      ∃ final,rangeInt (backend P) "ki" (block : Int) (p.problem.kTiles : Int) 1 (inner p) current=some final ∧
        InnerInvariant P p initial tile p.problem.kTiles final ∧ Registers p pid final := by
    intro fuel
    induction fuel using Nat.strong_induction_on with
    | h fuel ih =>
      intro block current he hb h regs
      by_cases active : block<p.problem.kTiles
      · let next := bodyResult P p initial current tile block
        obtain ⟨final,ran,invariant,registers⟩ := ih (p.problem.kTiles-(block+1)) (by omega)
          (block+1) next rfl (by omega) (bodyInvariant P p initial current tile block h)
          (bodyRegisters P p initial current pid tile block regs)
        refine ⟨final,?_,invariant,registers⟩
        rw [rangeInt,if_neg (by decide),if_pos (by exact_mod_cast active),bodyExecution P p initial current tile block h]
        simpa only [Nat.cast_add,Nat.cast_one] using ran
      · have equal : block=p.problem.kTiles := by omega
        subst block
        refine ⟨current,?_,h,regs⟩
        rw [rangeInt,if_neg (by decide),if_neg (by omega)]
  exact aux _ block current rfl hb h regs

def innerStatement (p : Parameters) : ComputeStmt :=
  (outer p)[13]'(by cases hp : p.problem.profile <;> simp [outer,source,TypedSource.kernel,TypedSource.fp16,TypedSource.f8e4,TypedSource.originalKernel,TypedSource.candidateKernel,hp,ComputeKernel.surfaceBody])

theorem dynamicInner (P : Source.Arithmetic α α) (p : Parameters)
    (current : State α) (pid : Nat) (h : Registers p pid current) :
    step (backend P) (innerStatement p) current=
      rangeInt (backend P) "ki" 0 (p.problem.kTiles : Int) 1 (inner p) current := by
  cases hp : p.problem.profile <;>
    simp [innerStatement,inner,outer,source,TypedSource.kernel,TypedSource.fp16,TypedSource.f8e4,TypedSource.originalKernel,TypedSource.candidateKernel,hp,ComputeKernel.surfaceBody,step,
      evalOp_unfold,evalExpr,evalComputeOp,Option.bind,h.count]

#axiomsClean innerLoop
#axiomsClean dynamicInner
end
end VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.Core
