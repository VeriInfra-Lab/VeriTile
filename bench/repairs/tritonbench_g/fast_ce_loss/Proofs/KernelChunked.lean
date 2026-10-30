import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.KernelForward

namespace VeriTile.Bench.Repairs.FastCeLoss.KernelExecution
open VeriTile.Triton FP.Structural KernelSource KernelModel
noncomputable section
set_option maxHeartbeats 24000000
set_option maxRecDepth 30000
set_option maxErrors 8
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

@[simp] theorem natural_block_address (a b c : Nat) :
    ((a : Int)*(b : Int)+(c : Int)).toNat=a*b+c := by norm_cast

@[simp] theorem natural_block_less (a b c d : Nat) :
    ((a : Int)*(b : Int)+(c : Int)<(d : Int)) ↔ a*b+c<d := by norm_cast

def chunkedMemory (M : Algebra α) (p : Parameters) (initial : State α) : State α :=
  let afterLoss := if initial.pids 1=0 then
    initial.write p.loss (initial.pids 0) (.mk .real (KernelModel.cast M (chunkLoss M p initial)))
    else initial
  afterLoss.write p.lse (initial.pids 0*p.chunks+initial.pids 1)
    (.mk .real (KernelModel.cast M (lseValue M p initial (initial.pids 1))))

/-- Every chunk writes its LSE. The first chunk alone writes the negative
target contribution, with literal zero for ignored labels. -/
theorem chunked_executes (M : Algebra α) (p : Parameters) (initial : State α)
    (positive : 0<p.block) (validLabel : label p initial = -100 ∨ 0≤label p initial) :
    ∃ final, exec (engine M) (chunked p) initial=some final ∧
      final.mem=(chunkedMemory M p initial).mem := by
  classical
  cases hc : p.hasSoftcap <;> cases hs : p.hasScaling
  all_goals
    by_cases first : initial.pids 1=0
    all_goals
      by_cases ignored : label p initial = -100
      all_goals
        have readIgnored := ignored
        simp only [label,Region.cast] at readIgnored
        refine ⟨?_,?_,?_⟩
        rotate_left
        · simp [exec,chunked,TypedSource.chunked,hc,hs,ComputeKernel.surfaceBody,
            run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
            engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
            numeric,natLt,numericLt,numericLe,numericEq,bop,ofFloat,toFloat,Region.cast,
            State.setReg,FP.Structural.store,TileShape.axisDim,positive,Function.comp_def,
            first,readIgnored]
          rfl
        · funext region offset
          simp only [chunkedMemory,chunkLoss,ignored,first,if_true,if_false]
          dsimp only [State.write,lseValue,lseTile,maxValue,maxTile,forwardLogits,loaded,
            forwardScale,cap,capPartial,literal,KernelModel.cast,binary,unary,chunkTarget,label]
          simp only [hc,hs,Bool.false_eq_true,↓reduceIte,Nat.zero_mul,Nat.zero_add,
            decimal_one,decimal_zero,bop,Broadcast.leftIndex,Broadcast.rightIndex,first]
          rfl

end
end VeriTile.Bench.Repairs.FastCeLoss.KernelExecution
