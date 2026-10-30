import bench.tritonbench_g.attention_llama.Proofs.Iteration
import bench.tritonbench_g.attention_llama.Proofs.CarrySource

/-! Constructive execution of the actual carry-first candidate. Source.Term,
Carries, Context, all input views, coordinates and frames are shared with the
original; only the normalizer addition order changes in the source graph. -/
namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.CarrySource
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open Source
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

def iteration_of_reached (c : Config) (s t : State (Term α)) (scale : Term α) (n : Nat)
    (a : Carries α c.D) (h : Context c s scale n a t) (bytes : ReachedBytes c s) :
    {u : State (Term α) // run algebra (parts c).body
      (t.setReg "start_n" .nat [] (fun _=>n*64))=some u} := by
  cases hf : c.fp8
  ·
    cases hc : c.causal
    all_goals
      refine ⟨?_,?_⟩
      swap
      simp [parts,kernel,hf,hc,TypedSource.carryFirstHalf,TypedSource.carryFirstBytes,
        ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,Option.map,Option.getD,
        run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,algebra,
        ComputeDType.eraseDType,
        numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
        TileShape.axisDim,Region.cast,State.setReg,
        h.scalar,h.maximum,h.denominator,h.accumulator,h.query,h.rows,h.columns,
        h.batch,h.head,h.key,h.value,h.memory,h.pids]
      rfl
  · have hk : ∀ (a : Fin 64) (b : Fin c.D), Float8E5Bits.isInt8
        (if n*64+a.val<c.N then
          (s.mem c.K (pointers c s c.K c.skz c.skh c.skn c.skk n (a,b,())).2).read .int else 0)=Bool.true :=
      fun a b=>masked_key_cell c s bytes n (a,b,()) hf
    have hv : ∀ (a : Fin 64) (b : Fin c.D), Float8E5Bits.isInt8
        (if n*64+a.val<c.N then
          (s.mem c.V (pointers c s c.V c.svz c.svh c.svk c.svn n (a,b,())).2).read .int else 0)=Bool.true :=
      fun a b=>masked_value_cell c s bytes n (a,b,()) hf
    cases hc : c.causal
    all_goals
      refine ⟨?_,?_⟩
      swap
      simp [parts,kernel,hf,hc,TypedSource.carryFirstHalf,TypedSource.carryFirstBytes,
        ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,Option.map,Option.getD,
        run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,algebra,
        ComputeDType.eraseDType,
        numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
        TileShape.axisDim,Region.cast,State.setReg,
        h.scalar,h.maximum,h.denominator,h.accumulator,h.query,h.rows,h.columns,
        h.batch,h.head,h.key,h.value,h.memory,h.pids,hk,hv]
      rfl

def iteration (c : Config) (s t : State (Term α)) (scale : Term α) (n : Nat)
    (a : Carries α c.D) (h : Context c s scale n a t) (bytes : ByteRange s) :
    {u : State (Term α) // run algebra (parts c).body
      (t.setReg "start_n" .nat [] (fun _=>n*64))=some u} :=
  iteration_of_reached c s t scale n a h (bytes.reached c s)

#axiomsClean iteration_of_reached
#axiomsClean iteration
end
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.CarrySource
