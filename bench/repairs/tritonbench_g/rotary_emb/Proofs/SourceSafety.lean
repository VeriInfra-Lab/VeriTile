import bench.repairs.tritonbench_g.rotary_emb.Proofs.Typed
import bench.repairs.tritonbench_g.rotary_emb.Proofs.Candidate
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.RotaryEmb.Typed
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- Coefficient loads have no head mask. Their actual source tile has singleton
head dimension, even when every Q/K head of this program is inactive. -/
abbrev CoefficientLane (p : Kernels.Layout) := TileIndex [p.BLOCK_SEQ,1,p.BLOCK_PAIRS]

def coefficientActive {A : Type} (p : Kernels.Layout) (s : State A)
    (i : CoefficientLane p) : Prop :=
  s.pids 1*p.BLOCK_SEQ+i.1.val<p.max_total_len ∧ i.2.2.1.val*2+1<p.ROTARY_DIM

def coefficientOffset {A : Type} (p : Kernels.Layout) (s : State A)
    (sine odd : Bool) (i : CoefficientLane p) : Nat :=
  (s.pids 1*p.BLOCK_SEQ+i.1.val)*(if sine then p.stride_sinbs else p.stride_cosbs)+
    (i.2.2.1.val*2+if odd then 1 else 0)*(if sine then p.stride_sind else p.stride_cosd)

/-- Access safety of every actual original load/store, including coefficient
reloads after the Q stores. This is independent of floating arithmetic laws. -/
theorem source_trace_safe {A : Type} [Inhabited A] (rest : Algebra A) (ops : Ops A)
    (p : Kernels.Layout) (s : State A) (bounds : RegionName → Nat)
    (payload : ∀ key odd i,active p s key i → offset p s key odd i<bounds (region p key))
    (coefficients : ∀ sine odd i,coefficientActive p s i →
      coefficientOffset p s sine odd i<bounds (if sine then p.Sin else p.Cos)) :
    TraceSafe (algebra rest ops) bounds (Kernels.kernel p) s := by
  simp (config := { maxSteps := 1000000, zetaDelta := Bool.false }) only
    [TraceSafe,Kernels.kernel,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
    Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,Option.bind,
    algebra,numericLt,natLt,FP.Structural.bop,numeric,store,VeriTile.Triton.Region.cast,
    ComputeDType.eraseDType,ofFloat,toFloat,TileShape.dropInsertedIndex,
    int_double,int_increment,←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast,Int.ofNat_lt,
    State.setReg_mem,State.setReg_pids,State.setReg_same,State.setReg_ne_name,
    pure,Bind.bind,Option.bind,Option.map,Functor.map,true_and,and_true]
  simp (config := { maxSteps := 1000000, zetaDelta := Bool.false })
  dsimp only [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
    TileShape.insertAxis,TileShape.dropInsertedIndex]
  simp only [int_double,int_increment,←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast,
    Int.ofNat_lt,Bool.and_eq_true,decide_eq_true_eq,pair_projection]
  have q0 := payload Bool.false Bool.false
  have q1 := payload Bool.false Bool.true
  have k0 := payload Bool.true Bool.false
  have k1 := payload Bool.true Bool.true
  have c0 := coefficients Bool.false Bool.false
  have c1 := coefficients Bool.false Bool.true
  have z0 := coefficients Bool.true Bool.false
  have z1 := coefficients Bool.true Bool.true
  simp only [active,offset,region,token,head,dimension,coefficientActive,coefficientOffset,
    Bool.false_eq_true,ite_true,ite_false,Nat.add_zero,Prod.forall,Fin.forall_fin_one,
    and_assoc] at q0 q1 k0 k1 c0 c1 z0 z1
  simpa only [and_assoc] using
    (show _ from ⟨q0,q1,c0,z0,c1,z1,q0,q1,k0,k1,c0,z0,c1,z1,k0,k1⟩)

/-- The candidate's pure integer prefix has no memory effects; its suffix
starts in the same actual state as the original prefix. -/
theorem candidate_trace_safe {A : Type} [Inhabited A] (rest : Algebra A) (ops : Ops A)
    (p : Kernels.Layout) (s : State A) (bounds : RegionName → Nat)
    (source : TraceSafe (algebra rest ops) bounds (Kernels.kernel p) s) :
    TraceSafe (algebra rest ops) bounds (Candidate.optimized p) s := by
  change Safety.traceSafeList (algebra rest ops) bounds
    ((Candidate.optimizedPrefix p).surfaceBody++(Kernels.kernel p).surfaceBody.drop 3) s
  apply (Safety.traceSafeList_append _ _ _ _ _).mpr
  constructor
  · simp [Candidate.optimizedPrefix,ComputeKernel.surfaceBody,
      Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
      Safety.safeCompute_unfold,Safety.safeOp_unfold,run,step,evalExpr,evalComputeOp,
      evalOp_unfold,Option.bind]
  · intro t executed
    have whole : Safety.traceSafeList (algebra rest ops) bounds
        ((Kernels.kernel p).surfaceBody.take 3++(Kernels.kernel p).surfaceBody.drop 3) s := by
      simpa only [List.take_append_drop] using source
    exact ((Safety.traceSafeList_append _ _ _ _ _).mp whole).2 t
      ((Candidate.prefix_equiv _ _ _).trans executed)

end
end VeriTile.Bench.Repairs.RotaryEmb.Typed
