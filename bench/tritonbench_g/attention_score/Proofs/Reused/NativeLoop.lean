import bench.tritonbench_g.attention_score.Proofs.Reused.NativeFormula
import bench.tritonbench_g.attention_score.Proofs.Reused.FPReturned

namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.NativeLoop
open VeriTile Triton FP.Structural NativeInstructions NativeStaging Geometry
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 2000000

theorem fold_lower (B D : Nat) (p : RealSource.Parameters) (s : State Expr)
    (cur stop stride : Nat) (acc : Fin B → Expr) :
    (fun i => lower (foldRange (FPValues.update algebra dotModel B D p s) cur stop stride acc i))=
      foldRange (NativeFormula.update B D p s) cur stop stride (fun i => lower (acc i)) := by
  by_cases hz : stride=0
  · simp [foldRange,hz]
  · by_cases hc : cur<stop
    · have hstep : (fun i => lower (FPValues.update algebra dotModel B D p s cur acc i))=
          NativeFormula.update B D p s cur (fun i => lower (acc i)) :=
        funext (NativeFormula.update_lower B D p s cur acc)
      have hl : foldRange (FPValues.update algebra dotModel B D p s) cur stop stride acc=
          foldRange (FPValues.update algebra dotModel B D p s) (cur+stride) stop stride
            (FPValues.update algebra dotModel B D p s cur acc) := by
        conv_lhs => rw [foldRange]
        simp [hz,hc]
      have hr : foldRange (NativeFormula.update B D p s) cur stop stride (fun i => lower (acc i))=
          foldRange (NativeFormula.update B D p s) (cur+stride) stop stride
            (NativeFormula.update B D p s cur (fun i => lower (acc i))) := by
        conv_lhs => rw [foldRange]
        simp [hz,hc]
      rw [hl,hr,←hstep]
      exact fold_lower B D p s (cur+stride) stop stride (FPValues.update algebra dotModel B D p s cur acc)
    · simp [foldRange,hz,hc]
termination_by stop-cur
decreasing_by omega

theorem accumulated_lower (B D : Nat) (p : RealSource.Parameters) (s : State Expr) (i : Fin B) :
    lower (FPLoop.accumulated algebra dotModel B D p s i)=NativeFormula.accumulated B D p s i :=
  congrFun (fold_lower B D p s 0 p.queryContext B (fun _ => FPContext.zero algebra)) i

def returnedGraph (c : Shape) (s : State Expr) (offset size : Int) (scale : ℝ)
    (window complement : Bool) (z : Fin c.Z) (h : Fin c.H) (j : Fin c.NK) : Graph :=
  NativeFormula.accumulated c.B c.D (parameters c offset size scale window complement)
    (FPGrid.programState c s (FPReturned.ownerJob c z h j)) (FPReturned.ownerLane c j)

theorem returned_lower (c : Shape) (s : State Expr) (offset size : Int) (scale : ℝ)
    (window complement : Bool) (z : Fin c.Z) (h : Fin c.H) (j : Fin c.NK) :
    lower (FPReturned.returnedValue algebra dotModel c s offset size scale window complement z h j)=
      returnedGraph c s offset size scale window complement z h j := by
  by_cases hn : c.N=0
  · simp [FPReturned.returnedValue,hn,FPContext.zero,algebra,lower,returnedGraph,
      NativeFormula.accumulated,parameters,foldRange,NativeFormula.zero32]
  · simp only [FPReturned.returnedValue,hn,if_false,FPReturned.target,FPGrid.values]
    change lower (FPLoop.accumulated algebra dotModel c.B c.D _ _ _)=_
    exact accumulated_lower _ _ _ _ _

#specNonCircular returnedGraph avoiding [FP.Structural.exec,FPGrid.run,FPReturned.hostRun,FPLoop.accumulated]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.NativeLoop
