import bench.repairs.tritonbench_g.attention_fwd_triton2.KernelOnly.Raw

namespace VeriTile.Bench.Repairs.AttentionFwdTriton2.RealCompatibility
open VeriTile Triton FP.Structural Whole001 Source Whole Machine KernelInterface
noncomputable section
set_option maxHeartbeats 6000000

/-- Complete original source grid, independent normalized base-2 attention and
full frame over raw memory, scalar geometry, pointers and input reads. -/
theorem kernel_raw_correct (p : Parameters) (bases : Bases) (dom : Domain p bases)
    (sep : p.Out≠p.Q ∧ p.Out≠p.K ∧ p.Out≠p.V ∧ p.Out≠p.QS ∧ p.Out≠p.KS)
    (before : State Real.R) (values : RegionName → Nat → ℝ) (reads : Real.Reads p before values)
    (work : List Program) (order : (programs (grid p)).Perm work) :
    ∃ after,runPrograms (fun _=>Real.algebra) (fun _=>Real.dotModel)
        p (grid p) work before=some after ∧
      (∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d<96,
        after.mem p.Out (offset p.o b h n d)=.mk .bf16 (some (RealReturned.target p values b h n d))) ∧
      RawFrame p before after ∧ Access.Safe p ∧ Closed p bases := by
  obtain ⟨after,ran,outputs,frame⟩ := complete_raw Real.algebra Real.dotModel p bases dom sep before work order
  refine ⟨after,ran,?_,frame,Access.safe p dom.canonical,Machine.closed p bases dom⟩
  intro b hb h hh n hn d hd
  rw [outputs b hb h hh n hn d hd]
  have input : Real.Reads p (start (grid p) (n/128,b*p.heads+h,0) before) values := reads
  have active : Source.active p (start (grid p) (n/128,b*p.heads+h,0) before) (lane n d hd) := by
    change n/128*128+n%128<p.length ∧ d<96
    exact ⟨by omega,hd⟩
  rw [Real.answer_real p _ values input _ active,RealReturned.target_at_program p before values b h n d hh hd]

#axiomsClean kernel_raw_correct
end
end VeriTile.Bench.Repairs.AttentionFwdTriton2.RealCompatibility
