import bench.tritonbench_g.attn_fwd_triton.Proofs.TypedBeforeTail
namespace VeriTile.Bench.Repairs.AttnFwdTriton.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def Written (p : Config) (BD pm ph : Nat) (region : RegionName) (offset : Nat) : Prop :=
  region=p.Out ∧ ∃ i : TileIndex [128,BD], outputActive p BD pm i=Bool.true ∧
    offset=outputAddress p BD pm ph i

def Frame {α : Type} (p : Config) (BD pm ph : Nat) (s t : State α) : Prop :=
  ∀ region offset, ¬Written p BD pm ph region offset → t.mem region offset=s.mem region offset

theorem tail_frame {α : Type} [Inhabited α] (A : Algebra α)
    (p : Config) (BD pm ph : Nat) (r : Bool) (s t : State α)
    (hs : Ready BD s) (ha : AddressReady p BD pm ph s)
    (ht : run (engine A) (tail p BD r) s=some t) : Frame p BD pm ph s t := by
  rcases hs with ⟨_,_,_,_,_,_,_,_,_,_,_,_,_,⟨li,hli⟩,⟨acc,hacc⟩⟩
  rcases ha with ⟨hn,ho,hp⟩
  intro region offset hnot
  have hmiss : region≠p.Out ∨ ∀ i : TileIndex [128,BD],
      i∈TileShape.allIndices [128,BD] → outputActive p BD pm i=Bool.true →
      offset≠outputAddress p BD pm ph i := by
    by_cases hr : region=p.Out
    · exact Or.inr (fun i _ hi he => hnot ⟨hr,i,hi,he⟩)
    · exact Or.inl hr
  cases r
  all_goals simp only [tail,source,Bool.false_eq_true,Bool.true_eq,↓reduceIte,
    TypedSource.repaired,TypedSource.reciprocal,ComputeKernel.surfaceBody,List.drop_succ_cons,List.drop_zero] at ht
  all_goals simp [run,step,store,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,natLt,numericLt,
    Cell.read,ofFloat,toFloat,Region.cast,State.setReg,hn,ho,hp,hli,hacc] at ht
  all_goals subst t
  all_goals
    apply State.masked_scatter_frame
    simpa [outputActive,bop] using hmiss

/-- Complete opaque source run and exact syntactic output footprint. -/
theorem executes_framed {α : Type} [Inhabited α] (A : Algebra α)
    (p : Config) (BD : Nat) (support : DotSupport A BD) (r : Bool) (s : State α) :
    ∃ t, exec (engine A) (source p BD r) s=some t ∧
      Frame p BD (s.pids 0) (s.pids 1) s t := by
  obtain ⟨u,hu,hr,ha,hm⟩ := before_tail A p BD support r s
  obtain ⟨t,ht⟩ := tail_success A p BD r u hr
  have hf := tail_frame A p BD (s.pids 0) (s.pids 1) r u t hr ha ht
  refine ⟨t,?_,?_⟩
  · change run (engine A) (source p BD r).surfaceBody s=some t
    rw [←List.take_append_drop 41 (source p BD r).surfaceBody,run_append,hu]
    exact ht
  · intro region offset hnot
    exact (hf region offset hnot).trans (congrFun (congrFun hm region) offset)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.TypedExecution
