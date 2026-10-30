import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceModel

/-! A genuine control-flow candidate: expand each admitted single-iteration
forward range into its index assignment followed by the complete body. No
floating expression, cast, reduction, mask, pointer or store is changed. -/
namespace VeriTile.Bench.Repairs.LayerNormTriton.SourceCandidate
open VeriTile.Triton FP.Structural SourceModel
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 12000

/-- The loop iterator is retained in the final register state. -/
def unrollStmt : ComputeStmt → List ComputeStmt
  | .forRange idx 0 _ _ body => [.assign .nat [] idx (.alg (.constNat 0))]++body
  | st => [st]

def admissible : ComputeStmt → Prop
  | .forRange _ start stop stride _ => start=0 ∧ 0<stop ∧ stop≤stride
  | _ => True

def unroll (k : ComputeKernel) : ComputeKernel :=
  .mk k.inputs k.outputs (k.surfaceBody.flatMap unrollStmt)

theorem step_eq [Inhabited α] (m : Algebra α) (st : ComputeStmt)
    (h : admissible st) (s : State α) : run m (unrollStmt st) s=step m st s := by
  cases st <;> try simp [unrollStmt,run,bind]
  case forRange idx start stop stride body =>
    obtain ⟨rfl,hs,hb⟩ := h
    simp only [unrollStmt,List.cons_append,List.nil_append,run,step,
      evalExpr,evalOp_unfold,bind,Option.bind,pure]
    exact (range_one m idx stop stride hs hb body s).symm

theorem run_eq [Inhabited α] (m : Algebra α) (code : List ComputeStmt)
    (h : ∀ st∈code,admissible st) (s : State α) :
    run m (code.flatMap unrollStmt) s=run m code s := by
  induction code generalizing s with
  | nil => rfl
  | cons st rest ih =>
    rw [List.flatMap_cons,run_append,step_eq m st (h st (by simp)),run]
    cases he : step m st s with
    | none => rfl
    | some t => simpa only [Option.bind_some] using ih (fun st hs => h st (by simp [hs])) t

def forward (stride N B : Nat) (eps : ℝ) : ComputeKernel :=
  TypedSource.forwardCandidate "X" "Y" "W" "Bias" "Mean" "Rstd" stride N B eps

theorem forward_source (stride N B : Nat) (eps : ℝ) :
    forward stride N B eps =
      unroll (TypedSource.forward "X" "Y" "W" "Bias" "Mean" "Rstd" stride N B eps) := by rfl

theorem forward_admissible (stride N B : Nat) (eps : ℝ) (hn : 0<N) (hb : N≤B) :
    ∀ st∈(TypedSource.forward "X" "Y" "W" "Bias" "Mean" "Rstd" stride N B eps).surfaceBody,
      admissible st := by
  simp [TypedSource.forward,ComputeKernel.surfaceBody,admissible,hn,hb]

theorem forward_equiv [Inhabited α] (m : Algebra α) (stride N B : Nat) (eps : ℝ)
    (hn : 0<N) (hb : N≤B) (s : State α) :
    FP.Structural.exec m (forward stride N B eps) s=
      FP.Structural.exec m (TypedSource.forward "X" "Y" "W" "Bias" "Mean" "Rstd" stride N B eps) s :=
by
  rw [forward_source]
  exact run_eq m _ (forward_admissible stride N B eps hn hb) s

#axiomsClean forward_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.SourceCandidate
