import bench.repairs.tritonbench_g.decay_cumsum.Proofs.FPLogical

/-! Complete execution of one actual source program. The original full and
short-tail phases share this statement. Success is constructed, and every
stored Cell and every untouched region/tag is retained in the same state. -/
namespace VeriTile.Bench.Repairs.DecayCumsum.KernelExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite Geometry
noncomputable section
variable {α : Type} [Inhabited α]

def read (h : Call) (p : Phase) (prog : Program h p) (u : State α)
    (j : Fin h.BK) (r : RegionName) (n : Nat) : α :=
  (u.mem r (FPPrefix.off h p prog n j)).read .real

def Outputs (M : Algebra α) (kind : Kind) (h : Call) (p : Phase)
    (prog : Program h p) (scale : ℝ) (u v : State α) : Prop :=
  ∀ r∈FPTile.outputs kind, ∀ (t : Fin p.span) (j : Fin h.BK),
    v.mem r (FPPrefix.off h p prog t.val j)=
      FPLogical.evaluate M kind p.span scale (read h p prog u j) r t.val

structure Facts (M : Algebra α) (kind : Kind) (h : Call) (p : Phase)
    (prog : Program h p) (scale : ℝ) (u v : State α) : Prop where
  success : FP.Structural.exec (engine M) (Kernels.source kind h p scale) u=some v
  outputs : Outputs M kind h p prog scale u v
  frame : FPMemory.Frame h p prog (FPTile.outputs kind) u v

theorem complete (M : Algebra α) (kind : Kind) (h : Call) (p : Phase)
    (positive : 0<p.span) (prog : Program h p) (scale : ℝ) (u : State α)
    (hf : u.pids 0=prog.feature) (hc : u.pids 1=prog.chunk) (hr : u.pids 2=prog.row) :
    ∃ v,Facts M kind h p prog scale u v := by
  obtain ⟨v,he,out,frame⟩ := FPTile.complete M kind h p positive prog scale u hf hc hr
  refine ⟨v,he,?_,frame⟩
  intro r member t j
  rw [out r member t j]
  exact FPLogical.tile_evaluate M kind h p positive prog scale u (read h p prog u j) t j
    (by intros; rfl) r

theorem phase_complete (M : Algebra α) (kind : Kind) (h : Call) (p : Phase)
    (hp : p∈phases h) (prog : Program h p) (scale : ℝ) (u : State α)
    (hf : u.pids 0=prog.feature) (hc : u.pids 1=prog.chunk) (hr : u.pids 2=prog.row) :
    ∃ v,Facts M kind h p prog scale u v :=
  complete M kind h p (phase_positive h p hp).2.1 prog scale u hf hc hr

#axiomsClean complete
#axiomsClean phase_complete
end
end VeriTile.Bench.Repairs.DecayCumsum.KernelExecution
