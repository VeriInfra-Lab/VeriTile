import bench.repairs.tritonbench_g.decay_cumsum.Proofs.KernelExecution
import bench.repairs.tritonbench_g.decay_cumsum.Proofs.NativeLowering

namespace VeriTile.Bench.Repairs.DecayCumsum.KernelNative
open VeriTile.Triton FP.Structural Geometry Native
noncomputable section

def observe (cell : Cell Node) : Word := (cell.read .real).value
def Outputs (instructions : Instructions) (kind : Kind) (h : Call) (p : Phase)
    (prog : Program h p) (scale : Word) (u v : State Node) : Prop :=
  ∀ r∈FPTile.outputs kind, ∀ (t : Fin p.span) (j : Fin h.BK),
    v.mem r (FPPrefix.off h p prog t.val j)=outputCell kind p.span r t.val
      (evaluateWords instructions kind p.span scale
        (fun s n => (KernelExecution.read h p prog u j s n).value) r t.val)

theorem complete (instructions : Instructions) (kind : Kind) (h : Call) (p : Phase)
    (hp : p∈phases h) (prog : Program h p) (scale : Word) (u : State Node)
    (hf : u.pids 0=prog.feature) (hc : u.pids 1=prog.chunk) (hr : u.pids 2=prog.row) :
    ∃ v,KernelExecution.Facts (algebra instructions selected kind scale) kind h p prog scaleSlot u v ∧
      Outputs instructions kind h p prog scale u v := by
  obtain ⟨v,facts⟩ := KernelExecution.phase_complete (algebra instructions selected kind scale)
    kind h p hp prog scaleSlot u hf hc hr
  refine ⟨v,facts,?_⟩
  intro r member t j
  rw [facts.outputs r member t j,evaluate_lowering]

theorem observe_stored (value : Word) : observe (storedWord value)=value := rfl

#axiomsClean complete
#specNonCircular evaluateWords avoiding [Kernels.source,FP.Structural.exec,FPLogical.evaluate]
end
end VeriTile.Bench.Repairs.DecayCumsum.KernelNative
