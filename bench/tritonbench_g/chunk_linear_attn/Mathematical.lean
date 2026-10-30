import bench.tritonbench_g.chunk_linear_attn.IOLayout

namespace VeriTile.Bench.Repairs.ChunkLinearAttn.Mathematical
open VeriTile Triton FP.Structural
noncomputable section

def inputs (c : KernelInterface.Config) (vals : ChannelValues ℝ (IOLayout.inputs c.dims))
    (head : Fin (c.dims.B*c.dims.H)) : Real.Inputs c.dims.T c.dims.K c.dims.V where
  q := fun t k => vals 0 (head,t,k,PUnit.unit)
  k := fun t k => vals 1 (head,t,k,PUnit.unit)
  v := fun t v => vals 2 (head,t,v,PUnit.unit)
  initial := fun k v => if c.seed then vals 4 (head,k,v,PUnit.unit) else 0

def incoming (c : KernelInterface.Config) (vals : ChannelValues ℝ (IOLayout.inputs c.dims))
    (head : Fin (c.dims.B*c.dims.H)) : Fin c.dims.T → Fin c.dims.V → ℝ :=
  fun t v => vals 3 (head,t,v,PUnit.unit)

def terminal (c : KernelInterface.Config) (vals : ChannelValues ℝ (IOLayout.inputs c.dims))
    (head : Fin (c.dims.B*c.dims.H)) : Fin c.dims.K → Fin c.dims.V → ℝ :=
  fun k v => if c.terminal then vals 5 (head,k,v,PUnit.unit) else 0

def gradient (c : KernelInterface.Config) (vals : ChannelValues ℝ (IOLayout.inputs c.dims))
    (head : Fin (c.dims.B*c.dims.H)) : Real.Inputs c.dims.T c.dims.K c.dims.V :=
  Real.gradients c.scale (inputs c vals head) (incoming c vals head) (terminal c vals head)

/-- Independent causal attention, its full VJP, and both optional state
endpoints. The initial and terminal branches follow the source flags. -/
def real (c : KernelInterface.Config) (vals : ChannelValues ℝ (IOLayout.inputs c.dims)) :
    ChannelValues ℝ (IOLayout.outputs c.dims) :=
  Fin.cases (fun j => Real.output c.scale (inputs c vals j.1) j.2.1 j.2.2.1)
    (Fin.cases (fun j => (gradient c vals j.1).q j.2.1 j.2.2.1)
      (Fin.cases (fun j => (gradient c vals j.1).k j.2.1 j.2.2.1)
        (Fin.cases (fun j => (gradient c vals j.1).v j.2.1 j.2.2.1)
          (Fin.cases (fun j => Real.finalState (inputs c vals j.1) j.2.1 j.2.2.1)
            (Fin.cases (fun j => (gradient c vals j.1).initial j.2.1 j.2.2.1)
              (fun i => Fin.elim0 i))))))

/-- The complete derivative statement is retained independently of execution. -/
theorem is_vjp (c : KernelInterface.Config) (vals : ChannelValues ℝ (IOLayout.inputs c.dims))
    (head : Fin (c.dims.B*c.dims.H)) :
    Real.IsVJP c.scale (inputs c vals head) (incoming c vals head) (terminal c vals head)
      (gradient c vals head) :=
  Real.gradients_correct _ _ _ _

#specNonCircular real avoiding [KernelExecution.run,KernelExecution.RealResult,FP.Structural.exec,
  runGrid,executeFour,gridMemory,stageMemory,modelFour,completeMemory]
#axiomsClean is_vjp
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.Mathematical
