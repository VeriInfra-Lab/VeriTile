import bench.tritonbench_g.iv_dependent_matmul.HostDefinitions
import bench.tritonbench_g.iv_dependent_matmul.Proofs.KernelSource
import bench.tritonbench_g.iv_dependent_matmul.Proofs.FPReference

namespace VeriTile.Bench.TritonBenchG.IvDependentMatmul.Whole
open VeriTile.Triton FP.Structural FPValue Kernels
noncomputable section

def instanceState (pid : Nat) (s : State Word) : State Word :=
  {s with pids := fun ax => if ax=0 then pid else 0}

/-- The wrapper's fresh tensors, including the actual FP32 output allocation.
These are allocator/metadata contracts; no output values are assumed. -/
structure Allocation (r : Request) (initial : State Word) where
  freshA : r.C≠r.A
  freshB : r.C≠r.B
  (baseA baseB baseC : Nat)
  physicalFreshA : r.rows*r.inner=0 ∨ r.rows*r.columns=0 ∨
    baseA+4*(r.rows*r.inner)≤baseC ∨ baseC+4*(r.rows*r.columns)≤baseA
  physicalFreshB : r.inner*r.columns=0 ∨ r.rows*r.columns=0 ∨
    baseB+4*(r.inner*r.columns)≤baseC ∨ baseC+4*(r.rows*r.columns)≤baseB
  alignA : baseA%4=0
  alignB : baseB%4=0
  alignC : baseC%4=0
  inputA : ∀ o<r.rows*r.inner, ∃ v, initial.mem r.A o=Cell.mk .fp32 v
  inputB : ∀ o<r.inner*r.columns, ∃ v, initial.mem r.B o=Cell.mk .fp32 v

structure LaunchDescriptor where
  grid : List Nat
  stages : Nat
  selector : String

def launchDescriptor (r : Request) : LaunchDescriptor :=
  ⟨[programs r],r.schedule.stages,r.schedule.name⟩

/-- Launch semantics pass the evolving memory to each actual program instance. -/
def runPrograms [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (D : Dot) (F : DotAcc)
    (g : Graph) (r : Request) : List Nat → State Word → Option (State Word)
  | [],s => some s
  | pid::ids,s => do
    let t ← FP.Structural.exec (backend rest cfg D F) (Kernels.source r.schedule g r.parameters)
      (instanceState pid s)
    runPrograms rest cfg D F g r ids t

def host [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (D : Dot) (F : DotAcc)
    (g : Graph) (r : Request) (initial : State Word) : Option (TensorMetadata × State Word) := do
  let final ← runPrograms rest cfg D F g r (List.range (programs r)) initial
  pure (outputMetadata r,final)

def offset (r : Request) (i j : Nat) := max r.columns 1*i+j

def canonicalPid (r : Request) (i j : Nat) := (i/r.blockRows)*tileColumns r+j/r.blockColumns
def canonicalIndex (r : Request) (i j : Nat) : CIndex r :=
  (⟨i%r.blockRows,Nat.mod_lt _ r.positiveRowsBlock⟩,
   ⟨j%r.blockColumns,Nat.mod_lt _ r.positiveColumnsBlock⟩,PUnit.unit)

def reference [Inhabited Word] (cfg : FP.Config) (D : Dot) (F : DotAcc) (g : Graph)
    (r : Request) (initial : State Word) (i j : Nat) : Word :=
  outputValues cfg r.parameters
    (accumulated cfg D F g r.parameters (instanceState (canonicalPid r i j) initial))
      (canonicalIndex r i j)

def Footprint (r : Request) (region : RegionName) (o : Nat) : Prop :=
  region=r.C ∧ o<r.rows*r.columns

def Good [Inhabited Word] (cfg : FP.Config) (D : Dot) (F : DotAcc) (g : Graph)
    (r : Request) (initial current : State Word) (i j : Nat) : Prop :=
  current.mem r.C (offset r i j)=Cell.mk .fp32 (reference cfg D F g r initial i j)

def Covered (r : Request) (s : State Word) (ids : List Nat) (i j : Nat) : Prop :=
  ∃ pid∈ids, ∃ q : CIndex r, row r.parameters (instanceState pid s) q.1=i ∧
    col r.parameters (instanceState pid s) q.2.1=j

def Result [Inhabited Word] (cfg : FP.Config) (D : Dot) (F : DotAcc) (g : Graph)
    (r : Request) (initial final : State Word) : Prop :=
  (∀ i<r.rows, ∀ j<r.columns, Good cfg D F g r initial final i j) ∧
  (∀ region o, ¬Footprint r region o → final.mem region o=initial.mem region o)

end
end VeriTile.Bench.TritonBenchG.IvDependentMatmul.Whole
