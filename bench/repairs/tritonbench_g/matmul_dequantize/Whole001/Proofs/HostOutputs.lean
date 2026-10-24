import bench.repairs.tritonbench_g.matmul_dequantize.Whole001.Proofs.TorchMM
namespace VeriTile.Bench.Repairs.MatmulDequantizeWhole001.Host
open VeriTile.Triton FP.Structural
noncomputable section
/-- The ordinary allocator/autotuner contract affects only its exact output
footprint. The GPU theorems begin at this prepared state and overwrite every
logical output, so neither empty/ones/zero contents are numerical premises. -/
def Initialized {α : Type} (a : Arguments) (route : Route) (before ready : State α) : Prop :=
  let t := if route=.dequant32 ∨ route=.dequant16 ∨ route=.s1 then a.d else a.c
  ∀ r o, (r≠t.region ∨ ∀ m<t.rows, ∀ n<t.columns, o≠m*t.rowStride+n*t.columnStride) →
    ready.mem r o=before.mem r o
structure Observation (α : Type) where
  state : State α
  returnedIdentity : Option Nat
  outputDescriptor : Tensor
  localScratchReleased : Bool

def observe {α : Type} (a : Arguments) (route : Route) (s : State α) : Observation α :=
  ⟨s,returned route a,if route=.dequant32 ∨ route=.dequant16 then a.d else a.c,decide (route=.s1)⟩
def complete {α : Type} (a : Arguments) (route : Route) (action : Option (State α)) : Option (Observation α) :=
  action.map (observe a route)

theorem first_return (a : Arguments) (s : State α) :
    (observe a .gptq s).returnedIdentity=(if a.supplied then none else some a.c.identity) ∧
    (observe a .gptq s).outputDescriptor=a.c := by simp [observe,returned]
theorem second_return (a : Arguments) (s : State α) :
    (observe a .s2 s).returnedIdentity=some a.c.identity ∧
    (observe a .s2 s).outputDescriptor=a.c := by simp [observe,returned]
theorem helper_return (a : Arguments) (s : State α) (route : Route) (hr : route=.dequant16 ∨ route=.dequant32) :
    (observe a route s).returnedIdentity=some a.d.identity ∧
    (observe a route s).outputDescriptor=a.d := by rcases hr with rfl|rfl <;> simp [observe,returned]
theorem s1_return (a : Arguments) (s : State α) :
    (observe a .s1 s).returnedIdentity=some a.c.identity ∧
    (observe a .s1 s).outputDescriptor=a.c ∧
    (observe a .s1 s).localScratchReleased=Bool.true := by simp [observe,returned]
#axiomsClean first_return
#axiomsClean second_return
#axiomsClean helper_return
#axiomsClean s1_return
end
end VeriTile.Bench.Repairs.MatmulDequantizeWhole001.Host
