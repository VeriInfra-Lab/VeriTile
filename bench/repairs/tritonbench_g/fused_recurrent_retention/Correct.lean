import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.LogicalCoordinates

namespace VeriTile.Bench.Repairs.FusedRecurrentRetention
open VeriTile.Triton FP.Structural
noncomputable section
open scoped BigOperators VeriTile.Triton.LogicalIO
/-- Both actual FP32 quotations over the complete 3-axis grid. The Real
carrier is mathematical; source masks, optional channels and incoming-gradient
strides remain explicit. Outputs are the kernel partial sums and optional final
state. There is no private scratch or host-reduction claim. Input nodes keep
arbitrary original expression tags; only the output observer takes their values. -/
abbrev retentionIO (mode : Quoted.Mode) (s : Shape) (p : PlanBinding s)
    (incoming finalGradient : Fin 4 → Nat) (seed final : Bool) :=
  PublicData.logicalIO mode s p incoming finalGradient seed final

/-- Independent prefix/suffix formulas at a logical partial coordinate.
`part` is the actual key/value shard; these are no host-reduced outputs. -/
def formula (s : Shape) (x : Inputs ℝ) (seed final : Bool) (kind : PartialKind)
    (part h t c : Nat) : ℝ :=
  match kind with
  | .output => ∑ j∈Finset.range s.BK,if part*s.BK+j<s.DK then
      mathState s x seed h (part*s.BK+j) c (t+1)*(x.q h t (part*s.BK+j)*realArithmetic.scale s.DK) else 0
  | .dq => (∑ j∈Finset.range s.BV,if part*s.BV+j<s.DV then
      mathState s x seed h c (part*s.BV+j) (t+1)*x.do_ h t (part*s.BV+j) else 0)*realArithmetic.scale s.DK
  | .dk => ∑ j∈Finset.range s.BV,if part*s.BV+j<s.DV then
      mathAdjoint s x final h t c (part*s.BV+j)*x.v h t (part*s.BV+j) else 0
  | .dv => ∑ j∈Finset.range s.BK,if part*s.BK+j<s.DK then
      mathAdjoint s x final h t (part*s.BK+j) c*x.k h t (part*s.BK+j) else 0
  | .finalState => mathState s x seed h t c s.T

/-- Named partial tensors and optional terminal state, with all input node tags retained. -/
def expectedOutputs (s : Shape) (seed final : Bool)
    (xs : PublicData.TensorInputs (Staged.Node ℝ) s) : PublicData.TensorOutputs ℝ s :=
  let x := PublicData.tensorData s xs
  { output := fun part h t c => formula s x seed final .output part.val h.val t.val c.val
    dq := fun part h t c => formula s x seed final .dq part.val h.val t.val c.val
    dk := fun part h t c => formula s x seed final .dk part.val h.val t.val c.val
    dv := fun part h t c => formula s x seed final .dv part.val h.val t.val c.val
    finalState := fun h k v => mathState s x seed h.val k.val v.val s.T }

specification kernel_forward_real
    (s : Shape) (shape : s.Admitted) (p : PlanBinding s)
    (incoming finalGradient : Fin 4 → Nat) (seed final : Bool) :
    Spec.Real (retentionIO .forward s p incoming finalGradient seed final ⊨ expectedOutputs s seed final) :=
  PublicData.logical_real_complete .forward s shape p incoming finalGradient seed final

#axiomsClean kernel_forward_real
#stmtSurfaceSubset kernel_forward_real ⊆
  [Shape,Shape.Admitted,PlanBinding,Spec.Real,retentionIO,Quoted.Mode.forward,
   LogicalIO.Implements,PublicData.io,ProgramIO.Observes,realArithmetic,Staged.Node,Staged.Node.value,
   Staged.instInhabitedNode,expectedOutputs,PublicData.inputPorts,PublicData.outputPorts,
   PublicData.TensorInputs,PublicData.TensorOutputs,ChannelValues,IOChannel,MemoryIO.outputs,
   ProgramIO.toMemoryIO,TileIndex,IOChannel.shape,ProgramIO.mapOutputs._proof_1]

specification kernel_backward_real
    (s : Shape) (shape : s.Admitted) (p : PlanBinding s)
    (incoming finalGradient : Fin 4 → Nat) (seed final : Bool) :
    Spec.Real (retentionIO .backward s p incoming finalGradient seed final ⊨ expectedOutputs s seed final) :=
  PublicData.logical_real_complete .backward s shape p incoming finalGradient seed final

#axiomsClean kernel_backward_real
#stmtSurfaceSubset kernel_backward_real ⊆
  [Shape,Shape.Admitted,PlanBinding,Spec.Real,retentionIO,Quoted.Mode.backward,
   LogicalIO.Implements,PublicData.io,ProgramIO.Observes,realArithmetic,Staged.Node,Staged.Node.value,
   Staged.instInhabitedNode,expectedOutputs,PublicData.inputPorts,PublicData.outputPorts,
   PublicData.TensorInputs,PublicData.TensorOutputs,ChannelValues,IOChannel,MemoryIO.outputs,
   ProgramIO.toMemoryIO,TileIndex,IOChannel.shape,ProgramIO.mapOutputs._proof_1]

#specNonCircular formula avoiding [partialValue,state,reverseCarry,adjoint,sourceCell,Quoted.gridExec]
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention
