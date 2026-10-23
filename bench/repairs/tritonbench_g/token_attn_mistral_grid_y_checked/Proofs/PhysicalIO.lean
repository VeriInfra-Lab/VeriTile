import bench.repairs.tritonbench_g.token_attn_mistral_grid_y_checked.Proofs.PublicIO
import Mathlib.Tactic.FinCases

namespace VeriTile.Bench.Repairs.TokenAttnMistralGridYChecked.PhysicalIO
open VeriTile Triton FP.Structural TokenAttnMistral
noncomputable section
set_option maxRecDepth 10000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Every original heap supplies its own finite metadata snapshot. No upper
bound on attention length, positivity of width, or typed-leaf condition is added. -/
def counts (p : Parameters) (work : List Grid.PID) (m : Grid.Memory α) (j : Fin work.length) :=
  (m p.attentionLength.cast (work[j].1*p.attentionLengthStride)).read .nat

def values (p : Parameters) (work : List Grid.PID) (m : Grid.Memory α) :
    Ports.Values α p work (counts p work m) :=
  fun i=>Fin.cases (fun j=>(m p.length.cast (work[j.1].1*p.sequenceStride)).read .nat)
    (Fin.cases (fun j=>(m p.start.cast (work[j.1].1*p.attentionStartStride)).read .nat)
      (Fin.cases (fun j=>(m p.request.cast (work[j.1].1*p.requestStride)).read .nat)
        (Fin.cases (fun j=>counts p work m j.1)
          (Fin.cases (fun j=>(m p.tokens.cast (gatherOffset p (Grid.state m work[j.1]) j.2.1.val)).read .nat)
            (Fin.cases (fun j=>(m p.prob (probOffset p (Grid.state m work[j.1]) j.2.1.val)).read .fp32)
              (Fin.cases (fun j=>(m p.value (valueOffset p (Grid.state m work[j.1]) j.2.2.1.val+
                (m p.tokens.cast (gatherOffset p (Grid.state m work[j.1]) j.2.1.val)).read .nat*p.valueBatchStride)).read .fp32)
                (fun i=>Fin.elim0 i))))))) i

theorem domain (p : Parameters) (work : List Grid.PID) (m : Grid.Memory α) :
    Ports.Domain p work (counts p work m) (values p work m) := by
  intro j
  rfl

theorem binding (M : Algebra α) (R : SumPrimitive α) (p : Parameters)
    (work : List Grid.PID) (s : State α) :
    (IO.program M R p work (counts p work s.mem)).InputBinding (values p work s.mem) s := by
  intro i j active
  change Fin 7 at i
  fin_cases i <;> rfl

private def inputOffsets (io : ProgramIO α α) (vals : io.Inputs) : List Nat :=
  (List.finRange io.inputs.length).flatMap fun i=>(TileShape.allIndices io.inputs[i].shape).map (io.read i vals)
private def outputOffsets (io : ProgramIO α α) (vals : io.Inputs) : List Nat :=
  (List.finRange io.outputs.length).flatMap fun o=>(TileShape.allIndices io.outputs[o].shape).map (io.write o vals)

/-- A concrete capacity witness for every old total raw heap. Actual bounded
Tensor/byte transports remain the existing checked TensorBridge applications. -/
def capacity (M : Algebra α) (R : SumPrimitive α) (p : Parameters)
    (work : List Grid.PID) (m : Grid.Memory α) (r : RegionName) : Nat :=
  let io := IO.program M R p work (counts p work m)
  let vals := values p work m
  if r∈io.buffers then
    (inputOffsets io vals).sum+(outputOffsets io vals).sum+1
  else 0

theorem inBounds (M : Algebra α) (R : SumPrimitive α) (p : Parameters)
    (work : List Grid.PID) (m : Grid.Memory α) :
    (IO.program M R p work (counts p work m)).InBounds (capacity M R p work m) (values p work m) := by
  refine ⟨?_,?_,?_,?_⟩
  · intro r outside
    simp [capacity,outside]
  · intro i j active
    have member : (IO.program M R p work (counts p work m)).inputs[i].buf∈
        (IO.program M R p work (counts p work m)).buffers := by
      change Fin 7 at i
      fin_cases i <;> simp [IO.program,Ports.memory,Ports.inputs]
    unfold capacity
    simp only [member,if_true]
    have hm : (IO.program M R p work (counts p work m)).read i (values p work m) j∈
        inputOffsets (IO.program M R p work (counts p work m)) (values p work m) := by
      exact List.mem_flatMap.mpr ⟨i,by simp,List.mem_map.mpr ⟨j,TileShape.mem_allIndices _ _,rfl⟩⟩
    have h := List.le_sum_of_mem hm
    omega
  · intro o j active
    have member : (IO.program M R p work (counts p work m)).outputs[o].buf∈
        (IO.program M R p work (counts p work m)).buffers := by
      change Fin 1 at o
      fin_cases o <;> simp [IO.program,Ports.memory,Ports.outputs]
    unfold capacity
    simp only [member,if_true]
    have hm : (IO.program M R p work (counts p work m)).write o (values p work m) j∈
        outputOffsets (IO.program M R p work (counts p work m)) (values p work m) := by
      exact List.mem_flatMap.mpr ⟨o,by simp,List.mem_map.mpr ⟨j,TileShape.mem_allIndices _ _,rfl⟩⟩
    have h := List.le_sum_of_mem hm
    omega
  · intro k
    exact Fin.elim0 k

#axiomsClean binding
#axiomsClean inBounds
end
end VeriTile.Bench.Repairs.TokenAttnMistralGridYChecked.PhysicalIO
