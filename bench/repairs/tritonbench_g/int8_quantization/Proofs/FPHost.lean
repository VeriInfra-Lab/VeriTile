import bench.repairs.tritonbench_g.int8_quantization.Proofs.PairResults
import bench.repairs.tritonbench_g.int8_quantization.Proofs.SourceFP
namespace VeriTile.Bench.Repairs.Int8Quantization.Direct
open VeriTile.Triton FP.Structural Revision002 Words
noncomputable section
set_option maxHeartbeats 500000

def logicalInputs (c : Request) (M : Algebra Word) (initial : Memory) (w : Schedule.Program) : Fin (p c).width → Word :=
  fun j=>M.fp32Load (if Grid.active (p c) w (j,PUnit.unit) then
    inputValue c initial (Grid.offset (p c) w (j,PUnit.unit)) else scalar32 M 0.0)

theorem inputs_original (c : Request) (M : Algebra Word) (before : Heap) (out : Allocation c before)
    (w : Schedule.Program) (member : w∈Schedule.programs 0 (count c)) :
    FPTrace.inputs M (Grid.parameters (p c) w) (Grid.atProgram (Bytes.bind (layout c out) before.memory) w)=
      logicalInputs c M before.memory w := by
  funext j
  unfold FPTrace.inputs loaded logicalInputs
  simp only [Grid.active_at,Grid.offset_at]
  apply congrArg M.fp32Load
  change (if Grid.active (p c) w (j,PUnit.unit) then
      Bytes.read32 before.memory ((view c).base+4*Grid.offset (p c) w (j,PUnit.unit)) else scalar32 M 0.0) =
    (if Grid.active (p c) w (j,PUnit.unit) then inputValue c before.memory (Grid.offset (p c) w (j,PUnit.unit)) else scalar32 M 0.0)
  by_cases active : Grid.active (p c) w (j,PUnit.unit)
  · simp only [if_pos active]
    exact input_pointer c before.memory _ (offset_bound c w member _ active)
  · simp only [if_neg active]

theorem logicalInputs_preserved (c : Request) (M : Algebra Word) (before later : Heap)
    (ext : Extension before later) (live : InputLive c before)
    (w : Schedule.Program) (member : w∈Schedule.programs 0 (count c)) :
    logicalInputs c M later.memory w=logicalInputs c M before.memory w := by
  funext j
  unfold logicalInputs
  by_cases active : Grid.active (p c) w (j,PUnit.unit)
  · simp only [if_pos active]
    rw [input_preserved c before later ext live _ (offset_bound c w member _ active)]
  · simp only [if_neg active]

def reference (c : Request) (M : Algebra Word) (software : Floating.Numerics Word)
    (original : Memory) (w : Schedule.Program) : (Fin (p c).width → Int) × Word :=
  if channels c.x.metadata=0 then (fun _=>0,M.literal (some .fp32) .real 0) else
    Floating.run (FPTrace.constants M (p c)) software c.query (logicalInputs c M original w)

structure NumericalValues (c : Request) (M : Algebra Word) (software : Floating.Numerics Word)
    (original : Memory) (output scales : Tensor) (result : Memory) : Prop where
  output : ∀ w, w∈Schedule.programs 0 (count c) → ∀ i : Lane (p c), Grid.active (p c) w i →
    result (output.base+Grid.offset (p c) w i)=BitVec.ofInt 8 ((reference c M software original w).1 i.1)
  scales : ∀ w, w∈Schedule.programs 0 (count c) →
    Bytes.read32 result (scales.base+4*w.index)=(reference c M software original w).2

/-- The only numerical premises are reached primitive operations. No premise
states an output, final heap, kernel equality, or exact-real computation. -/
theorem numerical_values (c : Request) (M : Algebra Word) (ops : Model.Primitives Word)
    (original before : Heap) (out : Allocation c before) (output scales : Tensor) (result : Memory)
    (observed : Values c M ops before.memory (layout c out) output scales result)
    (ext : Extension original before) (live : InputLive c original)
    (ctx : ∀ w, w∈Schedule.programs 0 (count c) → 0 < channels c.x.metadata →
      FPTrace.Context M (Grid.parameters (p c) w) (Grid.atProgram (Bytes.bind (layout c out) before.memory) w))
    (software : Floating.Numerics Word)
    (laws : ∀ w, w∈Schedule.programs 0 (count c) → 0 < channels c.x.metadata →
      Floating.Local (FPTrace.constants M (p c)) (FPTrace.numerics M ops) software c.query
        (logicalInputs c M original.memory w)) :
    NumericalValues c M software original.memory output scales result := by
  have trace (w : Schedule.Program) (member : w∈Schedule.programs 0 (count c))
      (positive : 0 < channels c.x.metadata) :
      ((fun j=>quantized M ops (Grid.parameters (p c) w) (width_positive c.x.metadata c.block)
        (Grid.atProgram (Bytes.bind (layout c out) before.memory) w) (j,PUnit.unit)),
       savedScale M ops (Grid.parameters (p c) w) (width_positive c.x.metadata c.block)
        (Grid.atProgram (Bytes.bind (layout c out) before.memory) w)) = reference c M software original.memory w := by
    have tr := FPTrace.arithmetic_trace M ops (Grid.parameters (p c) w)
      (width_positive c.x.metadata c.block) (Grid.atProgram (Bytes.bind (layout c out) before.memory) w) (ctx w member positive)
    rw [inputs_original c M before out w member,logicalInputs_preserved c M original before ext live w member] at tr
    exact tr.trans (by
      change Floating.run (FPTrace.constants M (p c)) (FPTrace.numerics M ops) c.query _ = _
      simpa only [reference,if_neg (Nat.ne_of_gt positive)] using
        Floating.local_equiv _ _ _ _ _ (laws w member positive))
  constructor
  · intro w hw i hi
    rw [observed.bytes w hw i hi]
    have t := congrFun (congrArg Prod.fst (trace w hw (active_channels c w i hi))) i.1
    cases i with | mk j u=>cases u;exact congrArg (BitVec.ofInt 8) t
  · intro w hw
    rw [observed.scales w hw]
    by_cases zero : channels c.x.metadata=0
    · simp [zero,reference]
    · simp only [if_neg zero]
      exact congrArg Prod.snd (trace w hw (by omega))

#axiomsClean numerical_values
#specNonCircular reference avoiding [after,typedRun,Grid.model,finalState,quantized,savedScale]
end
end VeriTile.Bench.Repairs.Int8Quantization.Direct
