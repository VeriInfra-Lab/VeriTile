import bench.tritonbench_g.chunk_linear_attn.HostSum
import bench.tritonbench_g.chunk_linear_attn.HostMetadata

namespace VeriTile.Bench.Repairs.ChunkLinearAttn.KernelReference
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
noncomputable section
variable {α : Type} [Inhabited α]

/-- A coordinate table specifies values only on its logical output cells.
Neither execution, launch order nor an output equation is an input. -/
def table {ι : Type} (region : ι → RegionName) (index : ι → Nat)
    (value : ι → Cell α) (before : Memory α) : Memory α := by
  classical
  exact fun r o => if h : ∃ i,region i=r ∧ index i=o then value h.choose else before r o

inductive StateCell (p : Parameters) (final : Bool) where
  | history (head : Fin (p.B*p.H)) (time : Fin p.NT) (key : Fin p.K) (value : Fin p.V)
  | endpoint (enabled : final=Bool.true) (head : Fin (p.B*p.H)) (key : Fin p.K) (value : Fin p.V)

def stateRegion (rev : Bool) : StateCell p final → RegionName
  | .history .. => historyName rev
  | .endpoint .. => finalName rev
def stateIndex : StateCell p final → Nat
  | .history h t k v => ((h.val*p.NT+t.val)*p.K+k.val)*p.V+v.val
  | .endpoint _ h k v => (h.val*p.K+k.val)*p.V+v.val
def stateValue (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (shape : Geometry.ShapeValid p) (scale : ℝ) (rev seed : Bool) (input : Memory α) :
    StateCell p final → Cell α
  | .history h t k v => .mk .real (cast M (stateValues M D p
      (Ownership.statePID p h.val k.val v.val) input scale rev seed (chunk p rev t.val)
      (Ownership.stateLane p shape k.val v.val)))
  | .endpoint _ h k v => .mk .real (cast M (stateValues M D p
      (Ownership.statePID p h.val k.val v.val) input scale rev seed p.NT
      (Ownership.stateLane p shape k.val v.val)))
def state (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (shape : Geometry.ShapeValid p) (scale : ℝ) (rev seed final : Bool) (input : Memory α) : Memory α :=
  table (stateRegion rev) stateIndex (stateValue M D p shape scale rev seed input (final:=final)) input

abbrev OutputCell (p : Parameters) := Fin (p.B*p.H) × Fin p.T × Fin p.V
def outputIndex (i : OutputCell p) := (i.1.val*p.T+i.2.1.val)*p.V+i.2.2.val
def outputValue (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (shape : Geometry.ShapeValid p) (scale : ℝ) (input : Memory α) (i : OutputCell p) : Cell α :=
  let pid := Ownership.outputPID p i.1.val i.2.1.val i.2.2.val
  .mk .real (cast M (outputResult M D p pid input scale
    (outputStates M D p pid input (keyBlocks p)) (Ownership.outputLane p shape i.2.1.val i.2.2.val)))
def output (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (shape : Geometry.ShapeValid p) (scale : ℝ) (input : Memory α) : Memory α :=
  table (fun _ : OutputCell p => "output") outputIndex (outputValue M D p shape scale input) input

inductive GradientCell (p : Parameters) where
  | query (head : Fin (p.B*p.H)) (time : Fin p.T) (key : Fin p.K)
  | key (head : Fin (p.B*p.H)) (time : Fin p.T) (key : Fin p.K)
  | valuePartial (block : Fin (keyBlocks p)) (head : Fin (p.B*p.H)) (time : Fin p.T) (value : Fin p.V)
def gradientRegion : GradientCell p → RegionName
  | .query .. => "queryGradient"
  | .key .. => "keyGradient"
  | .valuePartial .. => "valuePartials"
def gradientIndex : GradientCell p → Nat
  | .query h t k | .key h t k => (h.val*p.T+t.val)*p.K+k.val
  | .valuePartial b h t v => ((b.val*(p.B*p.H)+h.val)*p.T+t.val)*p.V+v.val
def gradientValue (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (shape : Geometry.ShapeValid p) (scale : ℝ) (input : Memory α) : GradientCell p → Cell α
  | .query h t k =>
    let pid := Ownership.gradientPID p h.val t.val k.val
    .mk .real (cast M (gradQueryResult M D p pid input scale
      (gradStates M D p pid input scale (valueBlocks p)) (Ownership.gradientLane p shape t.val k.val)))
  | .key h t k =>
    let pid := Ownership.gradientPID p h.val t.val k.val
    .mk .real (cast M (gradKeyResult M D p pid input scale
      (gradStates M D p pid input scale (valueBlocks p)) (Ownership.gradientLane p shape t.val k.val)))
  | .valuePartial b h t v =>
    let pid := Ownership.partialPID p b.val h.val t.val
    .mk .real (cast M (gradValue M D p pid input scale (v.val/p.BV)
      (Ownership.outputLane p shape t.val v.val)))
def gradient (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (shape : Geometry.ShapeValid p) (scale : ℝ) (input : Memory α) : Memory α :=
  table gradientRegion gradientIndex (gradientValue M D p shape scale input) input

/-- The independent library boundary is a finite scalar addition tree with
certified leaf coverage. It has no precomputed output field. -/
def sum (M : Algebra α) (p : Parameters) (selection : LibrarySelection p) (input : Memory α) : Memory α :=
  table (fun _ : Fin (valueCount p) => "valueGradient") Fin.val
    (fun i => .mk .real (cast M (LibrarySum.evaluate M (selection.plans i)
      (fun k => (input "valuePartials" (k.val*valueCount p+i.val)).read .real)))) input

/-- Four independent coordinate tables, followed by the selected library
reduction. Local arithmetic recurrences are retained from the existing proofs;
there is no grid controller, scatter loop or source execution in this target. -/
def result (M : Algebra α) (D : DotPrimitive α) (dims : Host.Dimensions) (valid : Host.Valid dims)
    (scale : ℝ) (seed final terminal : Bool) (selection : LibrarySelection (dims.parameters Bool.true))
    (input : Memory α) : Memory α :=
  let f := dims.parameters Bool.false
  let b := dims.parameters Bool.true
  let forward := state M D f (Host.shape_valid dims valid Bool.false) scale Bool.false seed final input
  let values := output M D f (Host.shape_valid dims valid Bool.false) scale forward
  let backward := state M D b (Host.shape_valid dims valid Bool.true) scale Bool.true terminal seed values
  let gradients := gradient M D b (Host.shape_valid dims valid Bool.true) scale backward
  sum M b selection gradients

#specNonCircular result avoiding [FP.Structural.exec,runGrid,executeFour,stageMemory,gridMemory,
  modelFour,completeMemory,stateMemory,outputMemory,gradientMemory,hostSumMemory]
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.KernelReference
