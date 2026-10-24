import bench.repairs.tritonbench_g.matmul_triton2.KernelOnly.Model
namespace VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.Targets
open VeriTile.Triton FP.Structural Revision001 Model
noncomputable section
set_option maxHeartbeats 8000000
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Arbitrary Real input values at the actual input pointers; the reference
has exactly K summands and no launch, padding or execution dependency. -/
def real (read : Nat → ℝ) (c : Arguments) (r col : Nat) : ℝ :=
  RealSum.sum (fun k => read (address c.a (r*columns c.a.metadata+k))*
    read (address c.b (k*columns c.b.metadata+col))) (columns c.a.metadata)

theorem real_value (read : Nat → ℝ) (c : Arguments) (choice : Fin 8)
    (other : State ℝ) (r col : Nat) :
    Core.logical Core.realPrimitives (parameters c choice) (entry read c other) r col=real read c r col := by
  rw [Core.logical_realMatmul _ _ _ _ (Native.config_legal choice).2.2.1]
  unfold Core.realMatmul real
  apply RealSum.sum_congr
  intro k _
  simp [Core.realProduct,parameters,entry,Cell.read_same]

/-- Two fixed-width hardware instruction ports for the source-selected SM90
profile. They consume eight TF32-mode FP32 words per operand and a live FP32
seed. Zero is fixed, and no whole-dot, tile or final-output oracle is exposed. -/
abbrev Ops := Native.Reference
abbrev instructions := Native.referenceInstructions

def a (read : Nat → Word) (c : Arguments) (BK r block : Nat) (k : Fin BK) : Word :=
  if k.val < columns c.a.metadata-block*BK then
    read (address c.a (r*columns c.a.metadata+(block*BK+k.val))) else FP.Value.zero .fp32
def b (read : Nat → Word) (c : Arguments) (BK col block : Nat) (k : Fin BK) : Word :=
  if k.val < columns c.a.metadata-block*BK then
    read (address c.b ((block*BK+k.val)*columns c.b.metadata+col)) else FP.Value.zero .fp32
def accumulate (ops : Ops) (read : Nat → Word) (c : Arguments) (choice : Fin 8)
    (r col : Nat) : Nat → Word
  | 0 => FP.Value.zero .fp32
  | block+1 => Native.dot (instructions ops) (Native.family choice)
      (a read c (Native.config choice).blockInner r block)
      (b read c (Native.config choice).blockInner col block)
      (accumulate ops read c choice r col block)

def native (ops : Ops) (read : Nat → Word) (c : Arguments) (choice : Fin 8)
    (r col : Nat) : Word :=
  accumulate ops read c choice r col
    ((columns c.a.metadata+(Native.config choice).blockInner-1)/(Native.config choice).blockInner)

theorem native_value (ops : Ops) (read : Nat → Word) (c : Arguments) (choice : Fin 8)
    (other : State Word) (r col : Nat) :
    Core.logical (Native.primitives (instructions ops) (Native.family choice))
      (parameters c choice) (entry read c other) r col=native ops read c choice r col := by
  have recurrence : ∀ n,Core.logicalAccum (Native.primitives (instructions ops) (Native.family choice))
      (parameters c choice) (entry read c other) r col n=accumulate ops read c choice r col n := by
    intro n
    induction n with
    | zero => rfl
    | succ n ih =>
      simp only [Core.logicalAccum,accumulate,ih,Native.primitives]
      congr 1
      · funext k
        simp [Core.logicalA,parameters,Core.inner,entry,Cell.read_same,a,instructions,Native.referenceInstructions]
      · funext k
        simp [Core.logicalB,parameters,Core.inner,entry,Cell.read_same,b,instructions,Native.referenceInstructions]
  exact recurrence _

#axiomsClean real_value
#axiomsClean native_value
#specNonCircular real avoiding [Model.run,Core.grid,Core.logical,Core.source,FP.Structural.exec]
#specNonCircular native avoiding [Model.run,Core.grid,Core.logical,Core.source,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.Targets
