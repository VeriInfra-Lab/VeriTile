import bench.repairs.tritonbench_g.llama_ff_triton_scalar_cast.Proofs.GateSource
import bench.repairs.tritonbench_g.llama_ff_triton_scalar_cast.Proofs.Grid
import VeriTile.Triton.Float.ArrayProgramIO
import bench.repairs.tritonbench_g.llama_ff_triton_scalar_cast.Proofs.Real
import bench.repairs.tritonbench_g.llama_ff_triton_scalar_cast.Proofs.RealGrid
import VeriTile.Triton.Float.LogicalProgramIO


namespace VeriTile.Bench.Repairs.LlamaFFScalarCast.Interface
open VeriTile.Triton FP.Structural Execution
noncomputable section
attribute [local instance] Classical.propDecidable

/-- Named arrays retain the old decoded-read domain, including simultaneous
half/int views and the global int8 condition. Only the selected branch's finite
read footprint requires allocation; unused suffixes are never source reads. -/
def inputPorts : List ArrayIOChannel :=
  [⟨.fp16,"A"⟩,⟨.fp16,"W1"⟩,⟨.fp16,"W3"⟩,⟨.fp16,"RMS"⟩,
   ⟨.int,"W1"⟩,⟨.int,"W3"⟩,⟨.int,"RMS"⟩]
def outputPorts (p : Parameters) : List IOChannel := [⟨.fp16,[p.rows,p.columns],"OUT"⟩]
abbrev Inputs (α : Type) := ArrayValues α inputPorts

def byteDomain {α : Type} (xs : Inputs α) : Prop :=
  (∀ o, Float8E5Bits.isInt8 (xs (4 : Fin 7) o) = Bool.true) ∧
  (∀ o, Float8E5Bits.isInt8 (xs (5 : Fin 7) o) = Bool.true) ∧
  (∀ o, Float8E5Bits.isInt8 (xs (6 : Fin 7) o) = Bool.true)

def reads (p : Parameters) (i : Fin 7) : List Nat :=
  if Grid.count p = 0 then [] else
  if i.val = 0 then List.range (p.rows*p.inner) else
  if (i.val < 4 ∧ p.byte=Bool.false) ∨ (4 ≤ i.val ∧ p.byte=Bool.true) then
    List.range (if i.val=3 ∨ i.val=6 then p.inner else p.columns*p.inner)
  else []

def job {α : Type} (M : Algebra α) (kernel : ComputeKernel) (pid : Nat)
    (first : Bool) : Invocation α where
  kernel := kernel
  backend := M
  pids := fun ax => if ax=0 then pid else 0
  setPids := fun _ => first

/-- The existing compiler-bound native execution, now an explicit schedule of
source fragments. The four original fragments keep `a.model`; only the scalar
gate stage selects its existing primitive ports. No execution callback. -/
def nativeJobs {α : Type} [Inhabited α] (a : Arithmetic α) (g : GateInstructions α)
    (p : Parameters) (pid : Nat) : List (Invocation α) :=
  [job a.model (.mk ["A","W1","W3","RMS"] ["OUT"] (prelude p)) pid Bool.true,
   job a.model (.mk ["A","W1","W3","RMS"] ["OUT"]
     [.forRangeInt "step" (.constInt 0) countOp (.constNat 1) (compiledBody p)]) pid Bool.false,
   job a.model (.mk [] [] ((epilogue p).take 2)) pid Bool.false,
   job (NativeStages.gateBackend a g) (.mk [] [] NativeStages.gateCode) pid Bool.false,
   job a.model (.mk [] ["OUT"] ((epilogue p).drop 5)) pid Bool.false]

def jobs {α : Type} [Inhabited α] (native : Bool) (a : Arithmetic α) (g : GateInstructions α)
    (p : Parameters) (pid : Nat) : List (Invocation α) :=
  if native then nativeJobs a g p pid else [job a.model (source p) pid Bool.true]

def programIO {α : Type} [Inhabited α] (native : Bool) (a : Arithmetic α)
    (g : GateInstructions α) (p : Parameters) : ArrayProgramIO α α where
  inputs := inputPorts
  outputs := outputPorts p
  scratch := []
  buffers := ["A","W1","W3","RMS","OUT"]
  reads := fun i _ => reads p i
  write := fun ⟨0,_⟩ _ j => p.columns*j.1.val+j.2.1.val
  outputMask := fun _ _ _ => True
  scratchWrite := fun i => Fin.elim0 i
  scratchMask := fun i => Fin.elim0 i
  schedule := fun _ => (List.range (Grid.count p)).flatMap (jobs native a g p)
  encode := id
  observe := fun _ _ => id

end
end VeriTile.Bench.Repairs.LlamaFFScalarCast.Interface


namespace VeriTile.Bench.Repairs.LlamaFFScalarCast.Reference
open VeriTile.Triton FP.Structural Execution Interface
noncomputable section

def decodedWeight (p : Parameters) (half : Nat → ℝ) (bytes : Nat → Int) (o : Nat) : ℝ :=
  if p.byte then decodeReal (Float8E5Bits.ofInt (bytes o)) else half o

end
end VeriTile.Bench.Repairs.LlamaFFScalarCast.Reference

namespace VeriTile.Bench.Repairs.LlamaFFScalarCast.ReaderIO
open VeriTile Triton FP.Structural Execution
noncomputable section

/-- All seven source arrays, including unused storage branches and suffixes.
Logical decoded accessors below do not discard any original input value. -/
structure Backing where
  activations : Nat → ℝ
  weight1Half : Nat → ℝ
  weight3Half : Nat → ℝ
  rmsHalf : Nat → ℝ
  weight1Byte : Nat → Int
  weight3Byte : Nat → Int
  rmsByte : Nat → Int

def backing : Interface.Inputs ℝ ≃ Backing where
  toFun := fun xs => ⟨xs (0 : Fin 7),xs (1 : Fin 7),xs (2 : Fin 7),xs (3 : Fin 7),
    xs (4 : Fin 7),xs (5 : Fin 7),xs (6 : Fin 7)⟩
  invFun := fun xs => Fin.cases xs.activations (Fin.cases xs.weight1Half
    (Fin.cases xs.weight3Half (Fin.cases xs.rmsHalf (Fin.cases xs.weight1Byte
      (Fin.cases xs.weight3Byte (Fin.cases xs.rmsByte (fun i => i.elim0)))))))
  left_inv := by intro xs; funext i j; fin_cases i <;> rfl
  right_inv := by intro xs; cases xs; rfl

/-- Logical coordinates are actual fields, rather than aliases hiding an
address computation. The graph connection uniquely determines their extension. -/
structure Arrays where
  x : Nat → Nat → ℝ
  rms : Nat → ℝ
  w1 : Nat → Nat → ℝ
  w3 : Nat → Nat → ℝ

def decode (p : Parameters) (raw : Backing) : Arrays where
  x := fun row k => raw.activations (row*p.inner+k)
  rms := Reference.decodedWeight p raw.rmsHalf raw.rmsByte
  w1 := fun column k => Reference.decodedWeight p raw.weight1Half raw.weight1Byte (k+column*p.inner)
  w3 := fun column k => Reference.decodedWeight p raw.weight3Half raw.weight3Byte (k+column*p.inner)

/-- The complete backing remains, including the unused branch and all suffixes;
logical arrays are primary fields with an exact, uniquely determined connection. -/
structure Inputs (p : Parameters) where
  backing : Backing
  arrays : Arrays
  connected : arrays = decode p backing

def inputs (p : Parameters) : Interface.Inputs ℝ ≃ Inputs p where
  toFun := fun raw => ⟨backing raw,decode p (backing raw),rfl⟩
  invFun := fun xs => backing.symm xs.backing
  left_inv := fun raw => backing.symm_apply_apply raw
  right_inv := by
    rintro ⟨raw,arrays,connected⟩
    cases connected
    simp only [Equiv.apply_symm_apply]

def byteDomain {p : Parameters} (xs : Inputs p) : Prop :=
  (∀ o, Float8E5Bits.isInt8 (xs.backing.weight1Byte o) = Bool.true) ∧
  (∀ o, Float8E5Bits.isInt8 (xs.backing.weight3Byte o) = Bool.true) ∧
  (∀ o, Float8E5Bits.isInt8 (xs.backing.rmsByte o) = Bool.true)

/-- Original complete ArrayProgramIO, including both storage branches, all
read windows and the byte domain; only input names/output coordinates change. -/
def view (p : Parameters) :
    LogicalDomainIO (Interface.Inputs ℝ) (ChannelValues ℝ (Interface.outputPorts p))
      (Inputs p) (Fin p.rows → Fin p.columns → ℝ)
      ((Interface.programIO Bool.false realArithmetic RealGrid.unusedGate p).ImplementsOn) where
  inputs := inputs p
  outputs := (ChannelValues.singleEquiv ℝ _).trans (Values.matrixEquiv ℝ .fp16 p.rows p.columns)
end
end VeriTile.Bench.Repairs.LlamaFFScalarCast.ReaderIO

namespace VeriTile.Bench.Repairs.LlamaFFScalarCast.Reference
open VeriTile.Triton FP.Structural Execution Interface
noncomputable section

/-- Ordinary finite RMSNorm–SwiGLU input arrays. Exceptional E5M2 values use
the same explicit ideal interpretation as the existing Real theorem. -/
def realInput (p : Parameters) (xs : Inputs ℝ) (row : Nat) : LlamaFF.RowInput ℝ where
  x := fun k => xs (0 : Fin 7) (row*p.inner+k)
  rms := decodedWeight p (xs (3 : Fin 7)) (xs (6 : Fin 7))
  weight := fun n k => decodedWeight p (xs (1 : Fin 7)) (xs (4 : Fin 7)) (k+n*p.inner)
  weight3 := fun n k => decodedWeight p (xs (2 : Fin 7)) (xs (5 : Fin 7)) (k+n*p.inner)

def realFormula (p : Parameters) (xs : Inputs ℝ) : ChannelValues ℝ (outputPorts p) :=
  fun ⟨0,_⟩ j => LlamaFF.realTarget ⟨p.inner,p.epsilon⟩ (realInput p xs j.1.val) j.2.1.val

#specNonCircular realFormula avoiding [source,run,Grid.grid,Grid.one]
end
end VeriTile.Bench.Repairs.LlamaFFScalarCast.Reference
