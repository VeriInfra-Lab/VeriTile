import VeriTile.Triton.Float.LogicalProgramIO
import bench.repairs.tritonbench_g.fp4_to_bf16.Proofs.Launch
import bench.tritonbench_g.fp4_to_bf16.Proofs.FPReal
import VeriTile.Triton.Float.ProgramIO

namespace VeriTile.Bench.Repairs.Fp4ToBf16.ConversionIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Fp4ToBf16 FPValue Host RealSpec
noncomputable section

/-- Packed uint8 input is represented by a discrete Nat channel and the
explicit byte domain below. Output cells are BF16, as fixed by the source. -/
def inputChannels (h : Input) : List IOChannel := [⟨.nat,[h.shape.count],h.input⟩]
def outputChannels (h : Input) : List IOChannel := [⟨.bf16,[h.shape.count*2],h.output⟩]
abbrev Inputs (A : Type) (h : Input) := ChannelValues A (inputChannels h)
abbrev Outputs (A : Type) (h : Input) := ChannelValues A (outputChannels h)

def byte (values : Inputs A h) (i : Fin h.shape.count) : Nat :=
  values ⟨0,by simp [inputChannels]⟩ (i,PUnit.unit)
def ByteDomain (h : Input) (values : Inputs A h) : Prop := ∀ i,byte values i<256

/-- High nibble first, exactly two logical outputs per packed input byte. -/
def nibble (values : Inputs A h) (j : Fin (h.shape.count*2)) : Bits.Nibble :=
  let packed := BitVec.ofNat 8 (byte values ⟨j.val/2,by have := j.isLt;omega⟩)
  if j.val%2=0 then Bits.high packed else Bits.low packed

/-- Independent BF16 table bits, with both zero signs retained. -/
def nativeTarget (h : Input) (values : Inputs Word h) : Outputs Word h :=
  Fin.cases (fun j => embed ⟨Bits.expectedBits (nibble values j.1)⟩) (fun o => Fin.elim0 o)

/-- Independent sign/exponent/mantissa E2M1 formula. -/
def mathematical (h : Input) (values : Inputs (Option ℝ) h) : Outputs (Option ℝ) h :=
  Fin.cases (fun j => some (e2m1 (nibble values j.1))) (fun o => Fin.elim0 o)

def memory (h : Input) : MemoryIO A where
  inputs := inputChannels h
  outputs := outputChannels h
  buffers := [h.input,h.output]
  read := Fin.cases (fun _ j => j.1.val) (fun i => Fin.elim0 i)
  inputMask := fun _ _ _ => True
  write := Fin.cases (fun _ j => j.1.val) (fun o => Fin.elim0 o)
  outputMask := fun _ _ _ => True
  scratch := []
  scratchWrite := fun k => Fin.elim0 k
  scratchMask := fun k => Fin.elim0 k

/-- Actual finite-ABI source invocation. Count and signed program ID widen
before arithmetic; fresh local registers and grid metadata match Launch. -/
def invocation (rest : Algebra Word) (h : Input) (pid : Nat) : Invocation Word where
  kernel := Kernels.source (Launch.parameters h)
  backend := backend rest
  clearRegisters := Bool.true
  pids := fun ax => if ax=0 then ((BitVec.ofNat 32 pid).signExtend 64).toNat else 0
  numPids := some (fun ax => if ax=0 then grid h else 1)

def conversionIO (rest : Algebra Word) (h : Input) (jobs : List Nat)
    (encode : A → Word) (observe : Word → A) : ProgramIO A Word where
  toMemoryIO := memory h
  schedule := fun _ => jobs.map (invocation rest h)
  encode := encode
  observe := fun _ _ => observe

def nativeIO (rest : Algebra Word) (h : Input) (jobs : List Nat) : ProgramIO Word Word :=
  conversionIO rest h jobs id id

/-- Native BF16 observation preserves finiteness as well as the Real value.
Every active output is proved to be `some` of the independent E2M1 formula. -/
def realObservation (word : Word) : Option ℝ := finiteReal ⟨word.bits.truncate 16⟩

/-- There are no floating input channels: mapValue preserves the packed Nat. -/
def realIO (rest : Algebra Word) (h : Input) (jobs : List Nat) : ProgramIO (Option ℝ) Word :=
  conversionIO rest h jobs (fun _ => FP.Value.zero .fp32) realObservation

end
end VeriTile.Bench.Repairs.Fp4ToBf16.ConversionIO

namespace VeriTile.Bench.Repairs.Fp4ToBf16.ConversionIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Fp4ToBf16 FPValue Host
noncomputable section

def byteCoordinates (h : Input) : Inputs (Option ℝ) h ≃ (Fin h.shape.count → Nat) where
  toFun := byte
  invFun := fun xs => Fin.cases (fun j => xs j.1) (fun i => Fin.elim0 i)
  left_inv := by
    intro xs
    funext i j
    fin_cases i
    cases j with | mk a b => cases b; rfl
  right_inv := fun _ => rfl

def realCoordinates (h : Input) : Outputs (Option ℝ) h ≃
    (Fin (h.shape.count * 2) → Option ℝ) where
  toFun := fun xs i => xs ⟨0, by simp [outputChannels]⟩ (i, ())
  invFun := fun xs => Fin.cases (fun j => xs j.1) (fun i => Fin.elim0 i)
  left_inv := by
    intro xs
    funext i j
    fin_cases i
    cases j with | mk a b => cases b; rfl
  right_inv := fun _ => rfl

def logicalRealIO (rest : Algebra Word) (h : Input) (jobs : List Nat) :
    LogicalProgramIO (Option ℝ) Word (Fin h.shape.count → Nat)
      (Fin (h.shape.count * 2) → Option ℝ) where
  core := realIO rest h jobs
  inputs := byteCoordinates h
  outputs := realCoordinates h

def LogicalByteDomain (h : Input) (xs : Fin h.shape.count → Nat) : Prop := ∀ i, xs i < 256

/-- High-first packed byte projection, with the finite index bound kept in IO. -/
def logicalNibble (h : Input) (xs : Fin h.shape.count → Nat)
    (j : Fin (h.shape.count * 2)) : Bits.Nibble :=
  let packed := BitVec.ofNat 8 (xs ⟨j.val / 2, by have := j.isLt; omega⟩)
  if j.val % 2 = 0 then Bits.high packed else Bits.low packed
end
end VeriTile.Bench.Repairs.Fp4ToBf16.ConversionIO
