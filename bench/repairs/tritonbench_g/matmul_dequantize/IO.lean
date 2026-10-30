import bench.repairs.tritonbench_g.matmul_dequantize.Whole001.Proofs.RealDequant
import bench.repairs.tritonbench_g.matmul_dequantize.Whole001.Proofs.NativeStaging
import VeriTile.Triton.Float.ObservedProgramIO
import VeriTile.Triton.Float.LogicalProgramIO


/-! Finite typed source IO. Native symbolic inputs retain their original full
expression domain; evaluation is a separate output observation. -/

namespace VeriTile.Bench.Repairs.MatmulDequantize.D32IO
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulDequantizeWhole001
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
open VeriTile.Bench.Repairs.MatmulDequantizeWhole001.Dequant32 (Parameters)
noncomputable section

abbrev Lane (p : Parameters) := TileIndex [Geometry.count (DequantGrid32.geometry p),p.blockRows,p.blockColumns]
def inputChannels (p : Parameters) : List IOChannel :=
  [⟨.int,[Geometry.count (DequantGrid32.geometry p),p.blockRows,p.blockColumns],p.weight⟩,
   ⟨.int,[Geometry.count (DequantGrid32.geometry p),p.blockRows,p.blockColumns],p.zero⟩,
   ⟨.fp32,[Geometry.count (DequantGrid32.geometry p),p.blockRows,p.blockColumns],p.scale⟩]
def outputChannels (p : Parameters) : List IOChannel :=
  [⟨.fp32,[Geometry.count (DequantGrid32.geometry p),p.blockRows,p.blockColumns],p.output⟩]
abbrev Inputs (A : Type) (p : Parameters) := ChannelValues A (inputChannels p)
abbrev Outputs (A : Type) (p : Parameters) := ChannelValues A (outputChannels p)

def row (p : Parameters) (j : Lane p) := Geometry.outRow (DequantGrid32.geometry p) j.1.val j.2
def column (p : Parameters) (j : Lane p) := Geometry.outColumn (DequantGrid32.geometry p) j.1.val j.2
def active (p : Parameters) (j : Lane p) := Geometry.Active (DequantGrid32.geometry p) j.1.val j.2

def weightAddress (p : Parameters) (j : Lane p) := row p j/8*p.weightRowStride+column p j*p.weightColumnStride
def zeroAddress (p : Parameters) (j : Lane p) := row p j/p.groupSize*p.zeroRowStride+column p j/8*p.zeroColumnStride
def scaleAddress (p : Parameters) (j : Lane p) := row p j/p.groupSize*p.scaleRowStride+column p j*p.scaleColumnStride

def inputAddress (p : Parameters) :
    (i : Fin (inputChannels p).length) → TileIndex (inputChannels p)[i].shape → Nat :=
  Fin.cases (weightAddress p) (Fin.cases (zeroAddress p) (Fin.cases (scaleAddress p) (fun i => Fin.elim0 i)))

def inputActive (p : Parameters) :
    (i : Fin (inputChannels p).length) → TileIndex (inputChannels p)[i].shape → Prop :=
  Fin.cases (active p) (Fin.cases (active p) (Fin.cases (active p) (fun i => Fin.elim0 i)))

/-- Each finite lane exposes its packed weight/zero and scale input at the
actual strided address. Repeated packed addresses are shared by the binding.
Only original active lanes are backed or writable; global scratch is empty. -/
def memory (p : Parameters) : MemoryIO A where
  inputs := inputChannels p
  outputs := outputChannels p
  buffers := [p.weight,p.zero,p.scale,p.output]
  read := fun i _ => inputAddress p i
  inputMask := fun i _ => inputActive p i
  write := Fin.cases (fun _ j => Geometry.offset (DequantGrid32.geometry p) j.1.val j.2) (fun i => Fin.elim0 i)
  outputMask := Fin.cases (fun _ => active p) (fun i => Fin.elim0 i)
  scratch := []
  scratchWrite := fun i => Fin.elim0 i
  scratchMask := fun i => Fin.elim0 i

def invocation (M : Algebra A) (p : Parameters) (pid : Nat) : Invocation A where
  kernel := VeriTile.Bench.Repairs.MatmulDequantizeWhole001.Dequant32.source p
  backend := M
  pids := fun ax => if ax=0 then Geometry.blockRow (DequantGrid32.geometry p) pid
    else if ax=1 then Geometry.blockColumn (DequantGrid32.geometry p) pid else 0

def program (M : Algebra A) (p : Parameters) (jobs : List Nat) : ProgramIO A A where
  toMemoryIO := memory p
  schedule := fun _ => jobs.map (invocation M p)
  encode := id
  observe := fun _ _ => id

def difference (p : Parameters) (vals : Inputs A p) (j : Lane p) : Int :=
  Real.signedNibble (vals ⟨0,by simp [inputChannels]⟩ j) (row p j)-
    Real.signedNibble (vals ⟨1,by simp [inputChannels]⟩ j) (column p j)

/-- Independent signed-int32 nibble difference times the logical scale. -/
def mathematical (p : Parameters) (vals : Inputs ℝ p) : Outputs ℝ p :=
  Fin.cases (fun j => (difference p vals j : ℝ)*vals ⟨2,by simp [inputChannels]⟩ j) (fun i => Fin.elim0 i)

/-- The original conversion and multiplication graph over logical operands.
No source execution, output state or FP-to-Real identity occurs. -/
def expression (M : Algebra A) (p : Parameters) (vals : Inputs A p) : Outputs A p :=
  Fin.cases (fun j => M.binary none .fp32 .mul
    (M.cast (some .fp32) .real .fp32 (M.fromInt (some .fp32) (difference p vals j)))
    (vals ⟨2,by simp [inputChannels]⟩ j)) (fun i => Fin.elim0 i)

def realIO (p : Parameters) (jobs : List Nat) := program Real.scalar p jobs
def nativeIO (p : Parameters) (jobs : List Nat) := program Native.scalar p jobs

def observe (p : Parameters) (plan : Native.Plan) (isa : Native.WordISA) :
    (o : Fin (outputChannels p).length) → TileIndex (outputChannels p)[o].shape → Native.Expr → BitVec 32 :=
  Fin.cases (fun j value => (Native.lower plan value j.1.val).eval isa) (fun i => Fin.elim0 i)

def nativeTarget (p : Parameters) (plan : Native.Plan) (isa : Native.WordISA)
    (vals : Inputs Native.Expr p) : ChannelValues (BitVec 32) (outputChannels p) :=
  Fin.cases (fun j => (Native.lower plan
    (expression Native.scalar p vals ⟨0,by simp [outputChannels]⟩ j) j.1.val).eval isa) (fun i => Fin.elim0 i)

end
end VeriTile.Bench.Repairs.MatmulDequantize.D32IO

namespace VeriTile.Bench.Repairs.MatmulDequantize.D16IO
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulDequantizeWhole001
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
open VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped (Parameters)
noncomputable section

abbrev Lane (p : Parameters) := TileIndex [Geometry.count (DequantGrid16.geometry p),p.blockRows,p.blockColumns]
def inputChannels (p : Parameters) : List IOChannel :=
  [⟨.int,[Geometry.count (DequantGrid16.geometry p),p.blockRows,p.blockColumns],p.weight⟩,
   ⟨.int,[Geometry.count (DequantGrid16.geometry p),p.blockRows,p.blockColumns],p.zero⟩,
   ⟨.fp16,[Geometry.count (DequantGrid16.geometry p),p.blockRows,p.blockColumns],p.scale⟩]
def outputChannels (p : Parameters) : List IOChannel :=
  [⟨.fp16,[Geometry.count (DequantGrid16.geometry p),p.blockRows,p.blockColumns],p.output⟩]
abbrev Inputs (A : Type) (p : Parameters) := ChannelValues A (inputChannels p)
abbrev Outputs (A : Type) (p : Parameters) := ChannelValues A (outputChannels p)

def row (p : Parameters) (j : Lane p) := Geometry.outRow (DequantGrid16.geometry p) j.1.val j.2
def column (p : Parameters) (j : Lane p) := Geometry.outColumn (DequantGrid16.geometry p) j.1.val j.2
def active (p : Parameters) (j : Lane p) := Geometry.Active (DequantGrid16.geometry p) j.1.val j.2

def weightAddress (p : Parameters) (j : Lane p) := row p j/8*p.weightRowStride+column p j*p.weightColumnStride
def zeroAddress (p : Parameters) (j : Lane p) := row p j/p.groupSize*p.zeroRowStride+column p j/8*p.zeroColumnStride
def scaleAddress (p : Parameters) (j : Lane p) := row p j/p.groupSize*p.scaleRowStride+column p j*p.scaleColumnStride

def inputAddress (p : Parameters) :
    (i : Fin (inputChannels p).length) → TileIndex (inputChannels p)[i].shape → Nat :=
  Fin.cases (weightAddress p) (Fin.cases (zeroAddress p) (Fin.cases (scaleAddress p) (fun i => Fin.elim0 i)))

def inputActive (p : Parameters) :
    (i : Fin (inputChannels p).length) → TileIndex (inputChannels p)[i].shape → Prop :=
  Fin.cases (active p) (Fin.cases (active p) (Fin.cases (active p) (fun i => Fin.elim0 i)))

/-- Each finite lane exposes its packed weight/zero and scale input at the
actual strided address. Repeated packed addresses are shared by the binding.
Only original active lanes are backed or writable; global scratch is empty. -/
def memory (p : Parameters) : MemoryIO A where
  inputs := inputChannels p
  outputs := outputChannels p
  buffers := [p.weight,p.zero,p.scale,p.output]
  read := fun i _ => inputAddress p i
  inputMask := fun i _ => inputActive p i
  write := Fin.cases (fun _ j => Geometry.offset (DequantGrid16.geometry p) j.1.val j.2) (fun i => Fin.elim0 i)
  outputMask := Fin.cases (fun _ => active p) (fun i => Fin.elim0 i)
  scratch := []
  scratchWrite := fun i => Fin.elim0 i
  scratchMask := fun i => Fin.elim0 i

def invocation (M : Algebra A) (p : Parameters) (pid : Nat) : Invocation A where
  kernel := VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.source p
  backend := M
  pids := fun ax => if ax=0 then Geometry.blockRow (DequantGrid16.geometry p) pid
    else if ax=1 then Geometry.blockColumn (DequantGrid16.geometry p) pid else 0

def program (M : Algebra A) (p : Parameters) (jobs : List Nat) : ProgramIO A A where
  toMemoryIO := memory p
  schedule := fun _ => jobs.map (invocation M p)
  encode := id
  observe := fun _ _ => id

def difference (p : Parameters) (vals : Inputs A p) (j : Lane p) : Int :=
  Real.signedNibble (vals ⟨0,by simp [inputChannels]⟩ j) (row p j)-
    Real.signedNibble (vals ⟨1,by simp [inputChannels]⟩ j) (column p j)

/-- Independent signed-int32 nibble difference times the logical scale. -/
def mathematical (p : Parameters) (vals : Inputs ℝ p) : Outputs ℝ p :=
  Fin.cases (fun j => (difference p vals j : ℝ)*vals ⟨2,by simp [inputChannels]⟩ j) (fun i => Fin.elim0 i)

/-- The original conversion and multiplication graph over logical operands.
No source execution, output state or FP-to-Real identity occurs. -/
def expression (M : Algebra A) (p : Parameters) (vals : Inputs A p) : Outputs A p :=
  Fin.cases (fun j => M.binary none .fp16 .mul
    (M.cast (some .fp32) .real .fp16 (M.fromInt (some .fp32) (difference p vals j)))
    (vals ⟨2,by simp [inputChannels]⟩ j)) (fun i => Fin.elim0 i)

def realIO (p : Parameters) (jobs : List Nat) := program Real.scalar p jobs
def nativeIO (p : Parameters) (jobs : List Nat) := program Native.scalar p jobs

def observe (p : Parameters) (plan : Native.Plan) (isa : Native.WordISA) :
    (o : Fin (outputChannels p).length) → TileIndex (outputChannels p)[o].shape → Native.Expr → BitVec 32 :=
  Fin.cases (fun j value => (Native.lower plan value j.1.val).eval isa) (fun i => Fin.elim0 i)

def nativeTarget (p : Parameters) (plan : Native.Plan) (isa : Native.WordISA)
    (vals : Inputs Native.Expr p) : ChannelValues (BitVec 32) (outputChannels p) :=
  Fin.cases (fun j => (Native.lower plan
    (expression Native.scalar p vals ⟨0,by simp [outputChannels]⟩ j) j.1.val).eval isa) (fun i => Fin.elim0 i)

end
end VeriTile.Bench.Repairs.MatmulDequantize.D16IO


namespace VeriTile.Bench.Repairs.MatmulDequantize.FirstIO
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulDequantizeWhole001
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
open VeriTile.Bench.TritonBenchG.MatmulDequantize.FPTypedFirst (Parameters steps groupIndex Dot)
noncomputable section

abbrev AShape (p : Parameters) := [Geometry.count (FirstGrid.geometry p),steps p,p.blockRows,p.blockInner]
abbrev BShape (p : Parameters) := [Geometry.count (FirstGrid.geometry p),steps p,p.blockInner,p.blockColumns]
abbrev SShape (p : Parameters) := [Geometry.count (FirstGrid.geometry p),max (steps p) 1,p.blockColumns]
abbrev CShape (p : Parameters) := [Geometry.count (FirstGrid.geometry p),p.blockRows,p.blockColumns]
abbrev OutputLane (p : Parameters) := TileIndex (CShape p)
def row (p : Parameters) (pid r : Nat) := Geometry.blockRow (FirstGrid.geometry p) pid*p.blockRows+r
def column (p : Parameters) (pid c : Nat) := Geometry.blockColumn (FirstGrid.geometry p) pid*p.blockColumns+c

def aAddress (p : Parameters) (pid k r e : Nat) :=
  row p pid r*p.inputRowStride+e*p.inputInnerStride+k*(p.blockInner*p.inputInnerStride)
def bAddress (p : Parameters) (pid k e c : Nat) :=
  e/8*p.weightInnerStride+column p pid c*p.weightColumnStride+k*(p.blockInner/8*p.weightInnerStride)
def sAddress (p : Parameters) (pid k c : Nat) :=
  column p pid c*p.scaleColumnStride+groupIndex p k*p.scaleGroupStride
def zAddress (p : Parameters) (pid k c : Nat) :=
  column p pid c/8*p.zeroColumnStride+groupIndex p k*p.zeroGroupStride

def aWindow (p : Parameters) (j : TileIndex (AShape p)) := aAddress p j.1.val j.2.1.val j.2.2.1.val j.2.2.2.1.val
def bWindow (p : Parameters) (j : TileIndex (BShape p)) := bAddress p j.1.val j.2.1.val j.2.2.1.val j.2.2.2.1.val
def sWindow (p : Parameters) (j : TileIndex (SShape p)) := sAddress p j.1.val j.2.1.val j.2.2.1.val
def zWindow (p : Parameters) (j : TileIndex (SShape p)) := zAddress p j.1.val j.2.1.val j.2.2.1.val

def aMask (p : Parameters) (j : TileIndex (AShape p)) : Prop := row p j.1.val j.2.2.1.val<p.rows
def sMask (p : Parameters) (j : TileIndex (SShape p)) : Prop :=
  j.2.1.val<steps p ∨ p.noGroups=Bool.true ∧ j.2.1.val=0

def inputChannels (p : Parameters) : List IOChannel :=
  [⟨.fp16,AShape p,p.input⟩,⟨.int,BShape p,p.weight⟩,⟨.fp16,SShape p,p.scale⟩,⟨.int,SShape p,p.zero⟩]
def outputChannels (p : Parameters) : List IOChannel := [⟨.fp16,CShape p,p.output⟩]
abbrev Inputs (A : Type) (p : Parameters) := ChannelValues A (inputChannels p)
abbrev Outputs (A : Type) (p : Parameters) := ChannelValues A (outputChannels p)

def inputAddress (p : Parameters) : (i : Fin (inputChannels p).length) → TileIndex (inputChannels p)[i].shape → Nat :=
  Fin.cases (aWindow p) (Fin.cases (bWindow p) (Fin.cases (sWindow p) (Fin.cases (zWindow p) (fun i => Fin.elim0 i))))
def inputActive (p : Parameters) : (i : Fin (inputChannels p).length) → TileIndex (inputChannels p)[i].shape → Prop :=
  Fin.cases (aMask p) (Fin.cases (fun _ => True) (Fin.cases (sMask p) (Fin.cases (sMask p) (fun i => Fin.elim0 i))))

/-- Pure finite-array lookup by strided address. Repeated read coordinates are
consistent through simultaneous input binding; this function contains no state,
source evaluation or numerical assumption. -/
def readAt {A : Type} {shape : TileShape} (zero : A) (values : TileIndex shape → A)
    (mask : TileIndex shape → Prop) (address : TileIndex shape → Nat) (off : Nat) : A := by
  classical
  exact if h : ∃ j,mask j ∧ address j=off then values (Classical.choose h) else zero

structure Operands (A : Type) where
  a : Nat → A
  weight : Nat → Int
  scale : Nat → A
  zero : Nat → Int

def operands (zero : A) (p : Parameters) (vals : Inputs A p) : Operands A where
  a := readAt zero (vals ⟨0,by simp [inputChannels]⟩) (aMask p) (aWindow p)
  weight := readAt 0 (vals ⟨1,by simp [inputChannels]⟩) (fun _ => True) (bWindow p)
  scale := readAt zero (vals ⟨2,by simp [inputChannels]⟩) (sMask p) (sWindow p)
  zero := readAt 0 (vals ⟨3,by simp [inputChannels]⟩) (sMask p) (zWindow p)

/-- The actual unmasked packed-weight and scale/zero windows are explicit.
The noGroups preloads remain present for empty K loops. A is row-masked only.
The output is separate; input regions may alias, and no global scratch exists. -/
def memory (p : Parameters) : MemoryIO A where
  inputs := inputChannels p
  outputs := outputChannels p
  buffers := [p.input,p.weight,p.scale,p.zero,p.output]
  read := fun i _ => inputAddress p i
  inputMask := fun i _ => inputActive p i
  write := Fin.cases (fun _ j => Geometry.offset (FirstGrid.geometry p) j.1.val j.2) (fun i => Fin.elim0 i)
  outputMask := Fin.cases (fun _ j => Geometry.Active (FirstGrid.geometry p) j.1.val j.2) (fun i => Fin.elim0 i)
  scratch := []
  scratchWrite := fun i => Fin.elim0 i
  scratchMask := fun i => Fin.elim0 i

def invocation (M : Algebra A) (D : Dot A) (p : Parameters) (pid : Nat) : Invocation A where
  kernel := VeriTile.Bench.TritonBenchG.MatmulDequantize.FPTypedFirst.source p
  backend := VeriTile.Bench.TritonBenchG.MatmulDequantize.FPTypedFirst.backend M D
  pids := fun ax => if ax=0 then pid else 0

def program (M : Algebra A) (D : Dot A) (p : Parameters) (jobs : List Nat) : ProgramIO A A where
  toMemoryIO := memory p
  schedule := fun _ => jobs.map (invocation M D p)
  encode := id
  observe := fun _ _ => id

def aValue (M : Algebra A) (p : Parameters) (v : Operands A) (pid k : Nat)
    (i : TileIndex [p.blockRows,p.blockInner]) : A :=
  if row p pid i.1.val<p.rows then v.a (aAddress p pid k i.1.val i.2.1.val)
  else M.cast none .real .fp16 (M.literal none .real 0)
def scaleValue (p : Parameters) (v : Operands A) (pid k c : Nat) := v.scale (sAddress p pid k c)
def weightNibble (p : Parameters) (v : Operands A) (pid k e c : Nat) :=
  Real.signedNibble (v.weight (bAddress p pid k e c)) e
def zeroNibble (p : Parameters) (v : Operands A) (pid k c : Nat) :=
  Real.signedNibble (v.zero (zAddress p pid k c)) (column p pid c)
def bValue (M : Algebra A) (p : Parameters) (v : Operands A) (pid k : Nat)
    (i : TileIndex [p.blockInner,p.blockColumns]) : A :=
  M.binary none .fp16 .sub
    (M.binary (some .fp32) .fp16 .mul
      (M.cast (some .fp32) .real .fp16 (M.fromInt (some .fp32) (weightNibble p v pid k i.1.val i.2.1.val)))
      (scaleValue p v pid k i.2.1.val))
    (M.binary none .fp16 .mul
      (M.cast (some .fp32) .real .fp16 (M.fromInt (some .fp32) (zeroNibble p v pid k i.2.1.val)))
      (scaleValue p v pid k i.2.1.val))

def accumulation (M : Algebra A) (D : Dot A) (p : Parameters) (v : Operands A) (pid : Nat) :
    Nat → TileIndex [p.blockRows,p.blockColumns] → A
  | 0 => fun _ => M.literal (some .fp32) .real 0
  | k+1 => fun i => M.binary (some .fp32) .real .add (accumulation M D p v pid k i)
      (M.cast (some .fp32) .fp32 .real (D.apply (some .fp32) {} (batch:=[])
        (aValue M p v pid k) (bValue M p v pid k) i))

/-- Independent ordered K-block expression with original FP16 operands,
FP32 accumulation and FP16 final cast, for arbitrary original block sizes. -/
def expression (M : Algebra A) (D : Dot A) (p : Parameters) (vals : Inputs A p) : Outputs A p :=
  Fin.cases (fun j => M.cast none .real .fp16 (accumulation M D p
    (operands (M.literal none .real 0) p vals) j.1.val (steps p) j.2)) (fun i => Fin.elim0 i)

def rawInputs (p : Parameters) (vals : Inputs ℝ p) : Real.RawInputs where
  a := fun m k => (operands 0 p vals).a (m*p.inputRowStride+k*p.inputInnerStride)
  weight := fun k n => (operands 0 p vals).weight (k*p.weightInnerStride+n*p.weightColumnStride)
  scale := fun g n => (operands 0 p vals).scale (g*p.scaleGroupStride+n*p.scaleColumnStride)
  zero := fun g n => (operands 0 p vals).zero (g*p.zeroGroupStride+n*p.zeroColumnStride)

/-- Independent quantized matrix product: Σ_k A[m,k]*(q[k,n]-z[k/G,n])*S[k/G,n]. -/
def mathematical (p : Parameters) (vals : Inputs ℝ p) : Outputs ℝ p :=
  Fin.cases (fun j => Real.rawMatmul (rawInputs p vals) p.inner p.groupSize
    (row p j.1.val j.2.1.val) (column p j.1.val j.2.2.1.val)) (fun i => Fin.elim0 i)

def realIO (p : Parameters) (jobs : List Nat) := program Real.scalar Real.firstDot p jobs
def nativeIO (p : Parameters) (jobs : List Nat) := program Native.scalar Native.firstDot p jobs

def observe (p : Parameters) (plan : Native.Plan) (isa : Native.WordISA) :
    (o : Fin (outputChannels p).length) → TileIndex (outputChannels p)[o].shape → Native.Expr → BitVec 32 :=
  Fin.cases (fun j value => (Native.lower plan value j.1.val).eval isa) (fun i => Fin.elim0 i)
def nativeTarget (p : Parameters) (plan : Native.Plan) (isa : Native.WordISA)
    (vals : Inputs Native.Expr p) : ChannelValues (BitVec 32) (outputChannels p) :=
  Fin.cases (fun j => (Native.lower plan
    (expression Native.scalar Native.firstDot p vals ⟨0,by simp [outputChannels]⟩ j) j.1.val).eval isa) (fun i => Fin.elim0 i)
end
end VeriTile.Bench.Repairs.MatmulDequantize.FirstIO


namespace VeriTile.Bench.Repairs.MatmulDequantize.SecondIO
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulDequantizeWhole001
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
open FPTypedSecond (Parameters Dot)
noncomputable section

abbrev AShape (p : Parameters) := [Geometry.count (SecondGrid.geometry p),p.steps,p.blockRows,p.blockInner]
abbrev BShape (p : Parameters) := [Geometry.count (SecondGrid.geometry p),p.steps,p.blockInner,p.blockColumns]
abbrev SShape (p : Parameters) := BShape p
abbrev CShape (p : Parameters) := [Geometry.count (SecondGrid.geometry p),p.blockRows,p.blockColumns]
abbrev OutputLane (p : Parameters) := TileIndex (CShape p)
def row (p : Parameters) (pid r : Nat) := Geometry.blockRow (SecondGrid.geometry p) pid*p.blockRows+r
def column (p : Parameters) (pid c : Nat) := Geometry.blockColumn (SecondGrid.geometry p) pid*p.blockColumns+c

def aAddress (p : Parameters) (pid k r e : Nat) :=
  (row p pid r%p.rows)*p.inputRowStride+e*p.inputInnerStride+k*(p.blockInner*p.inputInnerStride)
def bAddress (p : Parameters) (pid k e c : Nat) :=
  e/8*p.weightInnerStride+(column p pid c%p.columns)*p.weightColumnStride+k*(p.blockInner*p.weightInnerStride/8)
def sAddress (p : Parameters) (pid k e c : Nat) :=
  (e+k*p.blockInner)/p.groupSize*p.scaleGroupStride+(column p pid c%p.columns)*p.scaleColumnStride
def zAddress (p : Parameters) (pid k e c : Nat) :=
  (e+k*p.blockInner)/p.groupSize*p.zeroGroupStride+(column p pid c%p.columns)/8*p.zeroColumnStride

def aWindow (p : Parameters) (j : TileIndex (AShape p)) := aAddress p j.1.val j.2.1.val j.2.2.1.val j.2.2.2.1.val
def bWindow (p : Parameters) (j : TileIndex (BShape p)) := bAddress p j.1.val j.2.1.val j.2.2.1.val j.2.2.2.1.val
def sWindow (p : Parameters) (j : TileIndex (SShape p)) := sAddress p j.1.val j.2.1.val j.2.2.1.val j.2.2.2.1.val
def zWindow (p : Parameters) (j : TileIndex (SShape p)) := zAddress p j.1.val j.2.1.val j.2.2.1.val j.2.2.2.1.val

def aMask (p : Parameters) (j : TileIndex (AShape p)) : Prop :=
  j.2.2.2.1.val+j.2.1.val*p.blockInner<p.inner
def bMask (p : Parameters) (j : TileIndex (BShape p)) : Prop :=
  j.2.2.1.val+j.2.1.val*p.blockInner<p.inner

def inputChannels (p : Parameters) : List IOChannel :=
  [⟨.fp32,AShape p,p.input⟩,⟨.int,BShape p,p.weight⟩,⟨.fp32,SShape p,p.scale⟩,⟨.int,SShape p,p.zero⟩]
def outputChannels (p : Parameters) : List IOChannel := [⟨.fp16,CShape p,p.output⟩]
abbrev Inputs (A : Type) (p : Parameters) := ChannelValues A (inputChannels p)
abbrev Outputs (A : Type) (p : Parameters) := ChannelValues A (outputChannels p)

def inputAddress (p : Parameters) : (i : Fin (inputChannels p).length) → TileIndex (inputChannels p)[i].shape → Nat :=
  Fin.cases (aWindow p) (Fin.cases (bWindow p) (Fin.cases (sWindow p) (Fin.cases (zWindow p) (fun i => Fin.elim0 i))))
def inputActive (p : Parameters) : (i : Fin (inputChannels p).length) → TileIndex (inputChannels p)[i].shape → Prop :=
  Fin.cases (aMask p) (Fin.cases (bMask p) (Fin.cases (bMask p) (Fin.cases (bMask p) (fun i => Fin.elim0 i))))

abbrev readAt := @FirstIO.readAt

structure Operands (A : Type) where
  a : Nat → A
  weight : Nat → Int
  scale : Nat → A
  zero : Nat → Int

def operands (zero : A) (p : Parameters) (vals : Inputs A p) : Operands A where
  a := readAt zero (vals ⟨0,by simp [inputChannels]⟩) (aMask p) (aWindow p)
  weight := readAt 0 (vals ⟨1,by simp [inputChannels]⟩) (bMask p) (bWindow p)
  scale := readAt zero (vals ⟨2,by simp [inputChannels]⟩) (bMask p) (sWindow p)
  zero := readAt 0 (vals ⟨3,by simp [inputChannels]⟩) (bMask p) (zWindow p)

/-- Source windows retain the exact K masks, modulo row/column reads,
packed-pointer quotient order and arbitrary original iteration count.
No additional divisibility requirement, input separation or scratch is added. -/
def memory (p : Parameters) : MemoryIO A where
  inputs := inputChannels p
  outputs := outputChannels p
  buffers := [p.input,p.weight,p.scale,p.zero,p.output]
  read := fun i _ => inputAddress p i
  inputMask := fun i _ => inputActive p i
  write := Fin.cases (fun _ j => Geometry.offset (SecondGrid.geometry p) j.1.val j.2) (fun i => Fin.elim0 i)
  outputMask := Fin.cases (fun _ j => Geometry.Active (SecondGrid.geometry p) j.1.val j.2) (fun i => Fin.elim0 i)
  scratch := []
  scratchWrite := fun i => Fin.elim0 i
  scratchMask := fun i => Fin.elim0 i

def invocation (M : Algebra A) (D : Dot A) (p : Parameters) (pid : Nat) : Invocation A where
  kernel := VeriTile.Bench.Repairs.MatmulDequantizeWhole001.FPTypedSecond.source p
  backend := VeriTile.Bench.Repairs.MatmulDequantizeWhole001.FPTypedSecond.backend M D
  pids := fun ax => if ax=0 then pid else 0

def program (M : Algebra A) (D : Dot A) (p : Parameters) (jobs : List Nat) : ProgramIO A A where
  toMemoryIO := memory p
  schedule := fun _ => jobs.map (invocation M D p)
  encode := id
  observe := fun _ _ => id

def innerActive (p : Parameters) (k e : Nat) : Prop := e+k*p.blockInner<p.inner
instance (p : Parameters) (k e : Nat) : Decidable (innerActive p k e) := inferInstanceAs (Decidable (_<_))

def aValue (M : Algebra A) (p : Parameters) (v : Operands A) (pid k : Nat)
    (i : TileIndex [p.blockRows,p.blockInner]) : A := M.cast (some .fp32) .fp32 .fp16
  (if innerActive p k i.2.1.val then v.a (aAddress p pid k i.1.val i.2.1.val)
   else M.cast none .real .fp32 (M.literal none .real 0))
def difference (p : Parameters) (v : Operands A) (pid k e c : Nat) :=
  Real.signedNibble (if innerActive p k e then v.weight (bAddress p pid k e c) else 0) e-
    Real.signedNibble (if innerActive p k e then v.zero (zAddress p pid k e c) else 0)
      (column p pid c%p.columns)
def bValue (M : Algebra A) (p : Parameters) (v : Operands A) (pid k : Nat)
    (i : TileIndex [p.blockInner,p.blockColumns]) : A :=
  M.cast none .fp32 .fp16
    (M.binary (some .fp32) .fp32 .mul
      (M.cast (some .fp32) .real .fp32 (M.fromInt (some .fp32) (difference p v pid k i.1.val i.2.1.val)))
      (if innerActive p k i.1.val then v.scale (sAddress p pid k i.1.val i.2.1.val)
       else M.cast none .real .fp32 (M.literal none .real 0)))

def accumulation (M : Algebra A) (D : Dot A) (p : Parameters) (v : Operands A) (pid : Nat) :
    Nat → TileIndex [p.blockRows,p.blockColumns] → A
  | 0 => fun _ => M.literal (some .fp32) .real 0
  | k+1 => fun i => M.binary (some .fp32) .real .add (accumulation M D p v pid k i)
      (M.cast (some .fp32) .fp32 .real (D.apply (some .fp32) {} (batch:=[])
        (aValue M p v pid k) (fun j => M.cast (some .fp32) .fp16 .fp16 (bValue M p v pid k j)) i))

/-- Independent ordered K-block expression with FP32 inputs narrowed to
FP16 dot operands, FP32 accumulation and FP16 final output. -/
def expression (M : Algebra A) (D : Dot A) (p : Parameters) (vals : Inputs A p) : Outputs A p :=
  Fin.cases (fun j => M.cast none .real .fp16 (accumulation M D p
    (operands (M.literal none .real 0) p vals) j.1.val p.steps j.2)) (fun i => Fin.elim0 i)

def rawInputs (p : Parameters) (vals : Inputs ℝ p) : Real.RawInputs where
  a := fun m k => (operands 0 p vals).a (m*p.inputRowStride+k*p.inputInnerStride)
  weight := fun k n => (operands 0 p vals).weight (k*p.weightInnerStride+n*p.weightColumnStride)
  scale := fun g n => (operands 0 p vals).scale (g*p.scaleGroupStride+n*p.scaleColumnStride)
  zero := fun g n => (operands 0 p vals).zero (g*p.zeroGroupStride+n*p.zeroColumnStride)

/-- Independent quantized matrix product: Σ_k A[m,k]*(q[k,n]-z[k/G,n])*S[k/G,n]. -/
def mathematical (p : Parameters) (vals : Inputs ℝ p) : Outputs ℝ p :=
  Fin.cases (fun j => Real.rawMatmul (rawInputs p vals) p.inner p.groupSize
    (row p j.1.val j.2.1.val) (column p j.1.val j.2.2.1.val)) (fun i => Fin.elim0 i)

def realIO (p : Parameters) (jobs : List Nat) := program Real.scalar Real.secondDot p jobs
def nativeIO (p : Parameters) (jobs : List Nat) := program Native.scalar Native.secondDot p jobs

def observe (p : Parameters) (plan : Native.Plan) (isa : Native.WordISA) :
    (o : Fin (outputChannels p).length) → TileIndex (outputChannels p)[o].shape → Native.Expr → BitVec 32 :=
  Fin.cases (fun j value => (Native.lower plan value j.1.val).eval isa) (fun i => Fin.elim0 i)
def nativeTarget (p : Parameters) (plan : Native.Plan) (isa : Native.WordISA)
    (vals : Inputs Native.Expr p) : ChannelValues (BitVec 32) (outputChannels p) :=
  Fin.cases (fun j => (Native.lower plan
    (expression Native.scalar Native.secondDot p vals ⟨0,by simp [outputChannels]⟩ j) j.1.val).eval isa) (fun i => Fin.elim0 i)
end
end VeriTile.Bench.Repairs.MatmulDequantize.SecondIO


namespace VeriTile.Bench.Repairs.MatmulDequantize.ReaderIO
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulDequantizeWhole001
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section

abbrev Rows (g : Geometry.Layout) := Fin (Geometry.gridRows g*g.blockRows)
abbrev Columns (g : Geometry.Layout) := Fin (Geometry.gridColumns g*g.blockColumns)

/-- All tile positions, including padding, have unique mathematical coordinates. -/
def grid (g : Geometry.Layout) (valid : Geometry.Valid g) :
    Fin (Geometry.count g) ≃ Fin (Geometry.gridRows g) × Fin (Geometry.gridColumns g) := by
  by_cases h : 0<Geometry.gridColumns g
  · exact Equiv.ofBijective
      (GroupedGrid.mapping _ _ _ h valid.groupRows) (GroupedGrid.bijective _ _ _ h valid.groupRows)
  · have hz : Geometry.gridColumns g=0 := by omega
    exact { toFun := fun j => False.elim (by have := j.isLt; simp [Geometry.count,hz] at this)
            invFun := fun q => False.elim (by have := q.2.isLt; omega)
            left_inv := fun j => False.elim (by have := j.isLt; simp [Geometry.count,hz] at this)
            right_inv := fun q => False.elim (by have := q.2.isLt; omega) }

def coordinates (g : Geometry.Layout) (valid : Geometry.Valid g) :
    TileIndex [Geometry.count g,g.blockRows,g.blockColumns] ≃ Rows g × Columns g where
  toFun := fun j =>
    (finProdFinEquiv ((grid g valid j.1).1,j.2.1),
     finProdFinEquiv ((grid g valid j.1).2,j.2.2.1))
  invFun := fun q =>
    ((grid g valid).symm ((finProdFinEquiv.symm q.1).1,(finProdFinEquiv.symm q.2).1),
      (finProdFinEquiv.symm q.1).2,(finProdFinEquiv.symm q.2).2,())
  left_inv := by
    intro j
    simp only [Equiv.symm_apply_apply]
    change ((grid g valid).symm (grid g valid j.1),j.2.1,j.2.2.1,())=j
    rw [Equiv.symm_apply_apply]
  right_inv := by
    intro q
    simp only [Equiv.apply_symm_apply]
    change (finProdFinEquiv (finProdFinEquiv.symm q.1),
      finProdFinEquiv (finProdFinEquiv.symm q.2))=q
    simp only [Equiv.apply_symm_apply]

def matrix (A : Type) (g : Geometry.Layout) (valid : Geometry.Valid g) :
    (TileIndex [Geometry.count g,g.blockRows,g.blockColumns] → A) ≃
      (Rows g → Columns g → A) where
  toFun := fun xs r c => xs ((coordinates g valid).symm (r,c))
  invFun := fun xs j => xs (coordinates g valid j).1 (coordinates g valid j).2
  left_inv := by intro xs; funext j; exact congrArg xs ((coordinates g valid).symm_apply_apply j)
  right_inv := by intro xs; funext r c; simp only [Equiv.apply_symm_apply]

structure GPTQInputs (p : VeriTile.Bench.TritonBenchG.MatmulDequantize.FPTypedFirst.Parameters) where
  backing : FirstIO.Inputs ℝ p
  arrays : Real.RawInputs
  connected : arrays=FirstIO.rawInputs p backing

def gptqInputs (p : VeriTile.Bench.TritonBenchG.MatmulDequantize.FPTypedFirst.Parameters) :
    FirstIO.Inputs ℝ p ≃ GPTQInputs p where
  toFun := fun xs => ⟨xs,FirstIO.rawInputs p xs,rfl⟩
  invFun := GPTQInputs.backing
  left_inv := fun _ => rfl
  right_inv := by rintro ⟨xs,arrays,connected⟩; cases connected; rfl

structure S2Inputs (p : FPTypedSecond.Parameters) where
  backing : SecondIO.Inputs ℝ p
  arrays : Real.RawInputs
  connected : arrays=SecondIO.rawInputs p backing

def s2Inputs (p : FPTypedSecond.Parameters) : SecondIO.Inputs ℝ p ≃ S2Inputs p where
  toFun := fun xs => ⟨xs,SecondIO.rawInputs p xs,rfl⟩
  invFun := S2Inputs.backing
  left_inv := fun _ => rfl
  right_inv := by rintro ⟨xs,arrays,connected⟩; cases connected; rfl

/-- Complete per-coordinate windows. Repeated packed addresses and inactive
padding are retained separately, exactly as in the original finite input type. -/
structure DequantInputs (g : Geometry.Layout) where
  weight : Rows g → Columns g → Int
  zero : Rows g → Columns g → Int
  scale : Rows g → Columns g → ℝ

def gptq (p : VeriTile.Bench.TritonBenchG.MatmulDequantize.FPTypedFirst.Parameters)
    (valid : Geometry.Valid (FirstGrid.geometry p)) (jobs : List Nat) :
    LogicalProgramIO ℝ ℝ (GPTQInputs p) (Rows (FirstGrid.geometry p) → Columns (FirstGrid.geometry p) → ℝ) where
  core := FirstIO.realIO p jobs
  inputs := gptqInputs p
  outputs := (ChannelValues.singleEquiv ℝ _).trans (matrix ℝ _ valid)

def s2 (p : FPTypedSecond.Parameters) (valid : Geometry.Valid (SecondGrid.geometry p)) (jobs : List Nat) :
    LogicalProgramIO ℝ ℝ (S2Inputs p) (Rows (SecondGrid.geometry p) → Columns (SecondGrid.geometry p) → ℝ) where
  core := SecondIO.realIO p jobs
  inputs := s2Inputs p
  outputs := (ChannelValues.singleEquiv ℝ _).trans (matrix ℝ _ valid)

def d32Inputs (p : Dequant32.Parameters) (valid : Geometry.Valid (DequantGrid32.geometry p)) :
    D32IO.Inputs ℝ p ≃ DequantInputs (DequantGrid32.geometry p) where
  toFun := fun xs =>
    ⟨matrix Int _ valid (xs (0 : Fin 3)),matrix Int _ valid (xs (1 : Fin 3)),
      matrix ℝ _ valid (xs (2 : Fin 3))⟩
  invFun := fun xs => Fin.cases ((matrix Int _ valid).symm xs.weight)
    (Fin.cases ((matrix Int _ valid).symm xs.zero)
      (Fin.cases ((matrix ℝ _ valid).symm xs.scale) (fun i => i.elim0)))
  left_inv := by
    intro xs
    funext i j
    fin_cases i
    · exact congrFun ((matrix Int _ valid).symm_apply_apply (xs (0 : Fin 3))) j
    · exact congrFun ((matrix Int _ valid).symm_apply_apply (xs (1 : Fin 3))) j
    · exact congrFun ((matrix ℝ _ valid).symm_apply_apply (xs (2 : Fin 3))) j
  right_inv := by
    rintro ⟨weight,zero,scale⟩
    change DequantInputs.mk (matrix Int _ valid ((matrix Int _ valid).symm weight))
      (matrix Int _ valid ((matrix Int _ valid).symm zero))
      (matrix ℝ _ valid ((matrix ℝ _ valid).symm scale))=DequantInputs.mk weight zero scale
    simp only [Equiv.apply_symm_apply]

def d32 (p : Dequant32.Parameters) (valid : Geometry.Valid (DequantGrid32.geometry p)) (jobs : List Nat) :
    LogicalProgramIO ℝ ℝ (DequantInputs (DequantGrid32.geometry p))
      (Rows (DequantGrid32.geometry p) → Columns (DequantGrid32.geometry p) → ℝ) where
  core := D32IO.realIO p jobs
  inputs := d32Inputs p valid
  outputs := (ChannelValues.singleEquiv ℝ _).trans (matrix ℝ _ valid)


def d16Inputs (p : VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.Parameters) (valid : Geometry.Valid (DequantGrid16.geometry p)) :
    D16IO.Inputs ℝ p ≃ DequantInputs (DequantGrid16.geometry p) where
  toFun := fun xs =>
    ⟨matrix Int _ valid (xs (0 : Fin 3)),matrix Int _ valid (xs (1 : Fin 3)),
      matrix ℝ _ valid (xs (2 : Fin 3))⟩
  invFun := fun xs => Fin.cases ((matrix Int _ valid).symm xs.weight)
    (Fin.cases ((matrix Int _ valid).symm xs.zero)
      (Fin.cases ((matrix ℝ _ valid).symm xs.scale) (fun i => i.elim0)))
  left_inv := by
    intro xs
    funext i j
    fin_cases i
    · exact congrFun ((matrix Int _ valid).symm_apply_apply (xs (0 : Fin 3))) j
    · exact congrFun ((matrix Int _ valid).symm_apply_apply (xs (1 : Fin 3))) j
    · exact congrFun ((matrix ℝ _ valid).symm_apply_apply (xs (2 : Fin 3))) j
  right_inv := by
    rintro ⟨weight,zero,scale⟩
    change DequantInputs.mk (matrix Int _ valid ((matrix Int _ valid).symm weight))
      (matrix Int _ valid ((matrix Int _ valid).symm zero))
      (matrix ℝ _ valid ((matrix ℝ _ valid).symm scale))=DequantInputs.mk weight zero scale
    simp only [Equiv.apply_symm_apply]

def d16 (p : VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.Parameters) (valid : Geometry.Valid (DequantGrid16.geometry p)) (jobs : List Nat) :
    LogicalProgramIO ℝ ℝ (DequantInputs (DequantGrid16.geometry p))
      (Rows (DequantGrid16.geometry p) → Columns (DequantGrid16.geometry p) → ℝ) where
  core := D16IO.realIO p jobs
  inputs := d16Inputs p valid
  outputs := (ChannelValues.singleEquiv ℝ _).trans (matrix ℝ _ valid)

end
end VeriTile.Bench.Repairs.MatmulDequantize.ReaderIO
