import bench.repairs.tritonbench_g.int8_matmul_kernel_modular_checked.Proofs.TileSource

namespace VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.TileModel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001 (Byte DefinedCast)
noncomputable section

abbrev AIndex (p : Parameters) := TileIndex [p.blockRows,p.blockInner]
abbrev BIndex (p : Parameters) := TileIndex [p.blockInner,p.blockColumns]
abbrev CIndex (p : Parameters) := TileIndex [p.blockRows,p.blockColumns]
def blocksM (p : Parameters) : Nat := (p.rows+p.blockRows-1)/p.blockRows
def blocksN (p : Parameters) : Nat := (p.columns+p.blockColumns-1)/p.blockColumns
def firstRow (p : Parameters) (s : State Word32) : Nat :=
  s.pids 0/(p.groupRows*blocksN p)*p.groupRows
def rowsInGroup (p : Parameters) (s : State Word32) : Nat :=
  min (blocksM p-firstRow p s) p.groupRows
def rowBlock (p : Parameters) (s : State Word32) : Nat :=
  firstRow p s+((s.pids 0%(p.groupRows*blocksN p))%rowsInGroup p s)
def columnBlock (p : Parameters) (s : State Word32) : Nat :=
  (s.pids 0%(p.groupRows*blocksN p))/rowsInGroup p s
def inputRow (p : Parameters) (s : State Word32) (i : Fin p.blockRows) : Nat :=
  (rowBlock p s*p.blockRows+i.val)%p.rows
def inputColumn (p : Parameters) (s : State Word32) (i : Fin p.blockColumns) : Nat :=
  (columnBlock p s*p.blockColumns+i.val)%p.columns

def aOffset (p : Parameters) (s : State Word32) (field step : Nat) (i : AIndex p) : Nat :=
  inputRow p s i.1*p.inputRowStride+i.2.1.val*p.inputInnerStride+
    (field*p.steps+step)*(p.blockInner*p.inputInnerStride)
def bOffset (p : Parameters) (s : State Word32) (step : Nat) (i : BIndex p) : Nat :=
  i.1.val*p.weightInnerStride+inputColumn p s i.2.1*p.weightColumnStride+
    step*(p.blockInner*p.weightInnerStride)

/-- Raw FP32 words inhabit the frontend's floating `.real` carrier under its
explicit `.fp32` precision marker. No signed-int storage or low-byte
reinterpretation replaces the source numeric conversion. -/
def aWord (p : Parameters) (s : State Word32) (field step : Nat) (i : AIndex p) : Word32 :=
  if i.2.1.val < p.inner-(field*p.steps+step)*p.blockInner then
    (s.mem p.input (aOffset p s field step i)).read .real else 0
def aValue (cast : Word32 → Byte) (p : Parameters) (s : State Word32)
    (field step : Nat) (i : AIndex p) : Int := (cast (aWord p s field step i)).toInt
def bValue (p : Parameters) (s : State Word32) (field step : Nat) (i : BIndex p) : Int :=
  let packed := if i.1.val < p.inner then
    (s.mem p.weight (bOffset p s step i)).read .nat else 0
  wrap 8 (Int.ofNat ((packed &&& (3 <<< (2*field))) >>> (2*field)))-1

theorem bValue_source (p : Parameters) (s : State Word32) (field step : Nat) (i : BIndex p) :
    bValue p s field step i =
      (↑(((if i.1.val < p.inner then
        (s.mem p.weight (bOffset p s step i)).read .nat else 0) &&&
        (3 <<< (2*field))) >>> (2*field)) : Int)-1 := by
  unfold bValue
  dsimp only
  rw [extracted_field]
  congr 1
  have small (value : Nat) (bound : value ≤ 3) : wrap 8 (Int.ofNat value) = (value : Int) := by
    apply BitVec.toInt_ofInt_eq_self (by decide) <;> norm_num <;> omega
  exact small _ Nat.and_le_right

/-- Exactly the values the source may numerically cast on this launched tile.
The connection to wrapper admission must be proved using the original masks and
legal pointer/index bounds; this is not an added Python guard. -/
def InputDefined (p : Parameters) (s : State Word32) : Prop :=
  ∀ field, field < 4 → ∀ step, step < p.steps → ∀ i : AIndex p,
    DefinedCast (aWord p s field step i)

def innerValues (cast : Word32 → Byte) (p : Parameters) (s : State Word32)
    (field : Nat) (initial : CIndex p → Int) : Nat → CIndex p → Int
  | 0 => initial
  | step+1 => fun i => signedAdd32 (innerValues cast p s field initial step i)
      (wordDot (aValue cast p s field step) (bValue p s field step) i)
def outerValues (cast : Word32 → Byte) (p : Parameters) (s : State Word32) : Nat → CIndex p → Int
  | 0 => fun _ => 0
  | field+1 => innerValues cast p s field (outerValues cast p s field) p.steps

def outputOffset (p : Parameters) (s : State Word32) (i : CIndex p) : Nat :=
  p.outputRowStride*(rowBlock p s*p.blockRows+i.1.val)+
    p.outputColumnStride*(columnBlock p s*p.blockColumns+i.2.1.val)
def active (p : Parameters) (s : State Word32) (i : CIndex p) : Prop :=
  rowBlock p s*p.blockRows+i.1.val < p.rows ∧
    columnBlock p s*p.blockColumns+i.2.1.val < p.columns
instance (p : Parameters) (s : State Word32) : DecidablePred (active p s) :=
  fun _ => inferInstanceAs (Decidable (_ ∧ _))

theorem dot_inputs_valid (cast : Word32 → Byte) (p : Parameters) (s : State Word32)
    (field step : Nat) :
    FP.IntegerDot.valid (α := Word32) (aValue cast p s field step) ∧
      FP.IntegerDot.valid (α := Word32) (bValue p s field step) := by
  constructor
  · intro i
    change -128 ≤ (cast (aWord p s field step i)).toInt ∧
      (cast (aWord p s field step i)).toInt < 128
    have h := byte_signed_bounds (cast (aWord p s field step i))
    exact ⟨h.1, by omega⟩
  · intro i
    dsimp only [bValue]
    rw [extracted_field]
    have generic (value : Nat) (bound : value ≤ 3) :
        FP.IntegerDot.signed8 (wrap 8 (Int.ofNat value)-1) := by
      have hcast : wrap 8 (Int.ofNat value) = Int.ofNat value := by
        apply BitVec.toInt_ofInt_eq_self (by decide) <;> norm_num <;> omega
      rw [hcast]
      unfold FP.IntegerDot.signed8
      change -128 ≤ (value : Int)-1 ∧ (value : Int)-1 < 128
      omega
    exact generic _ Nat.and_le_right

#axiomsClean dot_inputs_valid
#axiomsClean bValue_source
end
end VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.TileModel
