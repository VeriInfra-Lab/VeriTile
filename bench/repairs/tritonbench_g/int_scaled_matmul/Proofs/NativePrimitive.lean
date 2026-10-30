import bench.repairs.tritonbench_g.int_scaled_matmul.IO
import bench.repairs.tritonbench_g.int_scaled_matmul.Proofs.KernelNative

namespace VeriTile.Bench.Repairs.IntScaledMatmul.NativePrimitive
open VeriTile.Triton FP.Structural Revision001
noncomputable section
open Classical
local instance : Inhabited KernelModel.Word := ⟨FP.Value.zero .fp32⟩

/-- Fixed 32-product saturating MMA instructions for the selected tile.
The source's signed8 loads always reach this branch. Other shapes or operands
are a total extension of the primitive algebra and never a native claim. -/
def dot (config : Contract.Config) (batch : TileShape) {m k n : Nat}
    (a : TileIndex (batch++[m,k]) → Int) (b : TileIndex (batch++[k,n]) → Int) :
    Option (TileIndex (batch++[m,n]) → Int) :=
  if shape : batch=[] ∧ m=config.blockM ∧ k=config.blockK ∧ n=config.blockN then by
    rcases shape with ⟨rfl,rfl,rfl,rfl⟩
    exact if typed : (∀ i,FP.IntegerDot.signed8 (a i)) ∧ (∀ i,FP.IntegerDot.signed8 (b i)) then
      some (fun i => SaturatingTile.mmaBlocks (SaturatingTile.tileProduct config a b i)
        (IntegerInstructions.partition config.blockK) 0 0)
    else some (fun i => (FP.IntegerDot.wordDot [] a b i).toInt)
  else some (fun i => (FP.IntegerDot.wordDot batch a b i).toInt)

/-- This finite instruction identity is proved from signed8 bounds and the
checked tile size. It assumes neither whole-K no-overflow nor a numeric law. -/
theorem dot_eq (config : Contract.Config) (legal : Contract.ConfigContract config)
    (batch : TileShape) {m k n : Nat}
    (a : TileIndex (batch++[m,k]) → Int) (b : TileIndex (batch++[k,n]) → Int) :
    dot config batch a b=some (fun i => (FP.IntegerDot.wordDot batch a b i).toInt) := by
  unfold dot
  split
  · rename_i shape
    rcases shape with ⟨rfl,rfl,rfl,rfl⟩
    dsimp only
    split
    · rename_i typed
      congr 1
      funext i
      exact (SaturatingTile.source_dot_eq config legal a b typed.1 typed.2 i _
        (IntegerInstructions.partition_cover _)).symm
    · rfl
  · rfl

theorem fixed_instructions (config : Contract.Config)
    (a : TileIndex [config.blockM,config.blockK] → Int)
    (b : TileIndex [config.blockK,config.blockN] → Int)
    (ha : ∀ i,FP.IntegerDot.signed8 (a i)) (hb : ∀ i,FP.IntegerDot.signed8 (b i)) :
    dot config [] a b=some (fun i =>
      SaturatingTile.mmaBlocks (SaturatingTile.tileProduct config a b i)
        (IntegerInstructions.partition config.blockK) 0 0) := by
  unfold dot
  rw [dif_pos (show ([] : TileShape)=[] ∧ config.blockM=config.blockM ∧ config.blockK=config.blockK ∧ config.blockN=config.blockN from ⟨rfl,rfl,rfl,rfl⟩)]
  dsimp only
  rw [dif_pos ⟨ha,hb⟩]
  rfl

def intBackend (rest : Algebra KernelModel.Word) (config : Contract.Config) : Algebra KernelModel.Word :=
  { rest with integerDot := fun _ {batch} {_} {_} {_} a b => dot config batch a b }

theorem intBackend_eq (rest : Algebra KernelModel.Word) (config : Contract.Config)
    (legal : Contract.ConfigContract config) : intBackend rest config=BlockWord.backend rest := by
  unfold intBackend
  simp only [dot_eq config legal]
  rfl

/-- The inspected scalar multiply order is row-scale times promoted carry. -/
def scaledBinary (rest : Algebra KernelModel.Word) (undefined : KernelModel.Word → BitVec 32)
    (precision : Option ComputeDType) (dtype : FloatDType) (op : Binary)
    (a b : KernelModel.Word) : KernelModel.Word :=
  if FP.Software.fp32Context precision dtype ∧ op=.mul then FP.mul Native.sourceConfig b a
  else (ScaledWord.backend rest Native.sourceConfig (KernelModel.fallback undefined)).binary precision dtype op a b

theorem scaledBinary_eq (rest : Algebra KernelModel.Word) (undefined : KernelModel.Word → BitVec 32)
    (precision : Option ComputeDType) (dtype : FloatDType) (op : Binary)
    (a b : KernelModel.Word) :
    scaledBinary rest undefined precision dtype op a b=
      (ScaledWord.backend rest Native.sourceConfig (KernelModel.fallback undefined)).binary precision dtype op a b := by
  unfold scaledBinary
  split
  · rename_i hit
    rcases hit with ⟨context,rfl⟩
    simp only [ScaledWord.backend,FP.IntegerConversion.fp32TotalBackend,
      FP.Software.scalarFP32,context,↓reduceIte]
    exact FP.ScalarOpsLaws.mul_comm _ _ _
  · rfl

def scaledBackend (rest : Algebra KernelModel.Word) (undefined : KernelModel.Word → BitVec 32)
    (config : Contract.Config) : Algebra KernelModel.Word :=
  { ScaledWord.backend rest Native.sourceConfig (KernelModel.fallback undefined) with
    binary := scaledBinary rest undefined
    integerDot := fun _ {batch} {_} {_} {_} a b => dot config batch a b }

theorem scaledBackend_eq (rest : Algebra KernelModel.Word) (undefined : KernelModel.Word → BitVec 32)
    (config : Contract.Config) (legal : Contract.ConfigContract config) :
    scaledBackend rest undefined config=
      ScaledWord.backend rest Native.sourceConfig (KernelModel.fallback undefined) := by
  have binary : scaledBinary rest undefined=
      (ScaledWord.backend rest Native.sourceConfig (KernelModel.fallback undefined)).binary := by
    funext precision dtype op a b
    exact scaledBinary_eq rest undefined precision dtype op a b
  unfold scaledBackend
  rw [binary]
  simp only [dot_eq config legal]
  rfl

/-- Every actual checked block-pointer load reaches the fixed MMA branch;
the algebra's total extension is irrelevant to the selected source. -/
theorem block_loads (x : FiniteExecution.Launch) (s : State KernelModel.Word) (k : Nat) :
    dot x.config [] (BlockWord.aValue (FiniteExecution.blockParameters x) s k)
      (BlockWord.bValue (FiniteExecution.blockParameters x) s k)=
      some (fun i => SaturatingTile.mmaBlocks
        (SaturatingTile.tileProduct x.config (BlockWord.aValue (FiniteExecution.blockParameters x) s k)
          (BlockWord.bValue (FiniteExecution.blockParameters x) s k) i)
        (IntegerInstructions.partition x.config.blockK) 0 0) :=
  fixed_instructions x.config _ _ (fun _ => SaturatingTile.wrap_signed8 _)
    (fun _ => SaturatingTile.wrap_signed8 _)

/-- Both EVEN_K paths cast the loaded operands to signed8 before the dot. -/
theorem scaled_loads (x : FiniteExecution.Launch) (scale : RegionName)
    (s : State KernelModel.Word) (k : Nat) :
    dot x.config [] (ScaledWord.aValue (FiniteExecution.scaledParameters x scale) s k)
      (ScaledWord.bValue (FiniteExecution.scaledParameters x scale) s k)=
      some (fun i => SaturatingTile.mmaBlocks
        (SaturatingTile.tileProduct x.config (ScaledWord.aValue (FiniteExecution.scaledParameters x scale) s k)
          (ScaledWord.bValue (FiniteExecution.scaledParameters x scale) s k) i)
        (IntegerInstructions.partition x.config.blockK) 0 0) :=
  fixed_instructions x.config _ _ (fun _ => SaturatingTile.wrap_signed8 _)
    (fun _ => SaturatingTile.wrap_signed8 _)

#axiomsClean dot_eq
#axiomsClean intBackend_eq
#axiomsClean scaledBackend_eq
#axiomsClean block_loads
#axiomsClean scaled_loads
end
end VeriTile.Bench.Repairs.IntScaledMatmul.NativePrimitive
