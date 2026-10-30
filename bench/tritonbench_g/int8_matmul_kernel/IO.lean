import VeriTile.Triton.Float.LogicalProgramIO
import bench.tritonbench_g.int8_matmul_kernel.Proofs.TileOutput
import VeriTile.Triton.Float.ProgramIO
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.PublicIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001
open TileModel
noncomputable section

/-- One source program in the existing grouped complete grid. Only axis zero
is supplied; source-local registers are computed by the selected quotation. -/
def launch (cast : Word32 → Byte) (rest : Algebra Word32) (layout : Layout)
    (s : Shape) (c : Fin 15) (pid : Nat) : Invocation Word32 where
  kernel := source (parameters layout s c)
  backend := backend cast rest
  pids := fun _ => pid
  setPids := fun ax => decide (ax=0)

/-- Explicit FP32 input / unsigned packed-byte / signed INT32 output channels.
Their Structural carrier tags are respectively real/nat/int; actual source
precision and casts remain in the unchanged quotation. Arbitrary original
input strides and input/weight aliases remain admitted. No private scratch. -/
def matmulIO (cast : Word32 → Byte) (rest : Algebra Word32) (layout : Layout)
    (s : Shape) (c : Fin 15) : ProgramIO Word32 Word32 where
  inputs := [⟨.real,[s.m,s.k],layout.input⟩,⟨.nat,[s.n,s.k/4],layout.weight⟩]
  outputs := [⟨.int,[s.m,s.n],layout.output⟩]
  scratch := []
  buffers := [layout.input,layout.weight,layout.output]
  read := fun port _ => match port with
    | ⟨0,_⟩ => fun i => i.1.val*layout.inputRowStride+i.2.1.val*layout.inputInnerStride
    | ⟨1,_⟩ => fun i => i.2.1.val*layout.weightInnerStride+i.1.val*layout.weightColumnStride
  inputMask := fun _ _ _ => True
  write := fun port _ => match port with
    | ⟨0,_⟩ => fun i => i.1.val*max s.n 1+i.2.1.val
  outputMask := fun _ _ _ => True
  scratchWrite := fun i => Fin.elim0 i
  scratchMask := fun i => Fin.elim0 i
  schedule := fun _ => (List.range (s.programs c)).map (launch cast rest layout s c)
  encode := id
  observe := fun _ _ => id

abbrev Inputs (layout : Layout) (s : Shape) := ChannelValues Word32
  [⟨.real,[s.m,s.k],layout.input⟩,⟨.nat,[s.n,s.k/4],layout.weight⟩]
abbrev Outputs (layout : Layout) (s : Shape) := ChannelValues Word32
  [⟨.int,[s.m,s.n],layout.output⟩]

/-- Zero extension is only a logical indexing convention outside the finite
channels. The formula reads exclusively indices in the original matrices. -/
def inputRow (layout : Layout) (s : Shape) (xs : Inputs layout s) (m : Fin s.m)
    (k : Nat) : Word32 := if h : k<s.k then xs ⟨0,by simp⟩ (m,⟨k,h⟩,PUnit.unit) else 0

def weightRow (layout : Layout) (s : Shape) (xs : Inputs layout s) (n : Fin s.n)
    (q : Nat) : Word32 := if h : q<s.k/4 then
      BitVec.ofNat 32 (xs ⟨1,by simp⟩ (n,⟨q,h⟩,PUnit.unit)) else 0

/-- The original defined FP32-to-signed8 cast domain and unsigned byte domain.
These are logical value conditions, not assumed source-execution facts. -/
def Domain (layout : Layout) (s : Shape) (xs : Inputs layout s) : Prop :=
  (∀ i : TileIndex [s.m,s.k],DefinedCast (xs ⟨0,by simp⟩ i)) ∧
  (∀ i : TileIndex [s.n,s.k/4],Nat.lt (xs ⟨1,by simp⟩ i) 256)

/-- Independent dense signed integer sum, followed by the source's explicit
low-32-bit projection and signed INT32 interpretation. -/
def formula (layout : Layout) (s : Shape) (xs : Inputs layout s) : Outputs layout s :=
  fun port => match port with
  | ⟨0,_⟩ => fun i => (denseWordTarget s.k (inputRow layout s xs i.1)
    (weightRow layout s xs i.2.1)).toInt

/-- Existing fixed native cast, K32 instruction and add.u32 observation. This
is the retained source-to-native result, not an invented second kernel. -/
def nativeFormula (cast : Word32 → Byte)
    (instruction : Instruction → (Fin 32 → Byte) → (Fin 32 → Byte) → Word32 → Word32)
    (add32 : Word32 → Word32 → Word32) (layout : Layout) (s : Shape) (c : Fin 15)
    (xs : Inputs layout s) : Outputs layout s := fun port => match port with
  | ⟨0,_⟩ => fun i => (nativeModularAccumulation cast instruction add32 s c
    (inputRow layout s xs i.1) (weightRow layout s xs i.2.1)).toInt

#specNonCircular formula avoiding [source,outerValues,innerValues,modularAccumulation]

/-- Two actual retained autotune specializations. The candidate changes the
matrix tile and complete CTA grid, while retaining the source's FP32-to-int8
conversion, packed uint8 reads, zero-seeded dot and explicit add.u32. -/
def configuredVariant (cast : Word32 → Byte) (rest : Algebra Word32) (layout : Layout)
    (s : Shape) (left right : Fin 15) : ProgramIO.Variant (matmulIO cast rest layout s left) where
  schedule := (matmulIO cast rest layout s right).schedule
  observe := fun _ _ => id

def original (cast : Word32 → Byte) (rest : Algebra Word32) (layout : Layout)
    (s : Shape) (left : Fin 15) :=
  ProgramIO.Implementation.original (matmulIO cast rest layout s left) (Domain layout s)

def configured (cast : Word32 → Byte) (rest : Algebra Word32) (layout : Layout)
    (s : Shape) (left right : Fin 15) :=
  ProgramIO.Implementation.ofVariant (configuredVariant cast rest layout s left right) (Domain layout s)

end
end VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.PublicIO

namespace VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.PublicIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001 TileModel
noncomputable section
structure LogicalInputs (s : Shape) where
  input : Fin s.m → Fin s.k → Word32
  packed : Fin s.n → Fin (s.k/4) → Nat

def logicalInputs (layout : Layout) (s : Shape) : Inputs layout s ≃ LogicalInputs s where
  toFun xs := ⟨fun r k => xs 0 (r,k,()),fun c q => xs 1 (c,q,())⟩
  invFun xs := fun o => match o with
    | ⟨0,_⟩ => fun j => xs.input j.1 j.2.1
    | ⟨1,_⟩ => fun j => xs.packed j.1 j.2.1
  left_inv := by intro xs; funext o; fin_cases o <;> funext j <;> rcases j with ⟨r,c,⟨⟩⟩ <;> rfl
  right_inv := by intro xs; cases xs; rfl

def logicalOutputs (layout : Layout) (s : Shape) : Outputs layout s ≃ (Fin s.m → Fin s.n → Int) where
  toFun xs := fun r c => xs 0 (r,c,())
  invFun xs := fun o => match o with | ⟨0,_⟩ => fun j => xs j.1 j.2.1
  left_inv := by intro xs; funext o; fin_cases o; funext j; rcases j with ⟨r,c,⟨⟩⟩; rfl
  right_inv := by intro xs; rfl

def inputAt {s : Shape} (xs : LogicalInputs s) (r : Fin s.m) (k : Nat) : Word32 :=
  if h : k<s.k then xs.input r ⟨k,h⟩ else 0

def packedAt {s : Shape} (xs : LogicalInputs s) (c : Fin s.n) (q : Nat) : Word32 :=
  if h : q<s.k/4 then BitVec.ofNat 32 (xs.packed c ⟨q,h⟩) else 0
end
end VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.PublicIO

namespace VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.PublicIO
open VeriTile.Triton FP.Structural
noncomputable section
open VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001 TileModel
def logicalIO (cast : Word32 → Byte) (rest : Algebra Word32) (layout : Layout) (s : Shape) (c : Fin 15) : LogicalProgramIO Word32 Word32 (LogicalInputs s) (Fin s.m → Fin s.n → Int) where
  core := matmulIO cast rest layout s c
  inputs := logicalInputs layout s
  outputs := logicalOutputs layout s

def logicalDomain (layout : Layout) (s : Shape) (xs : LogicalInputs s) : Prop :=
  Domain layout s ((logicalInputs layout s).symm xs)
end
end VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.PublicIO
