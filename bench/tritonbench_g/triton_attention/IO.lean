import bench.tritonbench_g.triton_attention.Proofs.NativeGraphMemory
import bench.tritonbench_g.triton_attention.Proofs.RealBackend
import bench.tritonbench_g.triton_attention.Proofs.ScalarBinding
import VeriTile.Triton.Float.MaterializedProgramIO
import VeriTile.Triton.Memory.LogicalIO
import Mathlib.Tactic.FinCases

namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.PublicIO
open VeriTile Triton FP.Structural Source
open VeriTile.Bench.Repairs.TritonAttention.Whole001
noncomputable section

abbrev Index (c : Shape) := TileIndex [c.groups,c.rows,c.dim]
def coordinate (c : Shape) (i : Index c) : Coord c := (i.1,i.2.1,i.2.2.1)
def position (c : Shape) (i : Index c) : Nat := offset c (coordinate c i)

@[reducible] def inputs (c : Shape) : List IOChannel :=
  [⟨.fp32,[c.groups,c.rows,c.dim],"Q"⟩,
   ⟨.fp32,[c.groups,c.rows,c.dim],"K"⟩,
   ⟨.fp32,[c.groups,c.rows,c.dim],"V"⟩,
   ⟨.fp32,[c.groups,c.rows,c.dim],"DO"⟩,
   ⟨.fp32,[c.groups,c.rows,c.dim],"DQ"⟩]
@[reducible] def outputs (c : Shape) : List IOChannel :=
  [⟨.fp32,[c.groups,c.rows,c.dim],"Out"⟩,
   ⟨.fp32,[c.groups,c.rows,c.dim],"DQ"⟩,
   ⟨.fp32,[c.groups,c.rows,c.dim],"DK"⟩,
   ⟨.fp32,[c.groups,c.rows,c.dim],"DV"⟩]
@[reducible] def scratch (c : Shape) : List IOChannel :=
  [⟨.fp32,[c.groups,c.rows],"L"⟩,
   ⟨.fp32,[c.groups,c.rows],"M"⟩,
   ⟨.fp32,[c.groups,c.rows,c.dim],"NewDO"⟩,
   ⟨.fp32,[c.groups,c.rows],"Delta"⟩]

abbrev Inputs (α : Type) (c : Shape) := ChannelValues α (inputs c)
abbrev Outputs (α : Type) (c : Shape) := ChannelValues α (outputs c)

/-- Source-required initialized in-place DQ; all four data inputs are arbitrary. -/
def ZeroDQ {α : Type} (zero : α) (c : Shape) (xs : Inputs α c) : Prop :=
  ∀ i,xs 4 i=zero

def inputOffset (c : Shape) (port : Fin (inputs c).length) :
    TileIndex (inputs c)[port].shape → Nat := match port with
  | ⟨0,_⟩ | ⟨1,_⟩ | ⟨2,_⟩ | ⟨3,_⟩ | ⟨4,_⟩ => position c
def outputOffset (c : Shape) (port : Fin (outputs c).length) :
    TileIndex (outputs c)[port].shape → Nat := match port with
  | ⟨0,_⟩ | ⟨1,_⟩ | ⟨2,_⟩ | ⟨3,_⟩ => position c
def scratchOffset (c : Shape) (port : Fin (scratch c).length) :
    TileIndex (scratch c)[port].shape → Nat := match port with
  | ⟨0,_⟩ | ⟨1,_⟩ | ⟨3,_⟩ => fun i=>rowOffset c i.1 i.2.1
  | ⟨2,_⟩ => position c

/-- Three actual readable quotations with the original ABI and fresh CTA-local
state. The memory itself is supplied by the preceding materialized launch. -/
def sourceInvocation {α : Type} (rest : Algebra α) (P : Primitives α)
    (kind : Native.Kind) (c : Shape) (scale : ℝ) (pid : Grid.Program) : Invocation α where
  kernel := Native.kernelFor c scale kind
  backend := backend rest P
  arguments := []
  clearRegisters := Bool.true
  pids := Grid.ids pid
  numPids := some (fun ax=>if ax=0 then (Native.gridFor c kind).1 else
    if ax=1 then (Native.gridFor c kind).2 else 1)

def realInvocation (kind : Native.Kind) (c : Shape) (scale : ℝ)
    (pid : Grid.Program) : MaterializedInvocation Erased.R Erased.R where
  source := sourceInvocation Erased.algebra Erased.model kind c scale pid
  encode := id
  decodeAt := fun _ _=>id
  decode_encode := fun _ _ _=>rfl

/-- Each store selects its original scalar replica and lowers its local graph
before the next CTA. This is the existing native interpretation of the source. -/
def nativeInvocation (one : Bool) (kind : Native.Kind) (c : Shape) (scale : ℝ)
    (owners : TritonAttention.Whole001.Native.Owners) (pid : Grid.Program) :
    MaterializedInvocation Native.Graph Native.Expr where
  source := sourceInvocation (Native.algebraScalar one kind) (Native.modelScalar one kind) kind c scale pid
  encode := fun g _=>Native.Stage.ready g
  decodeAt := fun r o e=>Native.lowerScalar one kind (e (Native.storeOwner c owners r o))
  decode_encode := fun _ _ _=>rfl

def layout {α β γ : Type} (c : Shape) (encode : α→β)
    (observe : β→α) (schedule : List (MaterializedInvocation β γ)) : MaterializedProgramIO α β γ where
  inputs := inputs c
  outputs := outputs c
  scratch := scratch c
  buffers := ["Q","K","V","DO","Out","DQ","DK","DV","L","M","NewDO","Delta"]
  read := fun port _=>inputOffset c port
  inputMask := fun _ _ _=>True
  write := fun port _=>outputOffset c port
  outputMask := fun _ _ _=>True
  scratchWrite := fun port _=>scratchOffset c port
  scratchMask := fun _ _ _=>True
  schedule := fun _=>schedule
  encode := encode
  observe := fun _ _=>observe

def realIO (c : Shape) (scale : ℝ) (fw pw bw : List Grid.Program) :
    MaterializedProgramIO ℝ Erased.R Erased.R :=
  layout c some (WithBot.unbotD 0)
    (fw.map (realInvocation .forward c scale) ++ pw.map (realInvocation .preprocess c scale) ++
      bw.map (realInvocation .backward c scale))

def nativeIO (scalar : Checked.ScalarABI) (c : Shape)
    (owners : TritonAttention.Whole001.Native.Owners) (I : Native.Prior.Instructions)
    (literal : ℝ→Native.Word) (fw pw bw : List Grid.Program) :
    MaterializedProgramIO Native.Word Native.Graph Native.Expr :=
  let one := Native.constexprScale scalar
  let scale := scalar.project scalar.raw
  layout c Native.Graph.word (Native.Graph.eval I literal)
    (fw.map (nativeInvocation one .forward c scale owners) ++
      pw.map (nativeInvocation one .preprocess c scale owners) ++
      bw.map (nativeInvocation one .backward c scale owners))

/-- The original scalar ABI binding, including runtime class specialization. -/
def SourceScale (scalar : Checked.ScalarABI) (I : Native.Prior.Instructions)
    (literal : ℝ→Native.Word) : Prop :=
  ∀ kind owner,(Native.lowerScalar (Native.constexprScale scalar) kind
    ((Native.modelScalar (Native.constexprScale scalar) kind).literal (some .fp32) .real
      (scalar.project scalar.raw) owner)).eval I literal=scalar.word

abbrev LogicalTensor (c : Shape) := Fin c.groups → Fin c.rows → Fin c.dim → ℝ
structure LogicalInputs (c : Shape) where
  query : LogicalTensor c
  key : LogicalTensor c
  value : LogicalTensor c
  outputCotangent : LogicalTensor c
  initialDQ : LogicalTensor c
structure LogicalOutputs (c : Shape) where
  output : LogicalTensor c
  dq : LogicalTensor c
  dk : LogicalTensor c
  dv : LogicalTensor c

def logicalInputs (c : Shape) : Inputs ℝ c ≃ LogicalInputs c where
  toFun xs := ⟨fun h i d => xs 0 (h,i,d,()),fun h i d => xs 1 (h,i,d,()),
    fun h i d => xs 2 (h,i,d,()),fun h i d => xs 3 (h,i,d,()),fun h i d => xs 4 (h,i,d,())⟩
  invFun xs := fun port => match port with
    | ⟨0,_⟩ => fun i => xs.query i.1 i.2.1 i.2.2.1
    | ⟨1,_⟩ => fun i => xs.key i.1 i.2.1 i.2.2.1
    | ⟨2,_⟩ => fun i => xs.value i.1 i.2.1 i.2.2.1
    | ⟨3,_⟩ => fun i => xs.outputCotangent i.1 i.2.1 i.2.2.1
    | ⟨4,_⟩ => fun i => xs.initialDQ i.1 i.2.1 i.2.2.1
  left_inv := by intro xs; funext port i; fin_cases port <;> rcases i with ⟨h,r,d,u⟩ <;> cases u <;> rfl
  right_inv := by intro xs; cases xs; rfl

def logicalOutputs (c : Shape) : Outputs ℝ c ≃ LogicalOutputs c where
  toFun xs := ⟨fun h i d => xs 0 (h,i,d,()),fun h i d => xs 1 (h,i,d,()),
    fun h i d => xs 2 (h,i,d,()),fun h i d => xs 3 (h,i,d,())⟩
  invFun xs := fun port => match port with
    | ⟨0,_⟩ => fun i => xs.output i.1 i.2.1 i.2.2.1
    | ⟨1,_⟩ => fun i => xs.dq i.1 i.2.1 i.2.2.1
    | ⟨2,_⟩ => fun i => xs.dk i.1 i.2.1 i.2.2.1
    | ⟨3,_⟩ => fun i => xs.dv i.1 i.2.1 i.2.2.1
  left_inv := by intro xs; funext port i; fin_cases port <;> rcases i with ⟨h,r,d,u⟩ <;> cases u <;> rfl
  right_inv := by intro xs; cases xs; rfl

/-- Materialized cell binding, initialized DQ and all three grids remain in
this fixed judgment; only tensor/channel coordinates are transported. -/
def logicalRealIO (c : Shape) (scale : ℝ) (fw pw bw : List Grid.Program) :
    LogicalDomainIO (Inputs ℝ c) (Outputs ℝ c) (LogicalInputs c) (LogicalOutputs c)
      (realIO c scale fw pw bw).ImplementsOnCells where
  inputs := logicalInputs c
  outputs := logicalOutputs c

def logicalZeroDQ (c : Shape) (xs : LogicalInputs c) : Prop := ∀ h i d,xs.initialDQ h i d=0

end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.PublicIO
