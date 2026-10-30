import VeriTile.Triton.Float.LogicalProgramIO
import bench.tritonbench_g.sgmv_expand_slice.Proofs.SourceGridProgram
import VeriTile.Triton.Float.ReadProgramIO

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.PublicIO
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

/-- The selected source uses FP32 input, FP16 weights and FP32 output.
Metadata keeps the signed integer channel of its actual source ABI. -/
abbrev inputChannels (p : KernelArgs) (shape : Shape) (segments : List Segment) : List IOChannel :=
  [⟨.fp32,[shape.tokens,shape.inner],p.input⟩,
   ⟨.fp16,[shape.adapters,shape.columns,shape.inner],p.weight⟩,
   ⟨.real,[shape.outputRows,shape.outputColumns],p.output⟩,
   ⟨.int,[segments.length],p.starts⟩,
   ⟨.int,[segments.length],p.lengths⟩,
   ⟨.int,[segments.length],p.adapters⟩]

abbrev outputChannels (p : KernelArgs) (shape : Shape) : List IOChannel :=
  [⟨.real,[shape.outputRows,shape.outputColumns],p.output⟩]

def inputRead (p : KernelArgs) (shape : Shape) (segments : List Segment)
    (i : Fin (inputChannels p shape segments).length) :
    ChannelValues α (inputChannels p shape segments) →
      TileIndex (inputChannels p shape segments)[i].shape → Nat :=
  match i with
  | ⟨0,_⟩ => fun _ j => j.1.val*p.inputRowStride+j.2.1.val*p.inputInnerStride
  | ⟨1,_⟩ => fun _ j => p.adapterStride*j.1.val+j.2.2.1.val*p.weightInnerStride+
      j.2.1.val*p.weightColumnStride
  | ⟨2,_⟩ => fun _ j => j.1.val*p.outputRowStride+j.2.1.val*p.outputColumnStride
  | ⟨3,_⟩ | ⟨4,_⟩ | ⟨5,_⟩ => fun _ j => j.1.val
  | ⟨_+6,h⟩ => False.elim (by simpa using h)

def inputMask (p : KernelArgs) (shape : Shape) (segments : List Segment)
    (i : Fin (inputChannels p shape segments).length) :
    ChannelValues α (inputChannels p shape segments) →
      TileIndex (inputChannels p shape segments)[i].shape → Prop :=
  match i with
  | ⟨0,_⟩ | ⟨1,_⟩ | ⟨2,_⟩ => fun _ _ => True
  | ⟨3,_⟩ | ⟨4,_⟩ | ⟨5,_⟩ => fun _ j => (segments[j.1]).active
  | ⟨_+6,h⟩ => False.elim (by simpa using h)

def outputWrite (p : KernelArgs) (shape : Shape) (segments : List Segment)
    (o : Fin (outputChannels p shape).length) :
    ChannelValues α (inputChannels p shape segments) →
      TileIndex (outputChannels p shape)[o].shape → Nat :=
  match o with
  | ⟨0,_⟩ => fun _ j => j.1.val*p.outputRowStride+j.2.1.val*p.outputColumnStride
  | ⟨_+1,h⟩ => False.elim (by simpa using h)

def outputMask (p : KernelArgs) (shape : Shape) (segments : List Segment)
    (o : Fin (outputChannels p shape).length) :
    ChannelValues α (inputChannels p shape segments) →
      TileIndex (outputChannels p shape)[o].shape → Prop :=
  match o with
  | ⟨0,_⟩ => fun _ j => ∃ segment ∈ segments, ActiveCell shape segment j.1.val j.2.1.val
  | ⟨_+1,h⟩ => False.elim (by simpa using h)

def invocation (M : Algebra α) (abi : SourceABI) (p : KernelArgs) (j : Job) : Invocation α where
  kernel := kernel abi {p with batch := j.batch}
  backend := M
  pids := fun ax => if ax=0 then j.pid else 0
  numPids := some (fun ax => if ax=0 then j.count else 1)

/-- Actual ordered source jobs; result reads include unchanged output entries,
whereas only active segment cells are writable. No private scratch is used. -/
abbrev io (M : Algebra α) (abi : SourceABI) (p : KernelArgs) (shape : Shape)
    (segments : List Segment) : ReadProgramIO α α where
  inputs := inputChannels p shape segments
  outputs := outputChannels p shape
  scratch := []
  buffers := [p.input,p.weight,p.output,p.starts,p.lengths,p.adapters]
  read := inputRead p shape segments
  inputMask := inputMask p shape segments
  write := outputWrite p shape segments
  outputMask := outputMask p shape segments
  scratchWrite := fun k => k.elim0
  scratchMask := fun k => k.elim0
  schedule := fun _ => (jobs shape.columns segments).map (invocation M abi p)
  encode := id
  observe := fun _ _ => id
  resultRead := outputWrite p shape segments
  resultMask := fun _ _ _ => True

/-- Skipped segments impose no values on unused signed metadata entries. -/
def metadataDomain (p : KernelArgs) (shape : Shape) (segments : List Segment)
    (xs : ChannelValues α (inputChannels p shape segments)) : Prop :=
  ∀ j : Fin segments.length, (segments[j]).active →
    xs ⟨3,by simp⟩ (j,PUnit.unit) = ((segments[j]).start : Int) ∧
    xs ⟨4,by simp⟩ (j,PUnit.unit) = ((segments[j]).length : Int) ∧
    xs ⟨5,by simp⟩ (j,PUnit.unit) = ((segments[j]).adapter.getD 0 : Int)

def matrixAt [Inhabited α] {rows cols : Nat}
    (v : Fin rows → Fin cols → α) (r c : Nat) : α :=
  if hr : r<rows then if hc : c<cols then v ⟨r,hr⟩ ⟨c,hc⟩ else default else default

def inputValues [Inhabited α] {p : KernelArgs} {shape : Shape} {segments : List Segment}
    (xs : ChannelValues α (inputChannels p shape segments)) : Nat → Nat → α :=
  matrixAt (fun r k => xs ⟨0,by simp⟩ (r,k,PUnit.unit))

def weightValues [Inhabited α] {p : KernelArgs} {shape : Shape} {segments : List Segment}
    (xs : ChannelValues α (inputChannels p shape segments)) (a c k : Nat) : α :=
  if ha : a<shape.adapters then
    matrixAt (fun c k => xs ⟨1,by simp⟩ (⟨a,ha⟩,c,k,PUnit.unit)) c k else default

def entryValues [Inhabited α] {p : KernelArgs} {shape : Shape} {segments : List Segment}
    (xs : ChannelValues α (inputChannels p shape segments)) : Nat → Nat → α :=
  matrixAt (fun r c => xs ⟨2,by simp⟩ (r,c,PUnit.unit))

def resultValues (p : KernelArgs) (shape : Shape) (f : Nat → Nat → α) :
    ChannelValues α (outputChannels p shape) := fun o =>
  match o with
  | ⟨0,_⟩ => fun j => f j.1.val j.2.1.val
  | ⟨_+1,h⟩ => False.elim (by simpa using h)


/-- Every original input value is retained, including inactive segment metadata
and the complete entry-output tensor needed for ordered overlap semantics. -/
structure LogicalInputs (α : Type) (shape : Shape) (segments : List Segment) where
  input : Fin shape.tokens → Fin shape.inner → α
  weight : Fin shape.adapters → Fin shape.columns → Fin shape.inner → α
  entryOutput : Fin shape.outputRows → Fin shape.outputColumns → α
  starts : Fin segments.length → Int
  lengths : Fin segments.length → Int
  adapters : Fin segments.length → Int

def logicalInputs {α : Type} (p : KernelArgs) (shape : Shape) (segments : List Segment) :
    ChannelValues α (inputChannels p shape segments) ≃ LogicalInputs α shape segments where
  toFun := fun xs => {
    input := fun r k => xs (0 : Fin 6) (r,k,())
    weight := fun a c k => xs (1 : Fin 6) (a,c,k,())
    entryOutput := fun r c => xs (2 : Fin 6) (r,c,())
    starts := fun j => xs (3 : Fin 6) (j,())
    lengths := fun j => xs (4 : Fin 6) (j,())
    adapters := fun j => xs (5 : Fin 6) (j,()) }
  invFun := fun xs => Fin.cases (fun j => xs.input j.1 j.2.1)
    (Fin.cases (fun j => xs.weight j.1 j.2.1 j.2.2.1)
      (Fin.cases (fun j => xs.entryOutput j.1 j.2.1)
        (Fin.cases (fun j => xs.starts j.1) (Fin.cases (fun j => xs.lengths j.1)
          (Fin.cases (fun j => xs.adapters j.1) (fun i => i.elim0))))))
  left_inv := by intro xs; funext port j; fin_cases port <;> rfl
  right_inv := by rintro ⟨x,w,out,s,l,a⟩; rfl

def logicalOutputs {α : Type} (p : KernelArgs) (shape : Shape) :
    ChannelValues α (outputChannels p shape) ≃ (Fin shape.outputRows → Fin shape.outputColumns → α) :=
  (ChannelValues.singleEquiv α _).trans (Values.matrixEquiv α .real _ _)

def LogicalInputs.inputAt {α : Type} [Inhabited α] {shape : Shape} {segments : List Segment}
    (xs : LogicalInputs α shape segments) := matrixAt xs.input

def LogicalInputs.weightAt {α : Type} [Inhabited α] {shape : Shape} {segments : List Segment}
    (xs : LogicalInputs α shape segments) (a c k : Nat) : α :=
  if ha : a<shape.adapters then matrixAt (xs.weight ⟨a,ha⟩) c k else default

def LogicalInputs.entryAt {α : Type} [Inhabited α] {shape : Shape} {segments : List Segment}
    (xs : LogicalInputs α shape segments) := matrixAt xs.entryOutput

def LogicalMetadataDomain {α : Type} {shape : Shape} {segments : List Segment}
    (xs : LogicalInputs α shape segments) : Prop :=
  ∀ j : Fin segments.length, (segments[j]).active →
    xs.starts j=((segments[j]).start : Int) ∧ xs.lengths j=((segments[j]).length : Int) ∧
      xs.adapters j=((segments[j]).adapter.getD 0 : Int)

def logicalIO {α : Type} [Inhabited α] (M : Algebra α) (abi : SourceABI)
    (p : KernelArgs) (shape : Shape) (segments : List Segment) :
    LogicalDomainIO (ChannelValues α (inputChannels p shape segments))
      (ChannelValues α (outputChannels p shape)) (LogicalInputs α shape segments)
      (Fin shape.outputRows → Fin shape.outputColumns → α)
      (io M abi p shape segments).ImplementsOn where
  inputs := logicalInputs p shape segments
  outputs := logicalOutputs p shape

end
end VeriTile.Bench.Repairs.SgmvExpandSlice.PublicIO
