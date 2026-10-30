import VeriTile.Triton.Float.RelationalProgramIO
import VeriTile.Triton.Memory.LogicalIO
import bench.tritonbench_g.max_reduction.Proofs.Source
import bench.tritonbench_g.max_reduction.Proofs.Mathematical_Candidate
import VeriTile.Triton.Float.ProgramEquivalence

namespace VeriTile.Bench.Repairs.MaxReduction.PublicIO
open VeriTile.Triton FP.Structural
noncomputable section

/-- Existing Algorithm/flat-memory mathematical stage-one interface, now
listing the selected shared source and all active input/output fields. -/
def stage1Real (inp out : RegionName) (N B : Nat) : MaskedTileKernelIO₁ where
  kernel := (stage1 inp out N B).eraseDType
  projection := VeriTile.Bench.TritonBenchG.MaxReduction.stage1_supported .fp32 Bool.false inp out N B
  inp := inp
  out := out
  shape := [B]
  read := fun pid j => pid*B+j.1.val
  write := fun pid _ => pid
  mask := fun pid j => pid*B+j.1.val<N
  writeMask := fun _ j => j.1.val=0

def stage2Real (inp out : RegionName) (N B : Nat) : MaskedTileKernelIO₁ where
  kernel := (stage2 inp out N B).eraseDType
  projection := VeriTile.Bench.TritonBenchG.MaxReduction.stage2_supported .fp32 Bool.false inp out N B
  inp := inp
  out := out
  shape := [B]
  read := fun _ j => j.1.val
  write := fun _ _ => 0
  mask := fun _ j => j.1.val<N
  writeMask := fun _ j => j.1.val=0

def stageSource (second candidate : Bool) (inp out : RegionName) (N B : Nat) : ComputeKernel :=
  if second then (if candidate then stage2Complement inp out N B else stage2 inp out N B)
  else (if candidate then stage1Commuted inp out N B else stage1 inp out N B)
def stageStart (second : Bool) (B pid : Nat) : Nat := if second then 0 else pid*B
def stageOutput (second : Bool) (pid : Nat) : Nat := if second then 0 else pid

def stageLaunch {α : Type} (e : Source.Engine α) (second candidate : Bool)
    (inp out : RegionName) (N B pid : Nat) : Invocation α where
  kernel := stageSource second candidate inp out N B
  backend := e.algebra
  pids := fun ax => if ax=0 then pid else 0

/-- Source FP32 loads/stores use the existing structural .real carrier;
primitive reduction values remain arbitrary. The singleton schedule describes
one program at every possible pid, matching the original stage theorem scope. -/
def stageIO {α : Type} (e : Source.Engine α) (second : Bool)
    (inp out : RegionName) (N B pid : Nat) : ProgramIO α α where
  inputs := [⟨.real,[B],inp⟩]
  outputs := [⟨.real,[],out⟩]
  buffers := [inp,out]
  scratch := []
  read := Fin.cases (fun _ j => stageStart second B pid+j.1.val) (fun i => i.elim0)
  inputMask := Fin.cases (fun _ j => stageStart second B pid+j.1.val<N) (fun i => i.elim0)
  write := Fin.cases (fun _ _ => stageOutput second pid) (fun i => i.elim0)
  outputMask := fun _ _ _ => True
  scratchWrite := fun i => i.elim0
  scratchMask := fun i => i.elim0
  schedule := fun _ => [stageLaunch e second Bool.false inp out N B pid]
  encode := id
  observe := fun _ _ => id

def stageCandidate {α : Type} (e : Source.Engine α) (second : Bool)
    (inp out : RegionName) (N B pid : Nat) : (stageIO e second inp out N B pid).Variant where
  schedule := fun _ => [stageLaunch e second Bool.true inp out N B pid]
  observe := fun _ _ => id

def dimensionLaunch {α : Type} (e : Source.Engine α) (candidate : Bool)
    (inp ov oi : RegionName) (M N K BM BN p k : Nat) : Invocation α where
  kernel := if candidate then dimensionCommuted inp ov oi M N K BM BN else dimension inp ov oi M N K BM BN
  backend := e.algebra
  pids := fun ax => if ax=0 then p else if ax=1 then k else 0

/-- Actual strided dimension window, independently masked input and output
rows, FP32 value and signed-INT64 index streams, and no writable scratch.
Arbitrary region aliases remain representable for the equivalence endpoint. -/
def dimensionIO {α β : Type} (e : Source.Engine β) (encode : α→β) (observe : β→α)
    (inp ov oi : RegionName) (M N K BM BN p k : Nat) : ProgramIO α β where
  inputs := [⟨.real,[BM,BN],inp⟩]
  outputs := [⟨.real,[BM],ov⟩,⟨.int,[BM],oi⟩]
  buffers := [inp,ov,oi]
  scratch := []
  read := Fin.cases (fun _ j => (p*BM+j.1.val)*N*K+j.2.1.val*K+k) (fun i => i.elim0)
  inputMask := Fin.cases (fun _ j => p*BM+j.1.val<M ∧ j.2.1.val<N) (fun i => i.elim0)
  write := Fin.cases (fun _ j => (p*BM+j.1.val)*K+k)
    (Fin.cases (fun _ j => (p*BM+j.1.val)*K+k) (fun i => i.elim0))
  outputMask := Fin.cases (fun _ j => p*BM+j.1.val<M)
    (Fin.cases (fun _ j => p*BM+j.1.val<M) (fun i => i.elim0))
  scratchWrite := fun i => i.elim0
  scratchMask := fun i => i.elim0
  schedule := fun _ => [dimensionLaunch e Bool.false inp ov oi M N K BM BN p k]
  encode := encode
  observe := fun _ _ => observe

def dimensionCandidate {α : Type} (e : Source.Engine α)
    (inp ov oi : RegionName) (M N K BM BN p k : Nat) :
    (dimensionIO e id id inp ov oi M N K BM BN p k).Variant where
  schedule := fun _ => [dimensionLaunch e Bool.true inp ov oi M N K BM BN p k]
  observe := fun _ _ => id

end
end VeriTile.Bench.Repairs.MaxReduction.PublicIO

namespace VeriTile.Bench.Repairs.MaxReduction.PublicIO
open VeriTile.Triton FP.Structural
noncomputable section

/-- The complete original block vector, without the singleton tile tail. -/
def blockVector (B : Nat) : (TileIndex [B] → ℝ) ≃ (Fin B → ℝ) where
  toFun := fun xs i => xs (i,PUnit.unit)
  invFun := fun xs i => xs i.1
  left_inv := by intro xs; funext i; rcases i with ⟨lane,⟨⟩⟩; rfl
  right_inv := fun _ => rfl

/-- Full padded input tile in row/column coordinates. Logical N remains separate. -/
def dimensionInputs (inp : RegionName) (BM BN : Nat) :
    ChannelValues ℝ [⟨.real,[BM,BN],inp⟩] ≃ (Fin BM → Fin BN → ℝ) where
  toFun := fun xs row col => xs 0 (row,col,PUnit.unit)
  invFun := fun xs => Fin.cases (fun i => xs i.1 i.2.1) (fun port => Fin.elim0 port)
  left_inv := by
    intro xs
    funext port i
    fin_cases port
    rcases i with ⟨row,col,⟨⟩⟩
    rfl
  right_inv := fun _ => rfl

/-- Both separately typed result arrays, with no channel-number selection. -/
def dimensionOutputs (ov oi : RegionName) (BM : Nat) :
    ChannelValues ℝ [⟨.real,[BM],ov⟩,⟨.int,[BM],oi⟩] ≃
      ((Fin BM → ℝ) × (Fin BM → Int)) where
  toFun := fun ys => (fun i => ys 0 (i,PUnit.unit), fun i => ys 1 (i,PUnit.unit))
  invFun := fun vi => Fin.cases (fun i => vi.1 i.1)
    (Fin.cases (fun i => vi.2 i.1) (fun port => Fin.elim0 port))
  left_inv := by
    intro ys
    funext port i
    fin_cases port <;> rcases i with ⟨row,⟨⟩⟩ <;> rfl
  right_inv := fun _ => rfl

end
end VeriTile.Bench.Repairs.MaxReduction.PublicIO

namespace VeriTile.Bench.Repairs.MaxReduction.PublicIO
open VeriTile.Triton FP.Structural
noncomputable section

/-- All PIDs and the complete original masked block vector. -/
def logicalStage1IO (inp out : RegionName) (N B : Nat) :
    LogicalIO (Nat × (TileIndex [B] → ℝ)) (TileIndex [B] → ℝ)
      (Nat × (Fin B → ℝ)) (Fin B → ℝ)
      (fun f => (stage1Real inp out N B).Implements (fun pid xs => f (pid,xs))) where
  inputs := Equiv.prodCongr (Equiv.refl Nat) (blockVector B)
  outputs := blockVector B

def logicalStage2IO (inp out : RegionName) (N B : Nat) :
    LogicalIO (Nat × (TileIndex [B] → ℝ)) (TileIndex [B] → ℝ)
      (Nat × (Fin B → ℝ)) (Fin B → ℝ)
      (fun f => (stage2Real inp out N B).Implements (fun pid xs => f (pid,xs))) where
  inputs := Equiv.prodCongr (Equiv.refl Nat) (blockVector B)
  outputs := blockVector B

/-- Natural logical-axis reads use the old zero extension past the padded tile. -/
def dimensionInput {BM BN : Nat} (xs : Fin BM → Fin BN → ℝ) (row : Fin BM) (col : Nat) : ℝ :=
  if h : col<BN then xs row ⟨col,h⟩ else 0

/-- Both separately typed observations and the full original relational judgment. -/
def logicalDimensionIO (e : Source.Engine (WithBot ℝ)) (observe : WithBot ℝ → ℝ)
    (inp ov oi : RegionName) (M N K BM BN p k : Nat) :
    LogicalRelationIO
      (ChannelValues ℝ [⟨.real,[BM,BN],inp⟩])
      (ChannelValues ℝ [⟨.real,[BM],ov⟩,⟨.int,[BM],oi⟩])
      (Fin BM → Fin BN → ℝ) ((Fin BM → ℝ) × (Fin BM → Int))
      (dimensionIO e (fun r : ℝ => (r : WithBot ℝ)) observe inp ov oi M N K BM BN p k).Satisfies where
  inputs := dimensionInputs inp BM BN
  outputs := dimensionOutputs ov oi BM

end
end VeriTile.Bench.Repairs.MaxReduction.PublicIO
