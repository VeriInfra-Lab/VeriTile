import VeriTile.Triton.Float.LogicalProgramIO
import Mathlib.Logic.Equiv.Fin.Basic
import bench.tritonbench_g.fused_rotary_embedding.InvocationIO
import bench.tritonbench_g.fused_rotary_embedding.Proofs.Logical

namespace VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.PublicIO
open VeriTile.Triton FP.Structural FusedRotaryEmbedding
noncomputable section

abbrev inputChannels (s : Shape) : List IOChannel :=
  [⟨.real,[s.tokens*s.headsQ*s.width],"q"⟩,
   ⟨.real,[s.tokens*s.headsK*s.width],"k"⟩,
   ⟨.real,[s.tokens*s.headsK*s.width],"v"⟩,
   ⟨.real,[s.tokens*s.cosWidth],"cos"⟩,
   ⟨.real,[s.tokens*s.sinWidth],"sin"⟩,
   ⟨.int,[s.tokens],"context_lengths"⟩,
   ⟨.int,[s.tokens*s.tableWidth],"BLOCK_TABLES"⟩]

abbrev outputChannels (s : Shape) : List IOChannel :=
  [⟨.real,[s.tokens,s.headsQ,s.width],"q"⟩,
   ⟨.real,[s.tokens,s.headsK,s.width],"k_cache"⟩,
   ⟨.real,[s.tokens,s.headsK,s.width],"v_cache"⟩]

def inputRead (s : Shape) (i : Fin (inputChannels s).length) :
    ChannelValues α (inputChannels s) → TileIndex (inputChannels s)[i].shape → Nat :=
  match i with
  | ⟨0,_⟩ | ⟨1,_⟩ | ⟨2,_⟩ | ⟨3,_⟩ | ⟨4,_⟩ | ⟨5,_⟩ | ⟨6,_⟩ => fun _ j => j.1.val
  | ⟨_+7,h⟩ => False.elim (by simpa using h)

/-- Only the table element actually selected by each token is constrained by
the indirect metadata domain. Unused table entries remain unrestricted. -/
def metadataDomain (c : Config) (xs : ChannelValues α (inputChannels c.shape)) : Prop :=
  (∀ t : Fin c.shape.tokens, xs ⟨5,by simp⟩ (t,PUnit.unit) = (c.metadata.lengths t.val : Int)) ∧
  (∀ t (ht : t<c.shape.tokens) (ha : t*c.shape.tableWidth+c.past t/c.shape.blockSize<c.shape.tokens*c.shape.tableWidth),
    xs ⟨6,by simp⟩ (⟨t*c.shape.tableWidth+c.past t/c.shape.blockSize,ha⟩,PUnit.unit) = (c.block t : Int))

def outputWrite (c : Config) (o : Fin (outputChannels c.shape).length) :
    ChannelValues α (inputChannels c.shape) → TileIndex (outputChannels c.shape)[o].shape → Nat :=
  match o with
  | ⟨0,_⟩ => fun _ j => logicalOutput c .q j.1.val j.2.1.val j.2.2.1.val
  | ⟨1,_⟩ => fun _ j => logicalOutput c .k j.1.val j.2.1.val j.2.2.1.val
  | ⟨2,_⟩ => fun _ j => logicalOutput c .v j.1.val j.2.1.val j.2.2.1.val
  | ⟨_+3,h⟩ => False.elim (by simpa using h)

def observeOutput (s : Shape) (observe : Kind → β → α) (o : Fin (outputChannels s).length) :
    TileIndex (outputChannels s)[o].shape → β → α :=
  match o with
  | ⟨0,_⟩ => fun _ => observe .q
  | ⟨1,_⟩ => fun _ => observe .k
  | ⟨2,_⟩ => fun _ => observe .v
  | ⟨_+3,h⟩ => False.elim (by simpa using h)

abbrev io (rest : Algebra β) (ops : Ops β) (encode : α → β) (observe : Kind → β → α)
    (c : Config) : ProgramIO α β where
  inputs := inputChannels c.shape
  outputs := outputChannels c.shape
  scratch := []
  buffers := ["q","k","v","cos","sin","context_lengths","BLOCK_TABLES","k_cache","v_cache"]
  read := inputRead c.shape
  inputMask := fun _ _ _ => True
  write := outputWrite c
  outputMask := fun _ _ _ => True
  scratchWrite := fun k => k.elim0
  scratchMask := fun k => k.elim0
  schedule := fun _ => (programs c.shape).map (Source.invocation rest ops c)
  encode := encode
  observe := observeOutput c.shape observe

def vectorAt [Inhabited α] {n : Nat} (v : Values α .real [n]) (a : Nat) : α :=
  if h : a<n then v (⟨a,h⟩,PUnit.unit) else default

def inputMemory [Inhabited α] (s : Shape) (xs : ChannelValues α (inputChannels s)) : Memory α :=
  fun r a => match r with
  | .q => vectorAt (xs ⟨0,by simp⟩) a
  | .k => vectorAt (xs ⟨1,by simp⟩) a
  | .v => vectorAt (xs ⟨2,by simp⟩) a
  | .cos => vectorAt (xs ⟨3,by simp⟩) a
  | .sin => vectorAt (xs ⟨4,by simp⟩) a
  | _ => default

def outputFormula (s : Shape) (f : Kind → Nat → Nat → Nat → α) :
    ChannelValues α (outputChannels s) := fun o =>
  match o with
  | ⟨0,_⟩ => fun j => f .q j.1.val j.2.1.val j.2.2.1.val
  | ⟨1,_⟩ => fun j => f .k j.1.val j.2.1.val j.2.2.1.val
  | ⟨2,_⟩ => fun j => f .v j.1.val j.2.1.val j.2.2.1.val
  | ⟨_+3,h⟩ => False.elim (by simpa using h)

end
end VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.PublicIO

namespace VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.PublicIO
open VeriTile.Triton FP.Structural FusedRotaryEmbedding
noncomputable section

def flatMatrix (α : Type) (dtype : TileDType) (m n : Nat) :
    Values α dtype [m*n] ≃ (Fin m → Fin n → Value α dtype) where
  toFun := fun xs r c => xs (finProdFinEquiv (r,c),())
  invFun := fun xs j => xs (finProdFinEquiv.symm j.1).1 (finProdFinEquiv.symm j.1).2
  left_inv := by
    intro xs
    funext j
    change xs (finProdFinEquiv (finProdFinEquiv.symm j.1),()) = xs j
    rw [Equiv.apply_symm_apply]
  right_inv := by
    intro xs
    funext r c
    change (let j := finProdFinEquiv.symm (finProdFinEquiv (r,c)); xs j.1 j.2) = xs r c
    rw [Equiv.symm_apply_apply]

def flatTensor (α : Type) (dtype : TileDType) (l m n : Nat) :
    Values α dtype [l*m*n] ≃ (Fin l → Fin m → Fin n → Value α dtype) where
  toFun := fun xs t h d => xs (finProdFinEquiv (finProdFinEquiv (t,h),d),())
  invFun := fun xs j => let p := finProdFinEquiv.symm j.1
    let q := finProdFinEquiv.symm p.1
    xs q.1 q.2 p.2
  left_inv := by
    intro xs
    funext j
    change xs (finProdFinEquiv (finProdFinEquiv (finProdFinEquiv.symm (finProdFinEquiv.symm j.1).1),
      (finProdFinEquiv.symm j.1).2),()) = xs j
    rw [Equiv.apply_symm_apply,Equiv.apply_symm_apply]
  right_inv := by
    intro xs
    funext t h d
    change (let p := finProdFinEquiv.symm (finProdFinEquiv (finProdFinEquiv (t,h),d));
      let q := finProdFinEquiv.symm p.1; xs q.1 q.2 p.2) = xs t h d
    simp only [Equiv.symm_apply_apply]

structure RotaryInputs (α : Type) (s : Shape) where
  q : Fin s.tokens → Fin s.headsQ → Fin s.width → α
  k : Fin s.tokens → Fin s.headsK → Fin s.width → α
  v : Fin s.tokens → Fin s.headsK → Fin s.width → α
  cos : Fin s.tokens → Fin s.cosWidth → α
  sin : Fin s.tokens → Fin s.sinWidth → α
  lengths : Fin s.tokens → Int
  blockTables : Fin s.tokens → Fin s.tableWidth → Int

def inputCoordinates (α : Type) (s : Shape) : ChannelValues α (inputChannels s) ≃ RotaryInputs α s where
  toFun := fun xs => ⟨(flatTensor α .real s.tokens s.headsQ s.width) (xs (0 : Fin 7)),
    (flatTensor α .real s.tokens s.headsK s.width) (xs (1 : Fin 7)),
    (flatTensor α .real s.tokens s.headsK s.width) (xs (2 : Fin 7)),
    (flatMatrix α .real s.tokens s.cosWidth) (xs (3 : Fin 7)),
    (flatMatrix α .real s.tokens s.sinWidth) (xs (4 : Fin 7)),
    (Values.vectorEquiv α .int s.tokens) (xs (5 : Fin 7)),
    (flatMatrix α .int s.tokens s.tableWidth) (xs (6 : Fin 7))⟩
  invFun := fun xs => (Fin.cases ((flatTensor α .real s.tokens s.headsQ s.width).symm xs.q) (Fin.cases ((flatTensor α .real s.tokens s.headsK s.width).symm xs.k) (Fin.cases ((flatTensor α .real s.tokens s.headsK s.width).symm xs.v) (Fin.cases ((flatMatrix α .real s.tokens s.cosWidth).symm xs.cos) (Fin.cases ((flatMatrix α .real s.tokens s.sinWidth).symm xs.sin) (Fin.cases ((Values.vectorEquiv α .int s.tokens).symm xs.lengths) (Fin.cases ((flatMatrix α .int s.tokens s.tableWidth).symm xs.blockTables) (fun i => Fin.elim0 i))))))))
  left_inv := by
    intro xs
    funext i
    fin_cases i
    · exact (flatTensor α .real s.tokens s.headsQ s.width).symm_apply_apply (xs (0 : Fin 7))
    · exact (flatTensor α .real s.tokens s.headsK s.width).symm_apply_apply (xs (1 : Fin 7))
    · exact (flatTensor α .real s.tokens s.headsK s.width).symm_apply_apply (xs (2 : Fin 7))
    · exact (flatMatrix α .real s.tokens s.cosWidth).symm_apply_apply (xs (3 : Fin 7))
    · exact (flatMatrix α .real s.tokens s.sinWidth).symm_apply_apply (xs (4 : Fin 7))
    · exact (Values.vectorEquiv α .int s.tokens).symm_apply_apply (xs (5 : Fin 7))
    · exact (flatMatrix α .int s.tokens s.tableWidth).symm_apply_apply (xs (6 : Fin 7))
  right_inv := by
    intro xs
    change RotaryInputs.mk ((flatTensor α .real s.tokens s.headsQ s.width) ((flatTensor α .real s.tokens s.headsQ s.width).symm xs.q)) ((flatTensor α .real s.tokens s.headsK s.width) ((flatTensor α .real s.tokens s.headsK s.width).symm xs.k)) ((flatTensor α .real s.tokens s.headsK s.width) ((flatTensor α .real s.tokens s.headsK s.width).symm xs.v)) ((flatMatrix α .real s.tokens s.cosWidth) ((flatMatrix α .real s.tokens s.cosWidth).symm xs.cos)) ((flatMatrix α .real s.tokens s.sinWidth) ((flatMatrix α .real s.tokens s.sinWidth).symm xs.sin)) ((Values.vectorEquiv α .int s.tokens) ((Values.vectorEquiv α .int s.tokens).symm xs.lengths)) ((flatMatrix α .int s.tokens s.tableWidth) ((flatMatrix α .int s.tokens s.tableWidth).symm xs.blockTables)) = xs
    simp only [Equiv.apply_symm_apply]

structure RotaryOutputs (α : Type) (c : Shape) where
  q : Fin (c.tokens) → Fin (c.headsQ) → Fin (c.width) → α
  k : Fin (c.tokens) → Fin (c.headsK) → Fin (c.width) → α
  v : Fin (c.tokens) → Fin (c.headsK) → Fin (c.width) → α

def outputCoordinates (α : Type) (c : Shape) : ChannelValues α (outputChannels c) ≃ RotaryOutputs α c where
  toFun := fun xs => ⟨Values.curryEquiv α .real [c.tokens, c.headsQ, c.width] (xs (0 : Fin 3)),
    Values.curryEquiv α .real [c.tokens, c.headsK, c.width] (xs (1 : Fin 3)),
    Values.curryEquiv α .real [c.tokens, c.headsK, c.width] (xs (2 : Fin 3))⟩
  invFun := fun xs => (Fin.cases ((Values.curryEquiv α .real [c.tokens, c.headsQ, c.width]).symm xs.q) (Fin.cases ((Values.curryEquiv α .real [c.tokens, c.headsK, c.width]).symm xs.k) (Fin.cases ((Values.curryEquiv α .real [c.tokens, c.headsK, c.width]).symm xs.v) (fun i => Fin.elim0 i))))
  left_inv := by
    intro xs
    funext i
    fin_cases i
    · exact (Values.curryEquiv α .real [c.tokens, c.headsQ, c.width]).symm_apply_apply (xs (0 : Fin 3))
    · exact (Values.curryEquiv α .real [c.tokens, c.headsK, c.width]).symm_apply_apply (xs (1 : Fin 3))
    · exact (Values.curryEquiv α .real [c.tokens, c.headsK, c.width]).symm_apply_apply (xs (2 : Fin 3))
  right_inv := by intro xs; cases xs; rfl

/-- Logical rows use exactly the finite coordinate bijection above. The
existing total extension is retained only outside the active logical shape. -/
def tensorInput (s : Shape) (xs : RotaryInputs ℝ s) (kind : Kind) (t h d : Nat) : ℝ :=
  inputMemory s ((inputCoordinates ℝ s).symm xs) kind.input (s.data kind t h d)

def tensorCos (s : Shape) (xs : RotaryInputs ℝ s) (t d : Nat) : ℝ :=
  inputMemory s ((inputCoordinates ℝ s).symm xs) .cos (s.coeff Bool.false t d)

def tensorSin (s : Shape) (xs : RotaryInputs ℝ s) (t d : Nat) : ℝ :=
  inputMemory s ((inputCoordinates ℝ s).symm xs) .sin (s.coeff Bool.true t d)

def logicalMetadataDomain (c : Config) (xs : RotaryInputs ℝ c.shape) : Prop :=
  metadataDomain c ((inputCoordinates ℝ c.shape).symm xs)

def logicalIO (rest : Algebra ℝ) (c : Config) :
    LogicalProgramIO ℝ ℝ (RotaryInputs ℝ c.shape) (RotaryOutputs ℝ c.shape) where
  core := io rest realOps id (fun _ => id) c
  inputs := inputCoordinates ℝ c.shape
  outputs := outputCoordinates ℝ c.shape
end
end VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.PublicIO
