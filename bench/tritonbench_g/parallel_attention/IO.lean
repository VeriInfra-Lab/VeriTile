import bench.tritonbench_g.parallel_attention.RealCombined
import bench.tritonbench_g.parallel_attention.Proofs.IOLogical
import VeriTile.Triton.Float.ObservedProgramIO
import VeriTile.Triton.Float.LogicalProgramIO
import Mathlib.Logic.Equiv.Fin.Basic
import Mathlib.Tactic.FinCases

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel.IOData
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.ParallelAttention
noncomputable section
open Classical

inductive Direction where | forward | backward deriving DecidableEq
abbrev heads (shape : Shape) := shape.B*shape.H
abbrev shards (shape : Shape) := blocks shape.V shape.BV
abbrev Point (shape : Shape) (L width : Nat) :=
  TileIndex [heads shape,blocks shape.T L,shards shape,L,width]
abbrev NormPoint (shape : Shape) := TileIndex [heads shape,blocks shape.T 128,shards shape,128]

def program (shape : Shape) (L width : Nat) (j : Point shape L width) : Program shape L :=
  ⟨j.1,j.2.1,j.2.2.1⟩
def normProgram (shape : Shape) (j : NormPoint shape) : Program shape 128 :=
  ⟨j.1,j.2.1,j.2.2.1⟩
def pid {shape : Shape} {L : Nat} (p : Program shape L) : Nat → Nat :=
  fun ax=>if ax=0 then p.value.val else if ax=1 then p.chunk.val else if ax=2 then p.head.val else 0

def row {shape : Shape} {L width : Nat} (j : Point shape L width) := j.2.1.val*L+j.2.2.2.1.val
def col {shape : Shape} {L width : Nat} (j : Point shape L width) :=
  j.2.2.1.val*shape.BV+j.2.2.2.2.1.val

abbrev inputPorts (shape : Shape) : List IOChannel :=
  [⟨.real,[heads shape*shape.T*shape.K],"q"⟩,⟨.real,[heads shape*shape.T*shape.K],"k"⟩,
   ⟨.real,[heads shape*shape.T*shape.V],"v"⟩,⟨.real,[heads shape*shape.T*shape.V],"do"⟩,
   ⟨.real,[heads shape*shape.T],"dz"⟩]
abbrev outputPorts (shape : Shape) : List IOChannel :=
  [⟨.real,[heads shape,blocks shape.T 128,shards shape,128,shape.BV],"o"⟩,
   ⟨.real,[heads shape,blocks shape.T 128,shards shape,128],"z"⟩,
   ⟨.real,[heads shape,blocks shape.T 64,shards shape,64,shape.BK],"dq"⟩,
   ⟨.real,[heads shape,blocks shape.T 64,shards shape,64,shape.BK],"dk"⟩,
   ⟨.real,[heads shape,blocks shape.T 64,shards shape,64,shape.BV],"dv"⟩]

def inputActive (direction : Direction) (i : Fin 5) : Prop := i.val<3 ∨ direction=.backward

def outputActive (direction : Direction) (shape : Shape) :
    (o : Fin (outputPorts shape).length) → TileIndex (outputPorts shape)[o].shape → Prop
  | ⟨0,_⟩ => fun j=>direction=.forward ∧ row j<shape.T ∧ col j<shape.V
  | ⟨1,_⟩ => fun j=>direction=.forward ∧ j.2.2.1.val=0 ∧ j.2.1.val*128+j.2.2.2.1.val<shape.T
  | ⟨2,_⟩ | ⟨3,_⟩ => fun j=>direction=.backward ∧ row j<shape.T ∧ j.2.2.2.2.1.val<shape.K
  | ⟨4,_⟩ => fun j=>direction=.backward ∧ row j<shape.T ∧ col j<shape.V

def outputWindow (shape : Shape) :
    (o : Fin (outputPorts shape).length) → TileIndex (outputPorts shape)[o].shape → Nat
  | ⟨0,_⟩ => fun j=>dense shape.T shape.V j.1.val (row j) (col j)
  | ⟨1,_⟩ => fun j=>j.1.val*shape.T+(j.2.1.val*128+j.2.2.2.1.val)
  | ⟨2,_⟩ | ⟨3,_⟩ => fun j=>dense shape.T shape.K
      (j.1.val+heads shape*j.2.2.1.val) (row j) j.2.2.2.2.1.val
  | ⟨4,_⟩ => fun j=>dense shape.T shape.V j.1.val (row j) (col j)

def inputWindow (shape : Shape) :
    (i : Fin (inputPorts shape).length) → TileIndex (inputPorts shape)[i].shape → Nat
  | ⟨0,_⟩ | ⟨1,_⟩ | ⟨2,_⟩ | ⟨3,_⟩ | ⟨4,_⟩ => fun j=>j.1.val

abbrev memory (α : Type) (direction : Direction) (shape : Shape) : MemoryIO α where
  inputs := inputPorts shape
  outputs := outputPorts shape
  scratch := []
  buffers := ["q","k","v","do","dz","o","z","dq","dk","dv"]
  read := fun i _=>inputWindow shape i
  inputMask := fun i _ _=>inputActive direction i
  write := fun o _=>outputWindow shape o
  outputMask := fun o _=>outputActive direction shape o
  scratchWrite := fun i=>Fin.elim0 i
  scratchMask := fun i=>Fin.elim0 i

/-- Finite logical input lookup, independent of source states and outputs. -/
def inputValues {α : Type} [Inhabited α] (shape : Shape) (xs : ChannelValues α (inputPorts shape)) :
    IOLogical.InputValues α := fun r off=>
  if r="q" then if h : off<heads shape*shape.T*shape.K then xs 0 (⟨off,h⟩,()) else default
  else if r="k" then if h : off<heads shape*shape.T*shape.K then xs 1 (⟨off,h⟩,()) else default
  else if r="v" then if h : off<heads shape*shape.T*shape.V then xs 2 (⟨off,h⟩,()) else default
  else if r="do" then if h : off<heads shape*shape.T*shape.V then xs 3 (⟨off,h⟩,()) else default
  else if r="dz" then if h : off<heads shape*shape.T then xs 4 (⟨off,h⟩,()) else default
  else default

def tensorValues {α : Type} [Inhabited α] (shape : Shape) (xs : ChannelValues α (inputPorts shape)) :
    TensorInputs α where
  q := fun h t d=>inputValues shape xs "q" (dense shape.T shape.K h t d)
  k := fun h t d=>inputValues shape xs "k" (dense shape.T shape.K h t d)
  v := fun h t d=>inputValues shape xs "v" (dense shape.T shape.V h t d)
  do_ := fun h t d=>inputValues shape xs "do" (dense shape.T shape.V h t d)
  dz := fun h t=>inputValues shape xs "dz" (h*shape.T+t)

#specNonCircular inputValues avoiding [State,FP.Structural.exec,runPrograms]
/-- Shared IO data specialization. The callers supply only actual source
invocations; the shared interface fixes execution and safety. -/
abbrev dataIO (α : Type) (direction : Direction) (shape : Shape)
    (jobs : List (Invocation α)) : ProgramIO α α where
  toMemoryIO := memory α direction shape
  schedule := fun _=>jobs
  encode := id
  observe := fun _ _=>id

def forwardJob {α : Type} (M : Algebra α) (D : DotPrimitive α)
    (shape : Shape) (scale : ℝ) (p : Program shape 128) : Invocation α where
  kernel := forwardKernel (forwardConfig shape scale)
  backend := withDot M D
  pids := pid p

def queryJob {α : Type} (M : Algebra α) (D : DotPrimitive α)
    (shape : Shape) (scale : ℝ) (p : Program shape 64) : Invocation α where
  kernel := queryKernel (dqConfig shape scale p.tile)
  backend := withDot M D
  pids := fun _=>0
  setPids := fun _=>Bool.false

def keyValueJob {α : Type} (M : Algebra α) (D : DotPrimitive α)
    (shape : Shape) (scale : ℝ) (p : Program shape 64) : Invocation α where
  kernel := backwardKernel (backwardConfig shape scale p.tile)
  backend := withDot M D
  pids := fun _=>0
  setPids := fun _=>Bool.false

def combinedJob (shape : Shape) (scale : ℝ) (p : Program shape 64) : Invocation ℝ where
  kernel := combinedKernel shape scale p.tile
  backend := withDot (scalarAlgebra realScalars) (scalarDotPrimitive realScalars)
  pids := pid p

abbrev realJobs (direction : Direction) (shape : Shape) (scale : ℝ) : List (Invocation ℝ) :=
  match direction with
  | .forward => (programs shape 128).map (forwardJob (scalarAlgebra realScalars)
      (scalarDotPrimitive realScalars) shape scale)
  | .backward => (programs shape 64).map (combinedJob shape scale)

/-- The native source model retains the original per-phase compiler algebra.
DQ precedes DKV for each grid point, exactly as in NativeStage.combinedRun. -/
abbrev nativeJobs (direction : Direction) (d : NativePlans.Descriptor) (scale : ℝ) :
    List (Invocation NativeStage.Expr) := match direction with
  | .forward => (programs d.shape 128).map (forwardJob (NativeStage.algebra .forward d)
      (NativeStage.dots .forward d.shape) d.shape scale)
  | .backward => (programs d.shape 64).flatMap fun p=>
      [queryJob (NativeStage.algebra .query d) (NativeStage.dots .query d.shape) d.shape scale p,
       keyValueJob (NativeStage.algebra .keyValue d) (NativeStage.dots .keyValue d.shape) d.shape scale p]

end
end VeriTile.Bench.Repairs.ParallelAttentionKernel.IOData

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel.LogicalView
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.ParallelAttention IOData
noncomputable section

/-- Dense coordinates, including empty dimensions, with no value discarded. -/
def flat3 (H T D : Nat) : (TileIndex [H*T*D] → ℝ) ≃ (Fin H → Fin T → Fin D → ℝ) := by
  let e : ((Fin H × Fin T) × Fin D) ≃ Fin (H*T*D) :=
    (Equiv.prodCongr finProdFinEquiv (Equiv.refl _)).trans finProdFinEquiv
  exact {
    toFun := fun xs h t d => xs (e ((h,t),d),())
    invFun := fun xs j => xs (e.symm j.1).1.1 (e.symm j.1).1.2 (e.symm j.1).2
    left_inv := by intro xs; funext j; simp only [e.apply_symm_apply]
    right_inv := by intro xs; funext h t d; simp only [e.symm_apply_apply] }

def flat2 (H T : Nat) : (TileIndex [H*T] → ℝ) ≃ (Fin H → Fin T → ℝ) where
  toFun := fun xs h t => xs (finProdFinEquiv (h,t),())
  invFun := fun xs j => xs (finProdFinEquiv.symm j.1).1 (finProdFinEquiv.symm j.1).2
  left_inv := by
    intro xs; funext j
    change xs (finProdFinEquiv (finProdFinEquiv.symm j.1),())=xs j
    rw [finProdFinEquiv.apply_symm_apply]
  right_inv := by intro xs; funext h t; simp only [finProdFinEquiv.symm_apply_apply]

structure Inputs (s : Shape) where
  q : Fin (heads s) → Fin s.T → Fin s.K → ℝ
  k : Fin (heads s) → Fin s.T → Fin s.K → ℝ
  v : Fin (heads s) → Fin s.T → Fin s.V → ℝ
  gradient : Fin (heads s) → Fin s.T → Fin s.V → ℝ
  normalizationGradient : Fin (heads s) → Fin s.T → ℝ

def inputEquiv (s : Shape) : ChannelValues ℝ (inputPorts s) ≃ Inputs s where
  toFun := fun xs => ⟨flat3 _ _ _ (xs 0),flat3 _ _ _ (xs 1),flat3 _ _ _ (xs 2),
    flat3 _ _ _ (xs 3),flat2 _ _ (xs 4)⟩
  invFun := fun xs => Fin.cases ((flat3 _ _ _).symm xs.q)
    (Fin.cases ((flat3 _ _ _).symm xs.k)
      (Fin.cases ((flat3 _ _ _).symm xs.v)
        (Fin.cases ((flat3 _ _ _).symm xs.gradient)
          (Fin.cases ((flat2 _ _).symm xs.normalizationGradient) (fun k => k.elim0)))))
  left_inv := by intro xs; funext o; fin_cases o <;> exact Equiv.symm_apply_apply _ _
  right_inv := by
    intro xs
    cases xs with
    | mk q k v g dz =>
      change Inputs.mk ((flat3 _ _ _ ) ((flat3 _ _ _).symm q))
        ((flat3 _ _ _ ) ((flat3 _ _ _).symm k))
        ((flat3 _ _ _ ) ((flat3 _ _ _).symm v))
        ((flat3 _ _ _ ) ((flat3 _ _ _).symm g))
        ((flat2 _ _ ) ((flat2 _ _).symm dz)) = _
      simp only [Equiv.apply_symm_apply]

/-- Natural coordinates extend the original independent finite tensor view.
Only active output coordinates are externally bound by the original masks. -/
def tensors (s : Shape) (xs : Inputs s) : TensorInputs ℝ :=
  tensorValues s ((inputEquiv s).symm xs)

abbrev Time (s : Shape) (L : Nat) := Fin (blocks s.T L*L)
structure Outputs (s : Shape) where
  value : Fin (heads s) → Fin (shards s) → Time s 128 → Fin s.BV → ℝ
  normalization : Fin (heads s) → Fin (shards s) → Time s 128 → ℝ
  queryGradient : Fin (heads s) → Fin (shards s) → Time s 64 → Fin s.BK → ℝ
  keyGradient : Fin (heads s) → Fin (shards s) → Time s 64 → Fin s.BK → ℝ
  valueGradient : Fin (heads s) → Fin (shards s) → Time s 64 → Fin s.BV → ℝ

def outputEquiv (s : Shape) : ChannelValues ℝ (outputPorts s) ≃ Outputs s where
  toFun := fun ys => ⟨
    (fun h v t c => ys 0 (h,(finProdFinEquiv.symm t).1,v,(finProdFinEquiv.symm t).2,c,())),
    (fun h v t => ys 1 (h,(finProdFinEquiv.symm t).1,v,(finProdFinEquiv.symm t).2,())),
    (fun h v t c => ys 2 (h,(finProdFinEquiv.symm t).1,v,(finProdFinEquiv.symm t).2,c,())),
    (fun h v t c => ys 3 (h,(finProdFinEquiv.symm t).1,v,(finProdFinEquiv.symm t).2,c,())),
    (fun h v t c => ys 4 (h,(finProdFinEquiv.symm t).1,v,(finProdFinEquiv.symm t).2,c,()))⟩
  invFun := fun ys => Fin.cases (fun j => ys.value j.1 j.2.2.1 (finProdFinEquiv (j.2.1,j.2.2.2.1)) j.2.2.2.2.1)
    (Fin.cases (fun j => ys.normalization j.1 j.2.2.1 (finProdFinEquiv (j.2.1,j.2.2.2.1)))
      (Fin.cases (fun j => ys.queryGradient j.1 j.2.2.1 (finProdFinEquiv (j.2.1,j.2.2.2.1)) j.2.2.2.2.1)
        (Fin.cases (fun j => ys.keyGradient j.1 j.2.2.1 (finProdFinEquiv (j.2.1,j.2.2.2.1)) j.2.2.2.2.1)
          (Fin.cases (fun j => ys.valueGradient j.1 j.2.2.1 (finProdFinEquiv (j.2.1,j.2.2.2.1)) j.2.2.2.2.1) (fun k => k.elim0)))))
  left_inv := by
    intro ys; funext o j; fin_cases o <;> simp only [finProdFinEquiv.symm_apply_apply] <;> rfl
  right_inv := by
    intro ys
    cases ys with
    | mk value norm dq dk dv =>
      change Outputs.mk
        (fun h v t c => value h v (finProdFinEquiv (finProdFinEquiv.symm t)) c)
        (fun h v t => norm h v (finProdFinEquiv (finProdFinEquiv.symm t)))
        (fun h v t c => dq h v (finProdFinEquiv (finProdFinEquiv.symm t)) c)
        (fun h v t c => dk h v (finProdFinEquiv (finProdFinEquiv.symm t)) c)
        (fun h v t c => dv h v (finProdFinEquiv (finProdFinEquiv.symm t)) c)=_
      simp only [Equiv.apply_symm_apply]

def attentionIO (direction : Direction) (s : Shape) (scale : ℝ) :
    LogicalProgramIO ℝ ℝ (Inputs s) (Outputs s) where
  core := dataIO ℝ direction s (realJobs direction s scale)
  inputs := inputEquiv s
  outputs := outputEquiv s
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel.LogicalView
