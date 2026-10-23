import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.NativePlans
namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.Native
open Whole
noncomputable section

/-- Shape and ordered lowering syntax extracted from an actual specialization.
There are no input values, numerical functions or result memories here. -/
structure Descriptor where
  shape : List Nat
  keyWidth : Width
  valueWidth : Width
  warps : Warps
  history : Layout
  correction : Layout
  correctionMMA : Bool
  appendThirdCorrection : Bool
  reverseRowOperands : Bool
  swapCorrectionOperands : Bool
  historyCarryFirst : Bool

/-- Runtime FP32 bits are separate from the shape-only compiler descriptor.
In particular the native scale may be negative zero. -/
structure Runtime where
  scaleBits : BitVec 32

def Descriptor.width (d : Descriptor) : Width := d.keyWidth

def Descriptor.historySum (d : Descriptor) (xs : Nat → Graph) : Graph :=
  let l:=d.history
  let partials:=fun thread=>productLocal l.count (fun slot=>xs (l.index thread slot))
  let first:=if l.count=1 then
      fun thread=>fusedAdd (partials thread) (.shuffle 16 (partials (thread ^^^ 16)))
    else butterfly [16] partials
  let partialWarps:=fun w=>butterfly [8,4,2,1] first (w*32)
  butterfly (match d.warps with | .four=>[2,1] | .eight=>[4,2,1]) partialWarps 0

def Descriptor.correctionSum (d : Descriptor) (xs : Nat → Graph) : Graph :=
  let partials:=if d.correctionMMA then
      fun thread=>productLocal (d.width.size/2) (fun slot=>xs (mmaIndex d.width.size thread slot))
    else fun thread=>addLocal d.appendThirdCorrection d.correction.count
      (fun slot=>xs (d.correction.index thread slot))
  let partialWarps:=fun w=>butterfly [16,8,4,2,1] partials (w*32)
  butterfly (match d.warps with | .four=>[2,1] | .eight=>[4,2,1]) partialWarps 0

def Layout.indices (l : Layout) (warps : Warps) : List Nat :=
  (List.range (32*warps.count)).flatMap fun thread=>
    (List.range l.count).map fun slot=>l.index thread slot

/-- Only integer layout coverage and source dimensions. No floating reduction
identity, kernel success or output equality is part of admission. -/
structure Descriptor.Admissible (d : Descriptor) (c : Config) : Prop where
  shape : d.shape=[c.B,c.H,c.T,c.K,c.V]
  keyWidth : d.keyWidth.size=c.BK
  valueWidth : d.valueWidth.size=c.BV
  historyRows : d.history.rows=c.BV
  historyCols : d.history.cols=c.BK
  historyCoverage : (d.history.indices d.warps).Perm (List.range (c.BV*c.BK))
  historyNonempty : 0<d.history.count
  correctionRows : d.correction.rows=64
  correctionCols : d.correction.cols=c.BK
  correctionCoverage : (d.correction.indices d.warps).Perm (List.range (64*c.BK))
  correctionNonempty : 0<d.correction.count
  correctionMMAWarps : d.correctionMMA=Bool.true → d.warps=.four

/-- Finite artifact catalog, retained only as concrete descriptor examples. -/
def Profile.shape (p : Profile) : List Nat :=
  match p.family with
  | .short=>[1,1,1,1,1] | .tile32=>[1,1,63,17,31]
  | .aligned=>[1,1,128,64,64] | .tails=>[1,2,65,65,65]
  | .multiblock=>[2,2,129,128,129] | .unaligned=>[1,2,67,33,63]

def Profile.descriptor (p : Profile) : Descriptor :=
  ⟨p.shape,p.width,p.width,p.warps,historyLayout p,correctionLayout p,
    decide (p.warps=.four),appendThirdCorrection p,reverseRowOperands p,
    !(reverseRowOperands p),decide (p.family=.tails ∨ p.family=.multiblock)⟩

theorem saved_history_plan (p : Profile) (xs : Nat → Graph) :
    p.descriptor.historySum xs=historySum p xs := by cases p with | mk f w=>cases w <;> rfl

theorem saved_correction_plan (p : Profile) (xs : Nat → Graph) :
    p.descriptor.correctionSum xs=correctionSum p xs := by cases p with | mk f w=>cases w <;> rfl

#axiomsClean saved_history_plan
#axiomsClean saved_correction_plan
#specNonCircular Descriptor.Admissible avoiding [VeriTile.Triton.FP.Structural.exec]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.Native
