/- Retained single-source native/software correctness. Original declaration
namespaces and full statements are preserved as ordinary auxiliary theorems. -/
import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.PublicForwardIO
import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.PublicBackwardIO
import VeriTile.Meta.Specification
import VeriTile.Triton.Float.BitValue

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.FPEquiv
open VeriTile.Triton FP.Structural Kernel Kernel.Native
noncomputable section

abbrev Word := VeriTile.Triton.FP.Value .fp32

/- Full source-to-native observation under the existing FP32 scalar primitives
and finite dot instruction trees. Every FP32 word is admitted. The .real cell
tag is the existing Structural representation; loads, casts, dot contexts and
stores retain the source's FP32 precision. No second optimized kernel is claimed. -/

def forwardIO (a : Scalar Word) (p : DotTrees) (s : Shape) (tile : Tile) :
    ProgramIO Word (Expr Word) where
  inputs := [⟨.real,[s.B*s.H,s.T,s.DK],"q"⟩,⟨.real,[s.B*s.H,s.T,s.DK],"k"⟩,
    ⟨.real,[s.B*s.H,s.T,s.DV],"v"⟩]
  outputs := [⟨.real,[128,s.BV],"o"⟩]
  scratch := []
  buffers := ["q","k","v","o"]
  read := PublicIO.forwardRead s
  inputMask := fun _ _ _ => True
  write := Fin.cases (fun _ j => forwardAddress s tile j) (fun k => k.elim0)
  outputMask := Fin.cases (fun _ j => activeTile s.T s.DV (tile.chunk*128) (tile.value*s.BV) j)
    (fun k => k.elim0)
  scratchWrite := fun k => k.elim0
  scratchMask := fun k => k.elim0
  schedule := fun _ => [{ (PublicIO.forwardInvocation Word s tile) with kernel := forwardKernel s }]
  encode := Expr.input
  observe := fun _ _ => evaluate a p

def backwardIO (a : Scalar Word) (p : DotTrees) (s : Shape) (tile : Tile) :
    ProgramIO Word (Expr Word) where
  inputs := [⟨.real,[s.B*s.H,s.T,s.DK],"q"⟩,⟨.real,[s.B*s.H,s.T,s.DK],"k"⟩,
    ⟨.real,[s.B*s.H,s.T,s.DV],"v"⟩,⟨.real,[s.B*s.H,s.T,s.DV],"g"⟩]
  outputs := [⟨.real,[64,s.BK],"dq"⟩,⟨.real,[64,s.BK],"dk"⟩,⟨.real,[64,s.BV],"dv"⟩]
  scratch := []
  buffers := ["q","k","v","g","dq","dk","dv"]
  read := PublicIO.backwardRead s
  inputMask := fun _ _ _ => True
  write := PublicIO.backwardWrite s tile
  outputMask := PublicIO.backwardMask s tile
  scratchWrite := fun k => k.elim0
  scratchMask := fun k => k.elim0
  schedule := fun _ => [{ (PublicIO.backwardInvocation Word s tile) with
    kernel := backwardKernel s (PublicIO.programIds s tile) }]
  encode := Expr.input
  observe := fun _ _ => evaluate a p


/-- Independent source-ordered forward scalar and dot formula. -/
def forwardFormula (a : Scalar Word) (p : DotTrees) (s : Shape) (tile : Tile)
    (xs : ChannelValues Word (PublicIO.forwardInputs s)) :
    ChannelValues Word (PublicIO.forwardOutputs s) :=
  Fin.cases (fun j => forwardLane a.ops (plans s p) s
    (PublicIO.forwardLogical a.ops.zero s xs) tile j.1.val j.2.1.val) (fun k => k.elim0)

/-- All three backward contributions, with their original operation order. -/
def backwardFormula (a : Scalar Word) (p : DotTrees) (s : Shape) (tile : Tile)
    (xs : ChannelValues Word (PublicIO.backwardInputs s)) :
    ChannelValues Word (PublicIO.backwardOutputs s) := by
  let x := PublicIO.backwardLogical a.ops.zero s xs
  refine Fin.cases ?_ (Fin.cases ?_ (Fin.cases ?_ (fun k => k.elim0)))
  · intro j; exact queryLane a.ops (plans s p) s x tile j.1.val j.2.1.val
  · intro j; exact (keyValueLane a.ops (plans s p) s x tile j.1.val j.2.1.val 0).1
  · intro j; exact (keyValueLane a.ops (plans s p) s x tile j.1.val 0 j.2.1.val).2

theorem forward (a : Scalar Word) (p : DotTrees) (s : Shape) (tile : Tile)
    (valid : ProgramDomain s 128 tile) :
    (forwardIO a p s tile).Implements (forwardFormula a p s tile) :=
  PublicIO.forward_implements a p s tile valid

theorem backward (a : Scalar Word) (p : DotTrees) (s : Shape) (tile : Tile)
    (valid : ProgramDomain s 64 tile) (keyPositive : 0<s.BK) (valuePositive : 0<s.BV) :
    (backwardIO a p s tile).Implements (backwardFormula a p s tile) :=
  PublicIO.backward_implements a p s tile valid keyPositive valuePositive

/- Scalar and dot-tree parameters are primitive/compiler models. No new
floating-point rewrite law or deferred numerical experiment is introduced. -/
#stmtSurfaceSubset forward ⊆
  [Scalar,Word,DotTrees,Shape,Tile,ProgramDomain,forwardIO,ProgramIO.Implements,forwardFormula,Expr,instInhabitedExpr]
#stmtSurfaceSubset backward ⊆
  [Scalar,Word,DotTrees,Shape,Shape.BK,Shape.BV,Tile,ProgramDomain,backwardIO,ProgramIO.Implements,backwardFormula,Expr,instInhabitedExpr]
#specNonCircular forwardFormula avoiding [forwardKernel,backwardKernel,FP.Structural.exec,FP.Structural.run]
#specNonCircular backwardFormula avoiding [forwardKernel,backwardKernel,FP.Structural.exec,FP.Structural.run]
#axiomsClean forward
#axiomsClean backward
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.FPEquiv
