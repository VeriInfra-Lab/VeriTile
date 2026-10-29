import bench.repairs.tritonbench_g.fused_layernorm_triton.IO
import bench.repairs.tritonbench_g.fused_layernorm_triton.Proofs.Native
import VeriTile.Triton.Float.ProgramEquivalence
import VeriTile.Spec

/-! Pure complete-program views of the shared row IO. Proofs of execution,
observations and equivalence remain in Proofs/PairEquivalence. -/
namespace VeriTile.Bench.Repairs.FusedLayernormTriton.Pair
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.FusedLayernormTriton FPStages Whole
open scoped VeriTile.Spec
noncomputable section

/-- The original accepted geometry, output separation and ordered reduction profile. -/
structure Parameters where
  X : RegionName
  W : RegionName
  Bias : RegionName
  Y : RegionName
  Mean : RegionName
  Rstd : RegionName
  rows : Nat
  D : Nat
  positive : 0<D
  configuration : Contract.Configuration
  separation : Separation X W Bias Y Mean Rstd

structure Program (p : Parameters) where
  hoist : Bool

def Program.source {p : Parameters} (program : Program p) : ComputeKernel :=
  SourcePieces.source program.hoist p.X p.W p.Bias p.Y p.Mean p.Rstd p.rows p.D
    p.configuration.block p.configuration.tree

def Program.io {p : Parameters} (program : Program p) (I : Native.Instructions)
    (P : Native.Profile) : ProgramIO Native.Term Native.Term :=
  PairIO.io program.hoist Native.algebra (Native.equal I P)
    ((fun word => (.bits word : Native.Term)) ∘ Native.observe I P)
    p.X p.W p.Bias p.Y p.Mean p.Rstd p.rows p.D p.configuration

def variant {p : Parameters} (I : Native.Instructions) (P : Native.Profile)
    (left right : Program p) : ProgramIO.Variant (left.io I P) where
  schedule := (right.io I P).schedule
  observe := (left.io I P).observe

/-- Both actual grids safely succeed, agree on Mean/Rstd/Y typed word
observations and preserve both complete exterior frames, for all raw FP inputs. -/
def Equivalent {p : Parameters} (left right : Program p) : Prop :=
  ∀ (I : Native.Instructions) (P : Native.Profile),
    (left.io I P).Equivalent (variant I P left right)

instance {p : Parameters} : Spec.ProgramSyntax (Program p) where
  Statement := ComputeStmt
  Signature := List IOChannel × List IOChannel
  signature _ := (PairIO.inputs p.X p.W p.Bias p.rows p.D,PairIO.outputs p.Y p.Mean p.Rstd p.rows p.D)
  body program := (List.range p.rows).flatMap (fun _ => program.source.surfaceBody)
  structural := some Equivalent
  sameContext := fun left right => left=right

def original (p : Parameters) : Program p := ⟨Bool.false⟩
def candidate (p : Parameters) : Program p := ⟨Bool.true⟩
def assumptions : Spec.Assumptions ComputeStmt := []

end
end VeriTile.Bench.Repairs.FusedLayernormTriton.Pair
