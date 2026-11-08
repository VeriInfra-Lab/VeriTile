import bench.tritonbench_g.chunk_retention_ops.Proofs.ScaleFirstSafety
import VeriTile.Spec

/-! Shared standalone-stage program views and their fixed IO judgment. -/
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.PairIO
open VeriTile Triton FP.Structural FP.ProfiledRewrite KernelPacket KernelNative PublicIO
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (withDot)
noncomputable section

def sourceInvocation (a : Call) : Invocation Term := launch (engine (withDot model (dotModel a))) a
def candidateInvocation (a : Call) : Invocation Term :=
  {sourceInvocation a with kernel:=candidateKernel a}

structure Implementation (isa : ISA) (literal : ℝ → Word) (a : Call) where
  invocation : Invocation Term

def original (isa : ISA) (literal : ℝ → Word) (a : Call) : Implementation isa literal a := ⟨sourceInvocation a⟩
def scaleFirst (isa : ISA) (literal : ℝ → Word) (a : Call) : Implementation isa literal a := ⟨candidateInvocation a⟩

def Equivalent {isa : ISA} {literal : ℝ → Word} {a : Call}
    (left right : Implementation isa literal a) : Prop :=
  let io:=nativeIO isa literal a
  ∀ xs bounds, io.InBounds bounds xs → ∀ initial, io.InputBinding xs initial →
    ScheduleSafe bounds [left.invocation] initial ∧ ScheduleSafe bounds [right.invocation] initial ∧
    ∃ l r, runSchedule [left.invocation] initial=some l ∧ runSchedule [right.invocation] initial=some r ∧
      (∀ o j, io.outputMask o xs j →
        mapCell (io.observe o j) (l.mem io.outputs[o].buf (io.write o xs j))=
        mapCell (io.observe o j) (r.mem io.outputs[o].buf (io.write o xs j))) ∧
      io.Frame xs initial l ∧ io.Frame xs initial r

instance {isa : ISA} {literal : ℝ → Word} {a : Call} :
    Spec.ProgramSyntax (Implementation isa literal a) where
  Statement := Implementation isa literal a
  Signature := Unit
  signature := fun _=>()
  body := fun i=>[i]
  structural := some Equivalent
  sameContext := Eq

end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.PairIO
