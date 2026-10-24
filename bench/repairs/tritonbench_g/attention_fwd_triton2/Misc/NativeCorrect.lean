import bench.repairs.tritonbench_g.attention_fwd_triton2.Proofs.NativeIO
import VeriTile.Meta.Specification

/-! Complete-grid source-to-native observation under the existing instruction
profile. Input words, all mixed precisions and physical owner replicas remain
explicit. This is an instruction observation, with no new rewrite or numerical
law; the historical local transport statements remain optional bridge results. -/
namespace VeriTile.Bench.Repairs.AttentionFwdTriton2.FPEquiv
open VeriTile Triton FP.Structural Whole001 Source Whole Machine Interface
noncomputable section

def nativeIO (I : KernelOnly.Instructions) (p : Parameters) (work : List Program) :
    ProgramIO (BitVec 32) Native.Expr :=
  io Native.algebra Native.dotModel p nativeInput (fun j=>observe32 I j.2.2.2.1.val) work

/-- Final BF16 payload observation; the source has already narrowed it. -/
def bf16Word (p : Parameters) (o : Fin (outputChannels p).length)
    (_ : TileIndex (outputChannels p)[o].shape) (word : BitVec 32) : BitVec 16 :=
  word.setWidth 16

/-- Every final BF16 word equals the independent native recurrence, retaining
all input aliases, grid permutations, actual accesses and exterior cells. -/
theorem kernel_raw_native (I : KernelOnly.Instructions) (p : Parameters) (bases : Bases) (dom : Domain p bases)
    (sep : p.Out≠p.Q ∧ p.Out≠p.K ∧ p.Out≠p.V ∧ p.Out≠p.QS ∧ p.Out≠p.KS)
    (work : List Program) (order : (programs (grid p)).Perm work) :
    (nativeIO I p work).Observes (bf16Word p) (nativeFormula I p) :=
  NativeIO.observes I p bases dom sep work order

#specNonCircular nativeFormula avoiding [FP.Structural.exec,Whole.runPrograms,Whole.answer,Source.accumulated,Source.update,Source.kernel]
#stmtSurfaceSubset kernel_raw_native ⊆
  [Native.Expr,Native.instInhabitedExpr,bf16Word,KernelOnly.Instructions,Parameters,Parameters.Out,Parameters.Q,Parameters.K,Parameters.V,Parameters.QS,Parameters.KS,
    RegionName,Bases,Domain,Program,programs,grid,nativeIO,ProgramIO.Observes,nativeFormula]
#axiomsClean kernel_raw_native
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionFwdTriton2.FPEquiv
