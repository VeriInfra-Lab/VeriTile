import bench.repairs.tritonbench_g.attn_fwd_causal.Proofs.SourceFormula

namespace VeriTile.Bench.Repairs.AttnFwdCausal.NativeIO
open VeriTile Triton FP.Structural Source Whole
noncomputable section

/-- All input FP32 words are admitted. The pure output observer uses each
feature's physical reduction owner and the existing local instruction model. -/
def program (instructions : Native.Instructions) (p : Parameters) (work : List Program) :
    ProgramIO Native.Word Native.Expr :=
  IO.program Native.algebra Native.dotModel p work (Native.inject .real)
    (fun feature=>Native.project instructions (Native.physicalOwner feature.val) .fp16)

def formula (instructions : Native.Instructions) (p : Parameters) (work : List Program) :
    (program instructions p work).Inputs → (program instructions p work).Outputs :=
  fun vals o=>Fin.cases (Reference.native instructions p vals) (fun i=>Fin.elim0 i) o

end
end VeriTile.Bench.Repairs.AttnFwdCausal.NativeIO
