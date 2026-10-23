import bench.repairs.tritonbench_g.token_attn_mistral_grid_y_checked.Proofs.SourceFormula

namespace VeriTile.Bench.Repairs.TokenAttnMistralGridYChecked.Reference
open VeriTile Triton FP.Structural TokenAttnMistral
noncomputable section

def terms (R : Native.Reductions) (p : Parameters) (work : List Grid.PID)
    (counts : Fin work.length → Nat) (vals : Ports.Values Native.Term p work counts) :
    ChannelValues Native.Term (Ports.outputs p work) :=
  fun o=>Fin.cases (fun i=>result Native.algebra (Native.reduction R) p (Ports.logical p work counts vals) i.1 i.2.1) (fun i=>Fin.elim0 i) o

def native (I : Native.Instructions) (R : Native.Reductions) (S : Native.Schedule)
    (p : Parameters) (work : List Grid.PID) (counts : Fin work.length → Nat)
    (vals : Ports.Values Native.Term p work counts) : ChannelValues Native.Word (Ports.outputs p work) :=
  fun o=>Fin.cases (fun i=>Native.eval I S (result Native.algebra (Native.reduction R) p
    (Ports.logical p work counts vals) i.1 i.2.1)) (fun i=>Fin.elim0 i) o

#specNonCircular native avoiding [kernel,FP.Structural.exec,Grid.run,outputValues]
end
end VeriTile.Bench.Repairs.TokenAttnMistralGridYChecked.Reference
