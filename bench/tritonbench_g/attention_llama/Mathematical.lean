import bench.tritonbench_g.attention_llama.IOLayout
import bench.tritonbench_g.attention_llama.Target

namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.Mathematical
open VeriTile Triton FP.Structural Source
noncomputable section

/-- Finite caller inputs extended only at masked key indices. The native
formula itself selects zero for those indices before any input observation. -/
def nativeRow (A : Target.Ops α) (B : Nat) (c : Config)
    (vals : ChannelValues α (IOLayout.inputs B c))
    (i : TileIndex [B,c.NH,c.H,c.D]) : Target.Inputs α c.D where
  query := fun d=>vals 0 (i.1,i.2.1,i.2.2.1,d,())
  halfKey := fun k d=>if h:k<c.N then vals 1 (i.1,i.2.1,⟨k,h⟩,d,()) else A.zero
  halfValue := fun k d=>if h:k<c.N then vals 2 (i.1,i.2.1,⟨k,h⟩,d,()) else A.zero
  byteKey := fun k d=>if h:k<c.N then
    UInt8.ofFin (Float8E5Bits.ofInt (vals 3 (i.1,i.2.1,⟨k,h⟩,d,()))) else 0
  byteValue := fun k d=>if h:k<c.N then
    UInt8.ofFin (Float8E5Bits.ofInt (vals 4 (i.1,i.2.1,⟨k,h⟩,d,()))) else 0

/-- Independent native attention formula with the original primitive
instruction profile, lane schedule and tagged runtime scalar ABI. -/
def native (A : Target.Ops α) (C : ScalarABI.Conversions α) (arg : ScalarABI.Argument)
    (B : Nat) (c : Config) (vals : ChannelValues α (IOLayout.inputs B c)) :
    ChannelValues α (IOLayout.outputs B c) :=
  Fin.cases (fun i=>Target.output A C (nativeRow A B c vals i) arg c.fp8 c.causal
    c.N c.SP (i.2.2.1.val/64) (i.2.2.1.val%64) i.2.2.2.1) (fun i=>Fin.elim0 i)

#specNonCircular native avoiding [FP.Structural.exec,FP.Structural.run]
end
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.Mathematical
