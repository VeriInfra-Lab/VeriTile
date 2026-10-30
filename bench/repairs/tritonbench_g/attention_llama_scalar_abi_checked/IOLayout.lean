import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.Proofs.Source
import VeriTile.Triton.Float.OrdinaryProgramIO

namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.IOLayout
open VeriTile Triton FP.Structural Source
noncomputable section

/-- Five logical source roles share the original three regions. Only the
selected FP16 or signed-byte K/V roles are active. -/
@[reducible] def inputs (B : Nat) (c : Config) : List IOChannel :=
  [⟨.fp16,[B,c.NH,c.H,c.D],c.Q⟩,
   ⟨.fp16,[B,c.NH,c.N,c.D],c.K⟩,⟨.fp16,[B,c.NH,c.N,c.D],c.V⟩,
   ⟨.int,[B,c.NH,c.N,c.D],c.K⟩,⟨.int,[B,c.NH,c.N,c.D],c.V⟩]

@[reducible] def outputs (B : Nat) (c : Config) : List IOChannel :=
  [⟨.fp16,[B,c.NH,c.H,c.D],c.Out⟩]

/-- Element offsets retain all four actual caller strides. -/
def offset (sz sh sr sd : Nat) (i : TileIndex [B,NH,R,D]) : Nat :=
  i.1.val*sz+i.2.1.val*sh+i.2.2.1.val*sr+i.2.2.2.1.val*sd

def query (c : Config) : TileIndex [B,c.NH,c.H,c.D] → Nat :=
  offset c.sqz c.sqh c.sqm c.sqk
def key (c : Config) : TileIndex [B,c.NH,c.N,c.D] → Nat :=
  offset c.skz c.skh c.skn c.skk
def value (c : Config) : TileIndex [B,c.NH,c.N,c.D] → Nat :=
  offset c.svz c.svh c.svk c.svn
def output (c : Config) : TileIndex [B,c.NH,c.H,c.D] → Nat :=
  offset c.soz c.soh c.som c.son

/-- Complete finite input and output windows. Padding outside the logical
query/key extents is masked; no private memory is introduced. -/
@[reducible] def memory (α : Type) (B : Nat) (c : Config) : MemoryIO α where
  inputs := inputs B c
  outputs := outputs B c
  scratch := []
  buffers := [c.Q,c.K,c.V,c.Out]
  read := fun i _ => Fin.cases (query c)
    (Fin.cases (key c) (Fin.cases (value c)
      (Fin.cases (key c) (Fin.cases (value c) (fun i=>Fin.elim0 i))))) i
  inputMask := fun i _ _ =>
    ((i.val=1 ∨ i.val=2) → c.fp8=Bool.false) ∧
      ((i.val=3 ∨ i.val=4) → c.fp8=Bool.true)
  write := fun o _ => Fin.cases (output c) (fun i=>Fin.elim0 i) o
  outputMask := fun _ _ _ => True
  scratchWrite := fun i=>Fin.elim0 i
  scratchMask := fun i=>Fin.elim0 i

/-- Signed-byte representation is required only on active E5M2 input cells.
No finite-floating-word assumption is added. -/
def nativeDomain (B : Nat) (c : Config) (vals : ChannelValues α (inputs B c)) : Prop :=
  c.fp8=Bool.true → ∀ j : TileIndex [B,c.NH,c.N,c.D],
    Float8E5Bits.isInt8 (vals 3 j)=Bool.true ∧
      Float8E5Bits.isInt8 (vals 4 j)=Bool.true

end
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.IOLayout
