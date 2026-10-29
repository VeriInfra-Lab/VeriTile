import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.Host

namespace VeriTile.Bench.Repairs.AttnFwdTritonTileChecked.RealBinding
open VeriTile Triton AttnFwdTriton
open Verified Verified.PhysicalCells
open VeriTile.Bench.Repairs.LayerNormFwd
noncomputable section

def cells (a : Layout) (memory : Nat→ℝ) : AliasMemory.Physical a.toLayout :=
  fun cell=>MemCell.real (memory cell.val)

/-- Independent input functions use actual byte pointers and each role's own
strides. They mention neither a source execution nor output cells. -/
def inputs (p : RealProofs.Parameters) (base : Role→Nat) (memory : Nat→ℝ) : RealProofs.Inputs :=
  ⟨fun z h n d=>memory (base .query+4*offset p z h n d .query),
   fun z h n d=>memory (base .key+4*offset p z h n d .key),
   fun z h n d=>memory (base .value+4*offset p z h n d .value),
   fun i=>memory (base .queryScale+4*i),fun i=>memory (base .keyScale+4*i)⟩

theorem read_input (p : RealProofs.Parameters) (a : Layout) (A : FlatAlloc) (base : Role→Nat)
    (views : Views a A p base) (memory : Nat→ℝ) (template : BlockState)
    (z h n d : Nat) (r : Role) (ha : active p z h n d r) :
    (AliasMemory.pull a.toLayout (cells a memory) template).readMem
      A.flat (offset p z h n d r+A.base (region p r)) =
      memory (base r+width r*offset p z h n d r) := by
  change Host.readReal (AliasMemory.read a.toLayout (cells a memory) (key A p z h n d r))=_
  rw [active_read a A p base views (cells a memory) z h n d r ha]
  rfl

theorem agreement (p : RealProofs.Parameters) (a : Layout) (A : FlatAlloc) (base : Role→Nat)
    (views : Views a A p base) (memory : Nat→ℝ) (template : BlockState) :
    LogicalAgreement.Agree p (Host.logicalInputs a A p (cells a memory) template) (inputs p base memory) := by
  constructor
  · intro z hz h hh n hn d hd
    exact read_input p a A base views memory template z h n d .query ⟨hz,hh,hn,hd⟩
  · intro z hz h hh n hn d hd
    exact read_input p a A base views memory template z h n d .key ⟨hz,hh,hn,hd⟩
  · intro z hz h hh n hn d hd
    exact read_input p a A base views memory template z h n d .value ⟨hz,hh,hn,hd⟩
  · intro i hi
    exact read_input p a A base views memory template 0 0 i 0 .queryScale ⟨rfl,rfl,rfl,hi⟩
  · intro i hi
    exact read_input p a A base views memory template 0 0 i 0 .keyScale ⟨rfl,rfl,rfl,hi⟩

#specNonCircular inputs avoiding [Host.logicalInputs,Kernel.GridLaunchedOrdinary]
#axiomsClean agreement
end
end VeriTile.Bench.Repairs.AttnFwdTritonTileChecked.RealBinding
