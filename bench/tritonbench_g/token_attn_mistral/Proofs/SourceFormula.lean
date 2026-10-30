import bench.tritonbench_g.token_attn_mistral.Ports
import bench.tritonbench_g.token_attn_mistral.Proofs.Native

namespace VeriTile.Bench.Repairs.TokenAttnMistralGridYChecked.Reference
open VeriTile Triton FP.Structural TokenAttnMistral
noncomputable section

/-- Independent gathered weighted sum. The value channel already carries the
page-table address binding declared by Ports.memory. -/
def weighted (p : Parameters) {jobs extent : Nat}
    (x : Ports.Logical ℝ jobs extent p.width) (j : Fin jobs) (d : Fin p.width) : ℝ :=
  ∑ n ∈ Finset.range (x.attention j),
    if n+Ports.window p x j<x.sequence j then probAt 0 x j n*valueAt 0 x j n d else 0

def mathematical (p : Parameters) (work : List Grid.PID) (counts : Fin work.length → Nat)
    (vals : Ports.Values ℝ p work counts) : ChannelValues ℝ (Ports.outputs p work) :=
  fun o=>Fin.cases (fun i=>weighted p (Ports.logical p work counts vals) i.1 i.2.1) (fun i=>Fin.elim0 i) o

/-- Exact primitive recurrence: every block lane, including padding, remains
in the reduction. Storage casts, accumulation precision and order are explicit. -/
def product {α : Type} (M : Algebra α) (p : Parameters) {jobs extent : Nat}
    (x : Ports.Logical α jobs extent p.width) (j : Fin jobs) (start : Nat)
    (i : TileIndex [p.block,p.width]) : α :=
  let zero := M.cast none .real .fp32 (M.literal none .real 0)
  M.binary (some .fp32) .fp32 .mul
    (if start+i.1.val<x.attention j then probAt zero x j (start+i.1.val) else zero)
    (if start+i.1.val+Ports.window p x j<x.sequence j then valueAt zero x j (start+i.1.val) i.2.1 else zero)

def update {α : Type} (M : Algebra α) (R : SumPrimitive α) (p : Parameters) {jobs extent : Nat}
    (x : Ports.Logical α jobs extent p.width) (j : Fin jobs) (start : Nat)
    (previous : TileIndex [p.width] → α) : TileIndex [p.width] → α :=
  let blockValue := R.apply (some .fp32) ⟨0,by simp⟩ Bool.false (product M p x j start)
  fun i=>M.binary (some .fp32) .real .add (previous i)
    (M.cast (some .fp32) .fp32 .real (blockValue i))

def result {α : Type} (M : Algebra α) (R : SumPrimitive α) (p : Parameters) {jobs extent : Nat}
    (x : Ports.Logical α jobs extent p.width) (j : Fin jobs) (d : Fin p.width) : α :=
  M.cast none .real .fp32
    (Range.foldRange (update M R p x j) 0 (x.attention j) p.block
      (fun _=>M.literal (some .fp32) .real 0) (d,()))

def formula {α : Type} (M : Algebra α) (R : SumPrimitive α) (p : Parameters)
    (work : List Grid.PID) (counts : Fin work.length → Nat) (vals : Ports.Values α p work counts) :
    ChannelValues α (Ports.outputs p work) :=
  fun o=>Fin.cases (fun i=>result M R p (Ports.logical p work counts vals) i.1 i.2.1) (fun i=>Fin.elim0 i) o

#specNonCircular mathematical avoiding [kernel,FP.Structural.exec,Grid.run,outputValues]
end
end VeriTile.Bench.Repairs.TokenAttnMistralGridYChecked.Reference
