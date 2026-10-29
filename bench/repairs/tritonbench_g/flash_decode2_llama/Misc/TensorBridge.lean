import bench.repairs.tritonbench_g.flash_decode2_llama.Correct
import bench.repairs.tritonbench_g.flash_decode2_llama.Misc.NativeCorrect
import bench.repairs.tritonbench_g.flash_decode2_llama.FPEquiv
import bench.repairs.tritonbench_g.flash_decode2_llama.Proofs.KernelEndpoints
import bench.repairs.tritonbench_g.flash_decode2_llama.Proofs.KernelOrder
import bench.repairs.tritonbench_g.flash_decode2_llama.Proofs.KernelSafety

/-! Complete actual-Tensor applications, pointers and logical readback. -/
namespace VeriTile.Bench.Repairs.FlashDecode2Llama.TensorBridge
open VeriTile.Triton FP.Structural FlashDecode2 KernelModel KernelTensor
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 8000000

theorem flash_decode2_llama_pointers (c : Call) (t : Arguments c) (heap : Heap)
    (h : InputContract c (lengths c heap)) : NativeSafety.AllAccess c.val :=
  KernelSafety.pointers c t heap h

theorem flash_decode2_llama_mid_read (c : Call) (t : Arguments c) (heap : Heap)
    (h : InputContract c (lengths c heap)) (i : Nat) (inside : i<c.val.mid.metadata.numel) :
    (⟨word c heap "M" i⟩ : VeriTile.Torch.FP32.Word) =
      VeriTile.Torch.FP32.logicalValue (readWord heap) t.mid.tensor t.mid.laws i :=
  KernelTensor.mid_read c t heap h i inside

theorem flash_decode2_llama_log_read (c : Call) (t : Arguments c) (heap : Heap)
    (h : InputContract c (lengths c heap)) (i : Nat) (inside : i<c.val.log.metadata.numel) :
    (⟨word c heap "E" i⟩ : VeriTile.Torch.FP32.Word) =
      VeriTile.Torch.FP32.logicalValue (readWord heap) t.log.tensor t.log.laws i :=
  KernelTensor.log_read c t heap h i inside

theorem flash_decode2_llama_correct (interpret : Nat → ℝ) (c : Call) (t : Arguments c) (heap : Heap)
    (other : State (Option ℝ)) (h : InputContract c (lengths c heap))
    (order : List Nat) (all : order.Perm (List.range (rows c))) :
    ∃ after,run Reference.ideal c order (KernelReal.entry interpret c heap other) = some after ∧
      (∀ b head d,b<Checked.batches c.val → head<Checked.heads c.val → d<Checked.width c.val →
        (after.mem "O" (t.output.tensor.physicalIndex ((b*Checked.heads c.val+head)*Checked.width c.val+d))).read .real =
          some (KernelReal.target interpret c heap (b*Checked.heads c.val+head) d)) ∧
      (∀ r i,r≠("O" : RegionName) ∨ size c ≤ i →
        after.mem r i=(KernelReal.entry interpret c heap other).mem r i) :=
  KernelEndpoints.real interpret c t heap other h order all

theorem flash_decode2_llama_fp32 (p : KernelNative.Ops) (c : Call) (t : Arguments c) (heap : Heap)
    (other : State Native.Term) (h : InputContract c (lengths c heap))
    (order : List Nat) (all : order.Perm (List.range (rows c))) :
    ∃ after,run Native.algebra c order (KernelNative.entry c heap other) = some after ∧
      (∀ b head d,b<Checked.batches c.val → head<Checked.heads c.val → d<Checked.width c.val →
        Native.observe (KernelNative.lower p) (KernelNative.schedule c) after "O"
          (t.output.tensor.physicalIndex ((b*Checked.heads c.val+head)*Checked.width c.val+d)) =
        KernelNative.target p c (lengths c heap) (word c heap "E") (word c heap "M")
          (b*Checked.heads c.val+head) d) ∧
      (∀ r i,r≠("O" : RegionName) ∨ size c ≤ i →
        after.mem r i=(KernelNative.entry c heap other).mem r i) :=
  KernelEndpoints.native p c t heap other h order all

theorem flash_decode2_llama_source (c : Call) :
    kernel c.val.masked (Checked.heads c.val) (Checked.capacity c.val) (Checked.width c.val)
      c.val.blockSeq (Checked.block c.val) =
    llama (Checked.heads c.val) (Checked.capacity c.val) (Checked.width c.val)
      c.val.blockSeq (Checked.block c.val) :=
  KernelModel.source c

theorem flash_decode2_llama_grid_order {α : Type} [Inhabited α] (A : Algebra α) (c : Call)
    (first second : List Nat) (firstAll : first.Perm (List.range (rows c)))
    (secondAll : second.Perm (List.range (rows c))) (before : State α) :
    ∃ a b,run A c first before=some a ∧ run A c second before=some b ∧ a.mem=b.mem :=
  KernelModel.order A c first second firstAll secondAll before

theorem flash_decode2_llama_grid_coverage (c : Call) (b h : Nat) (hb : b < Checked.batches c.val)
    (hh : h < Checked.heads c.val) :
    b*Checked.heads c.val+h ∈ List.range (rows c) ∧
    (b*Checked.heads c.val+h)/Checked.heads c.val=b ∧
    (b*Checked.heads c.val+h)%Checked.heads c.val=h :=
  KernelModel.coverage c b h hb hh

theorem flash_decode2_llama_empty {α : Type} [Inhabited α] (A : Algebra α) (c : Call)
    (before : State α) (h : Checked.batches c.val = 0 ∨ Checked.heads c.val = 0) :
    run A c (List.range (rows c)) before = some before :=
  KernelModel.empty A c before h

theorem flash_decode2_llama_offsets {α : Type} [Inhabited α] (observe : RegionName → Nat → Native.Word → α)
    (c : Call) (heap : Heap) (other : State α) (h : InputContract c (lengths c heap))
    (row j : Nat) (hr : row<rows c)
    (hj : j < iterations c (lengths c heap) row) (k : TileIndex [Checked.block c.val]) :
    let s := Grid.setup (Checked.batches c.val) (Checked.heads c.val) row (lift observe c heap other)
    s.pids 0<Checked.batches c.val ∧ s.pids 1<Checked.heads c.val ∧
    j<Checked.capacity c.val ∧
    eoff s (Checked.heads c.val) (Checked.capacity c.val) j<(Checked.geometry c.val).logSize ∧
    voff s (Checked.heads c.val) (Checked.capacity c.val) (Checked.width c.val) j k<(Checked.geometry c.val).midSize ∧
    ooff s (Checked.heads c.val) (Checked.width c.val) k<size c ∧
    4*eoff s (Checked.heads c.val) (Checked.capacity c.val) j<2^63 ∧
    4*voff s (Checked.heads c.val) (Checked.capacity c.val) (Checked.width c.val) j k<2^63 ∧
    4*ooff s (Checked.heads c.val) (Checked.width c.val) k<2^63 :=
  KernelSafety.offsets observe c heap other h row j hr hj k

theorem flash_decode2_llama_signed_length (c : Call) (heap : Heap) (b : Nat) :
    ((word c heap "L" b).signExtend 64).toInt=lengths c heap b :=
  KernelTensor.signed_length c heap b

theorem flash_decode2_llama_length_add (c : Call) (heap : Heap) (h : InputContract c (lengths c heap))
    (row : Nat) (hr : row<rows c) :
    (BitVec.ofNat 64 ((lengths c heap (row/Checked.heads c.val)).toNat+c.val.blockSeq-1)).toInt =
      Int.ofNat ((lengths c heap (row/Checked.heads c.val)).toNat+c.val.blockSeq-1) :=
  KernelSafety.length_add c heap h row hr

theorem flash_decode2_llama_maximal_q_once (c : Call) (heap : Heap) (h : InputContract c (lengths c heap))
    (row : Nat) (hr : row<rows c) (maximal : c.val.blockSeq=2^31-1) :
    iterations c (lengths c heap) row=1 :=
  KernelSafety.maximal_q_once c heap h row hr maximal

open Native KernelNative in
theorem flash_decode2_llama_native_recurrence (p : Ops) (c : Call) (heap : KernelTensor.Heap) (other : State Term)
    (row n : Nat) :
    let s := Grid.setup (Checked.batches c.val) (Checked.heads c.val) row (entry c heap other)
    let a := scan p c (fun j => KernelTensor.word c heap "E" (row*Checked.capacity c.val+j))
      (fun j d => KernelTensor.word c heap "M"
        (row*Checked.capacity c.val*Checked.width c.val+j*Checked.width c.val+d)) n
    eval (lower p) (schedule c) (values algebra s c.val.masked (Checked.heads c.val)
      (Checked.capacity c.val) (Checked.width c.val) (Checked.block c.val) n).m = a.maximum ∧
    eval (lower p) (schedule c) (values algebra s c.val.masked (Checked.heads c.val)
      (Checked.capacity c.val) (Checked.width c.val) (Checked.block c.val) n).z = a.denominator ∧
    ∀ k,eval (lower p) (schedule c) ((values algebra s c.val.masked (Checked.heads c.val)
      (Checked.capacity c.val) (Checked.width c.val) (Checked.block c.val) n).a k) = a.numerator (lane k) :=
  KernelNative.recurrence p c heap other row n

abbrev flash_decode2_llama_reference := KernelModel.realTarget

abbrev flash_decode2_llama_native_reference := KernelNative.target

#axiomsClean flash_decode2_llama_correct
#axiomsClean flash_decode2_llama_fp32
#axiomsClean flash_decode2_llama_source
#axiomsClean flash_decode2_llama_grid_order
#axiomsClean flash_decode2_llama_grid_coverage
#axiomsClean flash_decode2_llama_empty
#axiomsClean flash_decode2_llama_offsets
#axiomsClean flash_decode2_llama_signed_length
#axiomsClean flash_decode2_llama_length_add
#axiomsClean flash_decode2_llama_maximal_q_once
#axiomsClean flash_decode2_llama_native_recurrence

#axiomsClean flash_decode2_llama_pointers
#axiomsClean flash_decode2_llama_mid_read
#axiomsClean flash_decode2_llama_log_read
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FlashDecode2Llama.TensorBridge
