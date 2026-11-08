import bench.tritonbench_g.token_attn_mistral.Kernels
import bench.tritonbench_g.token_attn_mistral.Proofs.Stores
import VeriTile.Triton.Float.ProgramIO

namespace VeriTile.Bench.Repairs.TokenAttnMistralGridYChecked.Ports
open VeriTile Triton FP.Structural TokenAttnMistral
noncomputable section

/-- A finite port extent derived from this submitted work list. This is not an
upper bound imposed on runtime metadata; every metadata snapshot has one. -/
def span (p : Parameters) (work : List Grid.PID) (counts : Fin work.length → Nat) : Nat :=
  (∑ j, counts j)+p.block

def inputs (p : Parameters) (work : List Grid.PID) (counts : Fin work.length → Nat) : List IOChannel :=
  [⟨.int,[work.length],p.length.cast⟩,⟨.int,[work.length],p.start.cast⟩,
   ⟨.int,[work.length],p.request.cast⟩,⟨.int,[work.length],p.attentionLength.cast⟩,
   ⟨.int,[work.length,span p work counts],p.tokens.cast⟩,
   ⟨.fp32,[work.length,span p work counts],p.prob⟩,
   ⟨.fp32,[work.length,span p work counts,p.width],p.value⟩]
def outputs (p : Parameters) (work : List Grid.PID) : List IOChannel :=
  [⟨.fp32,[work.length,p.width],p.output⟩]

structure Logical (α : Type) (jobs extent width : Nat) where
  sequenceWord : Fin jobs → Int
  startWord : Fin jobs → Int
  requestWord : Fin jobs → Int
  attentionWord : Fin jobs → Int
  tokenWord : Fin jobs → Fin extent → Int
  probability : Fin jobs → Fin extent → α
  value : Fin jobs → Fin extent → Fin width → α

def Logical.sequence {α : Type} {jobs extent width : Nat} (x : Logical α jobs extent width)
    (j : Fin jobs) : Nat := (x.sequenceWord j).toNat
def Logical.start {α : Type} {jobs extent width : Nat} (x : Logical α jobs extent width)
    (j : Fin jobs) : Nat := (x.startWord j).toNat
def Logical.request {α : Type} {jobs extent width : Nat} (x : Logical α jobs extent width)
    (j : Fin jobs) : Nat := (x.requestWord j).toNat
def Logical.attention {α : Type} {jobs extent width : Nat} (x : Logical α jobs extent width)
    (j : Fin jobs) : Nat := (x.attentionWord j).toNat
def Logical.token {α : Type} {jobs extent width : Nat} (x : Logical α jobs extent width)
    (j : Fin jobs) (n : Fin extent) : Nat := (x.tokenWord j n).toNat

abbrev Values (α : Type) (p : Parameters) (work : List Grid.PID) (counts : Fin work.length → Nat) :=
  ChannelValues α (inputs p work counts)

def logical {α : Type} (p : Parameters) (work : List Grid.PID) (counts : Fin work.length → Nat)
    (vals : Values α p work counts) : Logical α work.length (span p work counts) p.width where
  sequenceWord := fun j=>vals ⟨0,by simp [inputs]⟩ (j,())
  startWord := fun j=>vals ⟨1,by simp [inputs]⟩ (j,())
  requestWord := fun j=>vals ⟨2,by simp [inputs]⟩ (j,())
  attentionWord := fun j=>vals ⟨3,by simp [inputs]⟩ (j,())
  tokenWord := fun j n=>vals ⟨4,by simp [inputs]⟩ (j,n,())
  probability := fun j n=>vals ⟨5,by simp [inputs]⟩ (j,n,())
  value := fun j n d=>vals ⟨6,by simp [inputs]⟩ (j,n,d,())

def window {α : Type} (p : Parameters) {jobs extent width : Nat}
    (x : Logical α jobs extent width) (j : Fin jobs) := max (x.sequence j-p.window) 0

def visited (p : Parameters) (attention n : Nat) : Prop :=
  0<p.block ∧ n/p.block*p.block<attention

/-- The inherited nonnegative domain is explicit on actual signed words.
Inactive token slots impose no condition; all their words remain in Logical. -/
structure Admissible {α : Type} {jobs extent width : Nat} (p : Parameters)
    (counts : Fin jobs → Nat) (x : Logical α jobs extent width) : Prop where
  attention : ∀ j,x.attentionWord j=(counts j : Int)
  sequence : ∀ j,0≤x.sequenceWord j
  start : ∀ j,0≤x.startWord j
  request : ∀ j,0≤x.requestWord j
  token : ∀ j (n : Fin extent),visited p (counts j) n.val →
    n.val+window p x j<x.sequence j → 0≤x.tokenWord j n

theorem Admissible.counts {α : Type} {jobs extent width : Nat} {p : Parameters}
    {counts : Fin jobs → Nat} {x : Logical α jobs extent width} (h : Admissible p counts x) :
    ∀ j,x.attention j=counts j := by
  intro j
  simpa only [Logical.attention,Int.toNat_natCast] using congrArg Int.toNat (h.attention j)

def Domain {α : Type} (p : Parameters) (work : List Grid.PID) (counts : Fin work.length → Nat)
    (vals : Values α p work counts) : Prop := Admissible p counts (logical p work counts vals)

/-- Simultaneous finite inputs preserve arbitrary read/read aliases. The
probability and page/value masks deliberately differ on the final block. -/
def memory {α : Type} (p : Parameters) (work : List Grid.PID) (counts : Fin work.length → Nat) : MemoryIO α where
  inputs := inputs p work counts
  outputs := outputs p work
  scratch := []
  buffers := [p.prob,p.value,p.output,p.tokens.cast,p.request.cast,p.start.cast,p.length.cast,p.attentionLength.cast]
  read := fun i vals=>Fin.cases (fun j=>work[j.1].1*p.sequenceStride)
    (Fin.cases (fun j=>work[j.1].1*p.attentionStartStride)
      (Fin.cases (fun j=>work[j.1].1*p.requestStride)
        (Fin.cases (fun j=>work[j.1].1*p.attentionLengthStride)
          (Fin.cases (fun j=>(logical p work counts vals).request j.1*p.requestBatchStride+
              (window p (logical p work counts vals) j.1+j.2.1.val)*p.requestTokenStride)
            (Fin.cases (fun j=>work[j.1].2*p.probHeadStride+
                ((logical p work counts vals).start j.1+j.2.1.val)*p.probTokenStride)
              (Fin.cases (fun j=>work[j.1].2/p.kvGroup*p.valueHeadStride+j.2.2.1.val*p.valueDimStride+
                (logical p work counts vals).token j.1 j.2.1*p.valueBatchStride)
                (fun i=>Fin.elim0 i))))))) i
  inputMask := fun i vals=>Fin.cases (fun _=>True)
    (Fin.cases (fun _=>True) (Fin.cases (fun _=>True) (Fin.cases (fun _=>True)
      (Fin.cases (fun j=>visited p (counts j.1) j.2.1.val ∧
          j.2.1.val+window p (logical p work counts vals) j.1<(logical p work counts vals).sequence j.1)
        (Fin.cases (fun j=>0<p.block ∧ j.2.1.val<counts j.1)
          (Fin.cases (fun j=>visited p (counts j.1) j.2.1.val ∧
              j.2.1.val+window p (logical p work counts vals) j.1<(logical p work counts vals).sequence j.1)
            (fun i=>Fin.elim0 i))))))) i
  write := fun o=>Fin.cases (fun _ j=>Grid.offset p work[j.1] (j.2.1,())) (fun i=>Fin.elim0 i) o
  outputMask := fun _ _ _=>True
  scratchWrite := fun i=>Fin.elim0 i
  scratchMask := fun i=>Fin.elim0 i

end
end VeriTile.Bench.Repairs.TokenAttnMistralGridYChecked.Ports

namespace VeriTile.Bench.Repairs.TokenAttnMistralGridYChecked.Reference
open VeriTile Triton FP.Structural TokenAttnMistral
noncomputable section

/-- Pure channel lookup; defaults are used only outside the finite port. -/
def probAt {α : Type} (zero : α) {jobs extent width : Nat}
    (x : Ports.Logical α jobs extent width) (j : Fin jobs) (n : Nat) : α :=
  if h:n<extent then x.probability j ⟨n,h⟩ else zero
def valueAt {α : Type} (zero : α) {jobs extent width : Nat}
    (x : Ports.Logical α jobs extent width) (j : Fin jobs) (n : Nat) (d : Fin width) : α :=
  if h:n<extent then x.value j ⟨n,h⟩ d else zero

end
end VeriTile.Bench.Repairs.TokenAttnMistralGridYChecked.Reference
