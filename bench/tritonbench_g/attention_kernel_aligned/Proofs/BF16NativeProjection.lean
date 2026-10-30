import bench.tritonbench_g.attention_kernel_aligned.Proofs.BF16NativeStaging
import bench.tritonbench_g.attention_kernel_aligned.Proofs.AttentionKernelAligned_BF16_Whole_BF16Binding
namespace VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.Native
open VeriTile Triton FP.Structural Source Whole VeriTile.Bench.Repairs.AttentionKernelAligned.Full
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000
abbrev Word := BitVec 32

/-- Raw words are embedded literally. Numerical contracts apply only to scalar
instructions and fixed-shape MMA ports, never to a tensor/result/run. -/
structure Instructions where
  ops : ISA Word
  bitsIdentity : ∀ w,ops.bits w=w

structure InstructionAgreement (A B : Instructions) : Prop where
  literal32 : ∀ x,A.ops.literal32 x=B.ops.literal32 x
  add32 : ∀ a b,A.ops.add32 a b=B.ops.add32 a b
  sub32 : ∀ a b,A.ops.sub32 a b=B.ops.sub32 a b
  mul32 : ∀ a b,A.ops.mul32 a b=B.ops.mul32 a b
  maxNum32 : ∀ a b,A.ops.maxNum32 a b=B.ops.maxNum32 a b
  divFull32 : ∀ a b,A.ops.divFull32 a b=B.ops.divFull32 a b
  exp2ApproxFTZ : ∀ a,A.ops.exp2ApproxFTZ a=B.ops.exp2ApproxFTZ a
  narrowBF16 : ∀ a,A.ops.narrowBF16 a=B.ops.narrowBF16 a
  widenBF16 : ∀ a,A.ops.widenBF16 a=B.ops.widenBF16 a
  fma32 : ∀ a b c,A.ops.fma32 a b c=B.ops.fma32 a b c
  fmaBF16 : ∀ a b c,A.ops.fmaBF16 a b c=B.ops.fmaBF16 a b c
  mmaBF32 : ∀ r c a b seed,A.ops.mmaBF32 r c a b seed=B.ops.mmaBF32 r c a b seed

def InstructionAgreement.transport (A B : Instructions) (h : InstructionAgreement A B) :
    LocalTransport id A.ops B.ops where
  bits:=by intro w;simp [A.bitsIdentity,B.bitsIdentity]
  literal32:=h.literal32
  add32:=h.add32
  sub32:=h.sub32
  mul32:=h.mul32
  maxNum32:=h.maxNum32
  divFull32:=h.divFull32
  exp2ApproxFTZ:=h.exp2ApproxFTZ
  narrowBF16:=h.narrowBF16
  widenBF16:=h.widenBF16
  fma32:=h.fma32
  fmaBF16:=h.fmaBF16
  mmaBF32:=h.mmaBF32

def inject : (d : TileDType) → Value Word d → Value Expr d
  | .real,w | .fp32,w | .fp16,w | .bf16,w | .f8e4,w | .f8e5,w=>fun _=>.ready (.bits w)
  | .nat,n=>n | .int,n=>n | .bool,n=>n | .ptr,n=>n | .blockPtr,n=>n

def injectState (s : State Word) : State Expr where
  mem:=fun r o=>match s.mem r o with | .mk d v=>.mk d (inject d v)
  regs:=fun _ _ _=>none
  pids:=s.pids
  numPids:=s.numPids
  undef:=fun d r o=>inject d (s.undef d r o)

def project (I : Instructions) (owner : Fin 5) : (d : TileDType) → Value Expr d → Value Word d
  | .real,g | .fp32,g | .fp16,g | .bf16,g | .f8e4,g | .f8e5,g=>(lower (g owner)).eval I.ops
  | .nat,n=>n | .int,n=>n | .bool,n=>n | .ptr,n=>n | .blockPtr,n=>n
def projectCell (I : Instructions) (owner : Fin 5) : Cell Expr → Cell Word
  | .mk d v=>.mk d (project I owner d v)

theorem project_inject (I : Instructions) (owner : Fin 5) (d : TileDType) (v : Value Word d) :
    project I owner d (inject d v)=v := by
  cases d <;> simp [project,inject,lower,Graph.eval,I.bitsIdentity]

theorem injectState_project (I : Instructions) (owner : Fin 5) (s : State Word) (r : RegionName) (o : Nat) :
    projectCell I owner ((injectState s).mem r o)=s.mem r o := by
  change projectCell I owner (match s.mem r o with | .mk d v=>.mk d (inject d v))=_
  generalize s.mem r o=cell
  cases cell with | mk d v=>simp only [projectCell,project_inject]

def Feature (p : Parameters) (off d : Nat) : Prop :=
  d<p.head ∧ ∃ b<p.batches,∃ h<p.heads,∃ n<p.length,off=logicalAddress p b h n d
def featureAt (p : Parameters) (off : Nat) : Nat := by
  classical
  exact if h : ∃ d,Feature p off d then Classical.choose h else 0

theorem featureAt_output (p : Parameters) (guard : Guard p) (b h n d : Nat)
    (hb : b<p.batches) (hh : h<p.heads) (hn : n<p.length) (hd : d<p.head) :
    featureAt p (logicalAddress p b h n d)=d := by
  have he : ∃ e,Feature p (logicalAddress p b h n d) e := ⟨d,hd,b,hb,h,hh,n,hn,rfl⟩
  rw [featureAt,dif_pos he]
  obtain ⟨hed,bb,hbb,hh',hhh,nn,hnn,heq⟩:=Classical.choose_spec he
  have ne : Whole.Nonempty p := ⟨Nat.zero_lt_of_lt hb,Nat.zero_lt_of_lt hh,Nat.zero_lt_of_lt hn⟩
  exact (outputInjective p (guard.layout ne) b h n d bb hh' nn _ hb hh hn hd hbb hhh hnn hed heq).2.2.2.symm

def ownerAt (p : Parameters) (r : RegionName) (off : Nat) : Fin 5 :=
  if r=p.output then physicalOwner (featureAt p off) else ⟨0,by decide⟩
def projectState (I : Instructions) (p : Parameters) (before : State Word) (after : State Expr) : State Word :=
  {before with mem:=fun r off=>projectCell I (ownerAt p r off) (after.mem r off)}

def encode16 : Cell Word → BitVec 16
  | .mk .bf16 w=>BitVec.ofNat 16 w.toNat
  | _=>0

def outputGraph (p : Parameters) (selected : Selected p) (s : State Word) (b h n d : Nat) (hd : d<p.head) : Graph :=
  reference p (start (grid p) (n/p.blockM,b*p.heads+h,0) (injectState s)) (lane p selected n d hd)
def outputWord (I : Instructions) (p : Parameters) (selected : Selected p) (s : State Word) (b h n d : Nat) (hd : d<p.head) : Word :=
  (outputGraph p selected s b h n d hd).eval I.ops

def RawInputs (p : Parameters) (s : State Word) : Prop :=
  ∀ r∈[p.q,p.k,p.v,p.bias],∀ off,∃ w : BitVec 16,s.mem r off=.mk .bf16 (w.zeroExtend 32)

def rawCell : Cell Word → Word
  | .mk .real w | .mk .fp32 w | .mk .fp16 w | .mk .bf16 w | .mk .f8e4 w | .mk .f8e5 w=>w
  | _=>0

/-- Current finite byte buffer binding: arbitrary half input words and caller-owned half output words, without a computed value or execution premise. -/
structure WordBinding (p : Parameters) (storage : TensorStorage p) (pl : Placement p storage)
    (flat : Fin pl.arena → BitVec 8) (s : State Word) : Prop where
  inputs : ∀ r,r≠Buffer.output → ∀ off<extent storage r,∀ k<2,∀ i : Fin pl.arena,
    i.val=pl.base r+2*off+k →
      flat i=BitVec.ofNat 8 ((rawCell (s.mem (region p r) off)).toNat/256^k)
  output : FlatBinding p storage pl (fun _=>encode16) flat s

#axiomsClean injectState_project
#axiomsClean featureAt_output
#specNonCircular outputGraph avoiding [invoke,runPrograms,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.Native
