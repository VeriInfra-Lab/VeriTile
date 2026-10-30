import bench.tritonbench_g.attention_llama.Proofs.NativeLoop
import bench.tritonbench_g.attention_llama.Proofs.Output
import bench.tritonbench_g.attention_llama.Proofs.HostGrid

namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.Tensor
open VeriTile Triton
open Source
open VeriTile.Bench.Repairs.AttentionLlama HostGrid MaskCoverage
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

structure Inputs (α : Type) (B NH H D : Nat) where
  query : Fin B→Fin NH→Fin H→Fin D→α
  halfKey : Fin B→Fin NH→Nat→Fin D→α
  halfValue : Fin B→Fin NH→Nat→Fin D→α
  byteKey : Fin B→Fin NH→Nat→Fin D→UInt8
  byteValue : Fin B→Fin NH→Nat→Fin D→UInt8

def Inputs.at (x : Inputs α B NH H D) (i : Index B NH H D) : Target.Inputs α D :=
  ⟨x.query i.batch i.head i.query,x.halfKey i.batch i.head,x.halfValue i.batch i.head,
    x.byteKey i.batch i.head,x.byteValue i.batch i.head⟩

def expected (A : Target.Ops α) (C : ScalarABI.Conversions α) (arg : ScalarABI.Argument)
    (c : Config) (x : Inputs α B c.NH c.H c.D) (i : Index B c.NH c.H c.D) : α :=
  Target.output A C (x.at i) arg c.fp8 c.causal c.N c.SP
    (i.query.val/64) (i.query.val%64) i.channel

def offset (c : Config) (i : Index B c.NH c.H c.D) : Nat :=
  i.batch.val*c.soz+i.head.val*c.soh+i.query.val*c.som+i.channel.val*c.son

def initialFor (c : Config) (s : FP.Structural.State (Term α)) (i : Index B c.NH c.H c.D) :
    FP.Structural.State (Term α) :=
  {s with pids:=fun ax=>if ax=0 then i.query.val/64 else
    if ax=1 then i.batch.val*c.NH+i.head.val else 0}

def laneIndex (i : Index B NH H D) : TileIndex [64,D] :=
  (⟨i.query.val%64,Nat.mod_lt _ (by decide)⟩,i.channel,())

structure Binding (c : Config) (s : FP.Structural.State (Term α))
    (x : Inputs α B c.NH c.H c.D) : Prop where
  query : ∀ b h q d, (s.mem c.Q (b.val*c.sqz+h.val*c.sqh+q.val*c.sqm+d.val*c.sqk)).read .fp16=
    .leaf (x.query b h q d)
  halfKey : c.fp8=Bool.false → ∀ b h k d, k<c.N →
    (s.mem c.K (b.val*c.skz+h.val*c.skh+k*c.skn+d.val*c.skk)).read .fp16=.leaf (x.halfKey b h k d)
  halfValue : c.fp8=Bool.false → ∀ b h k d, k<c.N →
    (s.mem c.V (b.val*c.svz+h.val*c.svh+k*c.svk+d.val*c.svn)).read .fp16=.leaf (x.halfValue b h k d)
  byteKey : c.fp8=Bool.true → ∀ b h k d, k<c.N → UInt8.ofFin (Float8E5Bits.ofInt
    ((s.mem c.K (b.val*c.skz+h.val*c.skh+k*c.skn+d.val*c.skk)).read .int))=x.byteKey b h k d
  byteValue : c.fp8=Bool.true → ∀ b h k d, k<c.N → UInt8.ofFin (Float8E5Bits.ofInt
    ((s.mem c.V (b.val*c.svz+h.val*c.svh+k*c.svk+d.val*c.svn)).read .int))=x.byteValue b h k d

theorem row_eq (c : Config) (s : FP.Structural.State (Term α)) (i : Index B c.NH c.H c.D) :
    rows (initialFor c s i) (laneIndex i).1=i.query.val := by
  simp [rows,initialFor,laneIndex]
  omega

theorem base_eq (c : Config) (s : FP.Structural.State (Term α)) (i : Index B c.NH c.H c.D)
    (sz sh : Nat) : base c (initialFor c s i) sz sh=i.batch.val*sz+i.head.val*sh := by
  have h := batch_head_roundtrip c.NH i.batch.val i.head.val i.head.isLt
  simp [base,initialFor,h.1,h.2]

theorem address_eq (c : Config) (s : FP.Structural.State (Term α)) (i : Index B c.NH c.H c.D) :
    address c (initialFor c s i) (laneIndex i)=offset c i := by
  rw [address,base_eq c s i,row_eq c s i]
  rfl

theorem bindings (c : Config) (s : FP.Structural.State (Term α))
    (x : Inputs α B c.NH c.H c.D) (h : Binding c s x) (i : Index B c.NH c.H c.D) :
    Native.Binding c (initialFor c s i) (laneIndex i).1 (x.at i) := by
  constructor
  · intro d
    rw [base_eq c s i,row_eq c s i]
    exact h.query i.batch i.head i.query d
  · intro hf k d hk
    rw [base_eq c s i]
    exact h.halfKey hf i.batch i.head k d hk
  · intro hf k d hk
    rw [base_eq c s i]
    exact h.halfValue hf i.batch i.head k d hk
  · intro hf k d hk
    rw [base_eq c s i]
    exact h.byteKey hf i.batch i.head k d hk
  · intro hf k d hk
    rw [base_eq c s i]
    exact h.byteValue hf i.batch i.head k d hk

theorem block_injective (c : Config) (s : FP.Structural.State (Term α))
    (hinj : Function.Injective (offset (B:=B) c)) (i : Index B c.NH c.H c.D) :
    ActiveInjective c (initialFor c s i) := by
  intro a b ha hb he
  let ia : Index B c.NH c.H c.D := ⟨i.batch,i.head,⟨rows (initialFor c s i) a.1,ha⟩,a.2.1⟩
  let ib : Index B c.NH c.H c.D := ⟨i.batch,i.head,⟨rows (initialFor c s i) b.1,hb⟩,b.2.1⟩
  have hi : ia=ib := hinj (by simpa only [address,base_eq,offset,ia,ib] using he)
  have hr := congrArg (fun j=>j.query.val) hi
  have hd := congrArg (fun j=>j.channel) hi
  apply Prod.ext
  · apply Fin.ext
    simp only [ia,ib,rows] at hr
    omega
  · exact Prod.ext hd (Subsingleton.elim _ _)

#axiomsClean bindings
#axiomsClean block_injective
#specNonCircular expected avoiding [FP.Structural.exec,FP.Structural.run,Source.accumulated]
end
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.Tensor
