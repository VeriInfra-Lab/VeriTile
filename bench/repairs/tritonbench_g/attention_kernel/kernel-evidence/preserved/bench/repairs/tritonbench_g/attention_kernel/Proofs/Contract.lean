import bench.repairs.tritonbench_g.attention_kernel.Proofs.Machine
namespace VeriTile.Bench.Repairs.AttentionKernel.Contract
open VeriTile Triton
open Whole001 Source Whole Machine
noncomputable section

/-- Ordinary classified wrapper arguments; descriptors contain API metadata
and storage identity, with no floating values or kernel result. -/
structure Arguments where
  query : Descriptor
  key : Descriptor
  value : Descriptor
  bias : Descriptor
  output : Descriptor
  /-- Actual FP32 launch word, bound by ordinary Python scalar/JIT conversion.
  This observation preserves signed zero; it is not a numerical-result premise. -/
  scaleWord : BitVec 32
  numWarps : Nat
  numStages : Nat
  tensors : Bool
  scalar : Bool
  integers : Bool
  strided : Bool
  supportedDTypes : Bool
  readSpans : List (Nat × Nat)
  outputSpan : Nat × Nat

def Metadata (p : Parameters) (a : Arguments) : Prop :=
  a.tensors=Bool.true ∧ a.scalar=Bool.true ∧ a.integers=Bool.true ∧ a.strided=Bool.true ∧ a.supportedDTypes=Bool.true ∧
  (p.bm∈[16,32,64,128] ∧ p.bn∈[16,32,64,128] ∧ p.dim∈[16,32,64,128]) ∧
  (a.numWarps∈[4,8] ∧ 1≤a.numStages ∧ a.numStages≤4) ∧
  |p.scale|≤3.4028234663852886e38 ∧ (p.length+p.bn-1)/p.bn≤128 ∧
  (a.query.shape=[p.batches,p.heads,p.length,p.dim] ∧ a.key.shape=a.query.shape ∧
    a.value.shape=a.query.shape ∧ a.output.shape=a.query.shape ∧ a.bias.shape=[p.batches,p.heads,p.length,128]) ∧
  (a.query.strides=[p.q.batch,p.q.head,p.q.row,p.q.dim] ∧ a.key.strides=[p.k.batch,p.k.head,p.k.row,p.k.dim] ∧
    a.value.strides=[p.v.batch,p.v.head,p.v.row,p.v.dim] ∧ a.output.strides=[p.o.batch,p.o.head,p.o.row,p.o.dim] ∧
    a.bias.strides=[p.bias.batch,p.bias.head,p.bias.row,p.bias.dim]) ∧
  a.bias.contiguous=Bool.true ∧
  (a.query.region=p.Q ∧ a.key.region=p.K ∧ a.value.region=p.V ∧ a.bias.region=p.Bias ∧ a.output.region=p.Out) ∧
  (∀ t∈[a.query,a.key,a.value,a.bias,a.output],t.dtype=p.storage) ∧
  (∀ t∈[a.query,a.key,a.value,a.bias,a.output],t.cuda=Bool.true ∧ t.device=a.query.device) ∧
  (∀ t∈[a.query,a.key,a.value,a.bias,a.output],t.negative=Bool.false ∧ t.conjugate=Bool.false)

/-- Each conjunct is an actual nonempty Python guard. Shape/stride/data_ptr
observations in these predicates are bound by ordinary Tensor API semantics. -/
def Active (p : Parameters) (b : Bases) : Prop :=
  (∀ s∈[p.q,p.k,p.v,p.o],s.row=p.dim ∧ s.dim=1 ∧ p.length*p.dim≤s.head ∧ p.heads*s.head≤s.batch) ∧
  envelope p<2147483648 ∧ p.batches*p.heads≤65535 ∧
  (∀ base∈bases b,base%2=0 ∧ base+2*envelope p<18446744073709551616)

def Separate (a : Arguments) : Prop :=
  ∀ span∈a.readSpans,a.outputSpan.1+a.outputSpan.2≤span.1 ∨ span.1+span.2≤a.outputSpan.1

def InputContract (p : Parameters) (b : Bases) (a : Arguments) : Prop :=
  Metadata p a ∧ (Live p → Active p b ∧ Separate a)

inductive Error where | arguments | launch
  deriving DecidableEq

/-- Semantic guard interpreter on classified API observations. No source
execution, numerical equality or output value occurs in a guard. -/
def check (p : Parameters) (b : Bases) (a : Arguments) : Except Error Unit := by
  classical
  exact if Metadata p a then
    if Live p then if Active p b ∧ Separate a then .ok () else .error .launch else .ok ()
  else .error .arguments

theorem check_iff (p : Parameters) (b : Bases) (a : Arguments) :
    check p b a=.ok () ↔ InputContract p b a := by
  classical
  by_cases live : Live p <;> simp [check,InputContract,live,ite_eq_iff]

def abi (p : Parameters) (a : Arguments) (h : Metadata p a) : InputABI p := by
  rcases h with ⟨_,_,_,_,_,blocks,launch,scale,keys,shapes,strides,contiguous,regions,types,device,ordinary⟩
  exact ⟨a.query,a.key,a.value,a.bias,a.output,a.numWarps,a.numStages,
    blocks,launch,scale,keys,shapes,strides,contiguous,regions,types,device,ordinary⟩

theorem domain (p : Parameters) (b : Bases) (a : Arguments) (h : InputContract p b a) : Domain p b := by
  have accepted :=  abi p a h.1
  have bp : 0<p.bm := by have hb:=accepted.blocks.1;simp only [List.mem_cons,List.not_mem_nil,or_false] at hb;omega
  have bn : 0<p.bn := by have hb:=accepted.blocks.2.1;simp only [List.mem_cons,List.not_mem_nil,or_false] at hb;omega
  have dim : 0<p.dim := by have hb:=accepted.blocks.2.2;simp only [List.mem_cons,List.not_mem_nil,or_false] at hb;omega
  refine ⟨?_,?_⟩
  · intro live
    have active := (h.2 live).1
    exact ⟨bp,bn,dim,accepted.keyBlocks,active.2.1,active.2.2.1,
      fun s hs => let v:=active.1 s hs;⟨v.1,v.2.1,v.2.2.1,v.2.2.2⟩⟩
  · intro live
    exact (h.2 live).1.2.2.2

theorem positive_block (p : Parameters) (b : Bases) (a : Arguments) (h : InputContract p b a) : 0<p.bm := by
  have hb:=(abi p a h.1).blocks.1
  simp only [List.mem_cons,List.not_mem_nil,or_false] at hb
  omega

#specNonCircular InputContract avoiding [Whole.invoke,FP.Structural.exec]
#axiomsClean check_iff
#axiomsClean domain
#axiomsClean positive_block
end
end VeriTile.Bench.Repairs.AttentionKernel.Contract
