import bench.repairs.tritonbench_g.attn_fwd_causal.Proofs.Coverage
namespace VeriTile.Bench.Repairs.AttnFwdCausal.Whole
open VeriTile Triton FP.Structural Source
noncomputable section
set_option maxHeartbeats 4000000

inductive DType where | fp32 | fp16 deriving DecidableEq
structure Descriptor where
  (object device : Nat)
  region : RegionName
  dtype : DType
  (shape strides : List Nat)
  (negative conjugate : Bool)

/-- Guarded ordinary input ABI. No returned tensor or output values are assumed. -/
structure InputABI (p : Parameters) where
  (query key value queryScale keyScale : Descriptor)
  shapes : query.shape=[p.batches,p.heads,p.length,128] ∧ key.shape=query.shape ∧ value.shape=query.shape ∧
    queryScale.shape=[p.batches,p.heads,(p.length+127)/128] ∧
    keyScale.shape=[p.batches,p.heads,(p.length+63)/64]
  strides : 0<p.batches → 0<p.heads → 0<p.length → query.strides=[p.q.batch,p.q.head,p.q.row,p.q.dim] ∧
    key.strides=[p.k.batch,p.k.head,p.k.row,p.k.dim] ∧ value.strides=[p.v.batch,p.v.head,p.v.row,p.v.dim]
  scaleStrides : 0<p.batches → 0<p.heads → 0<p.length →
    queryScale.strides=[p.heads*((p.length+127)/128),(p.length+127)/128,1] ∧
    keyScale.strides=[p.heads*((p.length+63)/64),(p.length+63)/64,1]
  regions : query.region=p.Q ∧ key.region=p.K ∧ value.region=p.V ∧ queryScale.region=p.QS ∧ keyScale.region=p.KS
  types : ∀ t∈[query,key,value,queryScale,keyScale],t.dtype=.fp32
  device : ∀ t∈[query,key,value,queryScale,keyScale],t.device=query.device
  ordinary : ∀ t∈[query,key,value,queryScale,keyScale],t.negative=Bool.false ∧ t.conjugate=Bool.false
  disjoint : 0<p.batches → 0<p.heads → 0<p.length → [p.Q,p.K,p.V,p.QS,p.KS].Nodup

variable {α : Type} [Inhabited α]

/-- Ordinary allocation contract for torch.empty_like(q,dtype=half) and the
unused FP32 M allocation. Their new cells are arbitrary; input cells are framed.
The actual returned output strides are the p.o fields passed to the JIT kernel. -/
structure Allocation (p : Parameters) (abi : InputABI p) (before : State α) where
  state : State α
  (output unusedM : Descriptor)
  outputRegion : output.region=p.Out
  outputShape : output.shape=abi.query.shape
  outputType : output.dtype=.fp16
  outputDevice : output.device=abi.query.device
  outputStrides : 0<p.batches → 0<p.heads → 0<p.length → output.strides=[p.o.batch,p.o.head,p.o.row,p.o.dim]
  outputOrdinary : output.negative=Bool.false ∧ output.conjugate=Bool.false
  mShape : unusedM.shape=[p.batches,p.heads,p.length]
  mType : unusedM.dtype=.fp32
  mDevice : unusedM.device=abi.query.device
  fresh : p.Out≠unusedM.region ∧ ∀ t∈[abi.query,abi.key,abi.value,abi.queryScale,abi.keyScale],
    output.object≠t.object ∧ unusedM.object≠t.object ∧ p.Out≠t.region ∧ unusedM.region≠t.region
  frame : ∀ r off,r≠p.Out → r≠unusedM.region → state.mem r off=before.mem r off

structure Returned (α : Type) where
  state : State α
  tensor : Descriptor

def invoke (M : Program → Algebra α) (dot : Program → Dot α) (p : Parameters)
    (abi : InputABI p) (before : State α) (alloc : Allocation p abi before)
    (work : List Program) : Option (Returned α) :=
  (runPrograms M dot p (grid p) work alloc.state).map fun t=>⟨t,alloc.output⟩

def logicalAddress (p : Parameters) (b h n d : Nat) : Nat :=
  b*p.o.batch+h*p.o.head+n*p.o.row+d*p.o.dim

theorem physical_address (p : Parameters) (s : State α) (b h n d : Nat)
    (hh : h<p.heads) (hd : d<128) :
    address p (start (grid p) (n/128,b*p.heads+h,0) s) (lane n d hd)=logicalAddress p b h n d := by
  have hH : 0<p.heads := by omega
  have hdiv : (b*p.heads+h)/p.heads=b := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ hH,Nat.div_eq_of_lt hh]
    omega
  have hmod : (b*p.heads+h)%p.heads=h := by simp [Nat.add_mod,Nat.mod_eq_of_lt hh]
  have hn : n/128*128+n%128=n := by omega
  simp [address,base,start,ids,lane,logicalAddress,hdiv,hmod,hn]

/-- Entire returned output is covered, including a short last query block and
all 128 logical feature lanes. Empty B/H/N paths have no active output index. -/
theorem host_result (M : Program → Algebra α) (dot : Program → Dot α)
    (p : Parameters) (storage : TensorStorage p) (abi : InputABI p) (before : State α)
    (alloc : Allocation p abi before) (work : List Program) (legal : (programs (grid p)).Perm work) :
    ∃ result,invoke M dot p abi before alloc work=some result ∧ result.tensor=alloc.output ∧
      (∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d (hd : d<128),
        result.state.mem p.Out (logicalAddress p b h n d)=.mk .fp16
          (answer (M (n/128,b*p.heads+h,0)) (dot (n/128,b*p.heads+h,0)) p
            (start (grid p) (n/128,b*p.heads+h,0) alloc.state) (lane n d hd))) ∧
      (∀ r off,r≠p.Out → r≠alloc.unusedM.region → result.state.mem r off=before.mem r off) := by
  have uniq : Unique p (grid p) work alloc.state := by
    intro a ha b hb i j hi hj he
    exact grid_unique p storage alloc.state a (legal.mem_iff.mpr ha) b (legal.mem_iff.mpr hb) i j hi hj he
  obtain ⟨final,hr,hv,hf⟩:=run_correct M dot p (separated p storage) (grid p) work alloc.state uniq
  refine ⟨⟨final,alloc.output⟩,by simp [invoke,hr],rfl,?_,?_⟩
  · intro b hb h hh n hn d hd
    have ha : active p (start (grid p) (n/128,b*p.heads+h,0) alloc.state) (lane n d hd) := by
      change n/128*128+n%128<p.length ∧ d<128
      exact ⟨by omega,hd⟩
    have hc:=(grid_covers p b h n d hb hh hn hd).1
    have value:=hv _ (legal.mem_iff.mp hc) _ ha
    rw [physical_address p alloc.state b h n d hh hd] at value
    exact value
  · intro r off ho hm
    rw [hf r off (Or.inl ho)]
    exact alloc.frame r off ho hm

#axiomsClean host_result
#specNonCircular Allocation avoiding [invoke,runPrograms,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.AttnFwdCausal.Whole
