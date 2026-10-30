import bench.repairs.tritonbench_g.attention_kernel.Proofs.Coverage
namespace VeriTile.Bench.Repairs.AttentionKernel.Whole001.Whole
open VeriTile Triton FP.Structural Source
noncomputable section
set_option maxHeartbeats 4000000

structure Descriptor where
  (object device : Nat)
  region : RegionName
  dtype : TypedSource.Storage
  (shape strides : List Nat)
  (negative conjugate contiguous cuda : Bool)

/-- The output descriptor is supplied by the caller. Ordinary Tensor/device,
shape, storage and launch-plan validity contain no output-value premise. -/
structure InputABI (p : Parameters) where
  (query key value bias output : Descriptor)
  (numWarps numStages : Nat)
  blocks : p.bm∈[16,32,64,128] ∧ p.bn∈[16,32,64,128] ∧ p.dim∈[16,32,64,128]
  launchConfig : numWarps∈[4,8] ∧ 1≤numStages ∧ numStages≤4
  scaleBound : |p.scale|≤3.4028234663852886e38
  keyBlocks : (p.length+p.bn-1)/p.bn≤128
  shapes : query.shape=[p.batches,p.heads,p.length,p.dim] ∧ key.shape=query.shape ∧
    value.shape=query.shape ∧ output.shape=query.shape ∧ bias.shape=[p.batches,p.heads,p.length,128]
  strides : query.strides=[p.q.batch,p.q.head,p.q.row,p.q.dim] ∧
    key.strides=[p.k.batch,p.k.head,p.k.row,p.k.dim] ∧
    value.strides=[p.v.batch,p.v.head,p.v.row,p.v.dim] ∧
    output.strides=[p.o.batch,p.o.head,p.o.row,p.o.dim] ∧
    bias.strides=[p.bias.batch,p.bias.head,p.bias.row,p.bias.dim]
  biasContiguous : bias.contiguous=Bool.true
  regions : query.region=p.Q ∧ key.region=p.K ∧ value.region=p.V ∧ bias.region=p.Bias ∧ output.region=p.Out
  types : ∀ t∈[query,key,value,bias,output],t.dtype=p.storage
  device : ∀ t∈[query,key,value,bias,output],t.cuda=Bool.true ∧ t.device=query.device
  ordinary : ∀ t∈[query,key,value,bias,output],t.negative=Bool.false ∧ t.conjugate=Bool.false

variable {α : Type} [Inhabited α]

structure Returned (α : Type) where
  state : State α
  result : Unit

def Live (p : Parameters) : Prop := 0<p.batches ∧ 0<p.heads ∧ 0<p.length
instance (p : Parameters) : Decidable (Live p) := inferInstanceAs (Decidable (_ ∧ _ ∧ _))

def invoke (M : Program→ Algebra α) (dot : Program→ Dot α) (p : Parameters)
    (_abi : InputABI p) (before : State α) (work : List Program) : Option (Returned α) :=
  if Live p then (runPrograms M dot p (grid p) work before).map fun t=>⟨t,()⟩
  else some ⟨before,()⟩

/-- The Python empty branch returns before layout, launch and address guards. -/
theorem empty_result (M : Program→ Algebra α) (dot : Program→ Dot α) (p : Parameters)
    (abi : InputABI p) (before : State α) (work : List Program) (empty : ¬Live p) :
    invoke M dot p abi before work=some ⟨before,()⟩ := by simp [invoke,empty]

def logicalAddress (p : Parameters) (b h n d : Nat) : Nat :=
  b*p.o.batch+h*p.o.head+n*p.o.row+d*p.o.dim

theorem physical_address (p : Parameters) (hbm : 0<p.bm) (s : State α) (b h n d : Nat)
    (hh : h<p.heads) (hd : d<p.dim) :
    address p (start (grid p) (n/p.bm,b*p.heads+h,0) s) (lane p hbm n d hd)=logicalAddress p b h n d := by
  have hH : 0<p.heads := by omega
  have hdiv : (b*p.heads+h)/p.heads=b := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ hH,Nat.div_eq_of_lt hh]
    omega
  have hmod : (b*p.heads+h)%p.heads=h := by simp [Nat.add_mod,Nat.mod_eq_of_lt hh]
  have hn : n/p.bm*p.bm+n%p.bm=n := (by simpa only [Nat.mul_comm] using Nat.div_add_mod n p.bm)
  simp [address,base,start,ids,lane,logicalAddress,hdiv,hmod,hn]

/-- Complete caller output, arbitrary legal CTA order, unchanged input memory,
and the actual Python None return. Exterior output-view frame follows Post. -/
theorem host_result (M : Program→ Algebra α) (dot : Program→ Dot α)
    (p : Parameters) (live : Live p) (guard : Guard p) (storage : TensorStorage p) (abi : InputABI p)
    (before : State α) (work : List Program) (legal : (programs (grid p)).Perm work) :
    ∃ result,invoke M dot p abi before work=some result ∧ result.result=() ∧
      (∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d (hd : d<p.dim),
        result.state.mem p.Out (logicalAddress p b h n d)=.mk p.storage.tile
          (ofFloat p.storage.float (answer (M (n/p.bm,b*p.heads+h,0)) (dot (n/p.bm,b*p.heads+h,0)) p
            (start (grid p) (n/p.bm,b*p.heads+h,0) before) (lane p guard.bm n d hd)))) ∧
      (∀ r off,Outside p (grid p) work before r off→ result.state.mem r off=before.mem r off) := by
  have uniq : Unique p (grid p) work before := by
    intro a ha b hb i j hi hj he
    exact grid_unique p guard before a (legal.mem_iff.mpr ha) b (legal.mem_iff.mpr hb) i j hi hj he
  obtain ⟨final,hr,hv,hf⟩:=run_correct M dot p (separated p storage live) guard.bn (grid p) work before uniq
  refine ⟨⟨final,()⟩,by simp [invoke,live,hr],rfl,?_,hf⟩
  intro b hb h hh n hn d hd
  have ha : active p (start (grid p) (n/p.bm,b*p.heads+h,0) before) (lane p guard.bm n d hd) := by
    change n/p.bm*p.bm+n%p.bm<p.length
    simpa only [Nat.mul_comm, Nat.div_add_mod] using hn
  have hc:=(grid_covers p guard.bm b h n d hb hh hn hd).1
  have value:=hv _ (legal.mem_iff.mp hc) _ ha
  rw [physical_address p guard.bm before b h n d hh hd] at value
  exact value

#axiomsClean host_result
#specNonCircular InputABI avoiding [invoke,runPrograms,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.AttentionKernel.Whole001.Whole
