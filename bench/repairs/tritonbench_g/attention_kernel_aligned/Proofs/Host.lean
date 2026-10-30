import bench.repairs.tritonbench_g.attention_kernel_aligned.Proofs.Coverage
namespace VeriTile.Bench.Repairs.AttentionKernelAligned.Whole
open VeriTile Triton FP.Structural Source Full FPValue
noncomputable section
attribute [local instance] Classical.propDecidable
set_option maxHeartbeats 4000000

inductive DType where | fp16 | bf16 | fp32 deriving DecidableEq
structure Descriptor where
  (object device : Nat)
  region : RegionName
  dtype : DType
  (shape strides : List Nat)
  (ordinaryTensor negative conjugate : Bool)

/-- Metadata of the five caller-owned ordinary tensors. No numerical or
post-state facts are included; the wrapper creates no tensor allocation. -/
inductive ScalarKind where | pythonFloat | pythonInt deriving DecidableEq

structure InputABI (p : Parameters) where
  launchConfig : p.numWarps∈[4,8] ∧ 1≤p.numStages ∧ p.numStages≤4
  resourceGuard : ¬(p.blockN=128 ∧ p.head=128 ∧ (p.numStages=4 ∨ (p.blockM=128 ∧ p.numStages=3)))
  scaleKind : ScalarKind
  /-- `scale` is the mathematical projection of the observed launch word.
  Both IEEE zeros project to zero; the raw word remains separate. -/
  scaleDecoded : (Float32Bits.mk p.scaleWord).decodeReal=some p.scale
  scaleRange : |p.scale|≤340282346638528860000000000000000000000
  (query key value bias output : Descriptor)
  shapes : query.shape=[p.batches,p.heads,p.length,p.head] ∧ key.shape=query.shape ∧ value.shape=query.shape ∧
    output.shape=query.shape ∧ bias.shape=[p.batches,p.heads,p.length,128]
  strides : query.strides=[p.qBatchStride,p.qHeadStride,p.qRowStride,p.qColStride] ∧
    key.strides=[p.kBatchStride,p.kHeadStride,p.kSeqStride,p.kDimStride] ∧
    value.strides=[p.vBatchStride,p.vHeadStride,p.vSeqStride,p.vDimStride] ∧
    output.strides=[p.outBatchStride,p.outHeadStride,p.outRowStride,p.outColStride] ∧
    bias.strides=[p.biasBatchStride,p.biasHeadStride,p.biasRowStride,1]
  biasContiguous : Nonempty p → (1<p.batches → p.biasBatchStride=p.heads*p.length*128) ∧
    (1<p.heads → p.biasHeadStride=p.length*128) ∧ (1<p.length → p.biasRowStride=128)
  regions : query.region=p.q ∧ key.region=p.k ∧ value.region=p.v ∧ bias.region=p.bias ∧ output.region=p.output
  types : ∀ t∈[query,key,value,bias,output],t.dtype=.fp16
  device : ∀ t∈[query,key,value,bias,output],t.device=query.device
  ordinary : ∀ t∈[query,key,value,bias,output],t.ordinaryTensor=Bool.true ∧ t.negative=Bool.false ∧ t.conjugate=Bool.false
  disjoint : Nonempty p → [p.q,p.k,p.v,p.bias,p.output].Nodup

structure Returned (α : Type) where
  state : State α
  output : Descriptor
  pythonNone : Unit

variable {α : Type} [Inhabited α]
def invoke (M : Program → Algebra α) (P : Program → Primitives α) (p : Parameters)
    (abi : InputABI p) (before : State α) (work : List Program) : Option (Returned α) :=
  if Nonempty p then (runPrograms M P p (grid p) work before).map fun t=>⟨t,abi.output,()⟩
  else some ⟨before,abi.output,()⟩

def logicalAddress (p : Parameters) (b h n d : Nat) := offset (oStrides p) b h n d

def OutsideLogical (p : Parameters) (r : RegionName) (off : Nat) : Prop :=
  r≠p.output ∨ ∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d<p.head,off≠logicalAddress p b h n d

theorem physical_address (p : Parameters) (selected : Selected p) (s : State α) (b h n d : Nat)
    (hh : h<p.heads) (hd : d<p.head) :
    address p (start (grid p) (n/p.blockM,b*p.heads+h,0) s) (lane p selected n d hd)=logicalAddress p b h n d := by
  have hH : 0<p.heads := by omega
  have hdiv : (b*p.heads+h)/p.heads=b := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ hH,Nat.div_eq_of_lt hh]
    omega
  have hmod : (b*p.heads+h)%p.heads=h := by simp [Nat.add_mod,Nat.mod_eq_of_lt hh]
  have hn : n/p.blockM*p.blockM+n%p.blockM=n := by have:=Nat.div_add_mod' n p.blockM; omega
  simp [address,outputOffset,outputBase,row,start,ids,lane,logicalAddress,offset,oStrides,hdiv,hmod,hn]

/-- Full caller output readback and exact complement frame for any legal CTA
permutation. Empty B/H/N returns the identical state and Python None. -/
theorem host_result (M : Program → Algebra α) (P : Program → Primitives α)
    (p : Parameters) (g : Guard p) (storage : TensorStorage p) (abi : InputABI p) (before : State α)
    (work : List Program) (legal : (programs (grid p)).Perm work) :
    ∃ result,invoke M P p abi before work=some result ∧ result.output=abi.output ∧ result.pythonNone=() ∧
      (∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d (hd : d<p.head),
        result.state.mem p.output (logicalAddress p b h n d)=.mk .fp16
          (answer (P (n/p.blockM,b*p.heads+h,0)) p
            (start (grid p) (n/p.blockM,b*p.heads+h,0) before) (lane p g.selected n d hd))) ∧
      (∀ r off,OutsideLogical p r off → result.state.mem r off=before.mem r off) := by
  by_cases ne : Nonempty p
  · have uniq : Unique p (grid p) work before := by
      intro a ha b hb i j hi hj he
      exact grid_unique p g before a (legal.mem_iff.mpr ha) b (legal.mem_iff.mpr hb) i j hi hj he
    obtain ⟨final,hr,hv,hf⟩:=run_correct M P p (separated p storage ne)
      (g.selected.blockN_pos p) (grid p) work before uniq
    refine ⟨⟨final,abi.output,()⟩,by simp [invoke,ne,hr],rfl,rfl,?_,?_⟩
    · intro b hb h hh n hn d hd
      have ha : active p (start (grid p) (n/p.blockM,b*p.heads+h,0) before) (lane p g.selected n d hd) := by
        change n/p.blockM*p.blockM+n%p.blockM<p.length
        have:=Nat.div_add_mod' n p.blockM
        omega
      have hc:=(grid_covers p g.selected b h n hb hh hn).1
      have value:=hv _ (legal.mem_iff.mp hc) _ ha
      rw [physical_address p g.selected before b h n d hh hd] at value
      exact value
    · intro r off outside
      apply hf
      rcases outside with outside|outside
      · exact Or.inl outside
      · right
        intro a ha i hi he
        obtain ⟨hx,hy,hz⟩:=(mem_programs (grid p) a.1 a.2.1 a.2.2).mp (legal.mem_iff.mpr ha)
        obtain ⟨_,hB,hH⟩:=head_decode p a.2.1 hy
        exact outside _ hB _ hH _ hi _ i.2.1.isLt he
  · refine ⟨⟨before,abi.output,()⟩,by simp [invoke,ne],rfl,rfl,?_,?_⟩
    · intro b hb h hh n hn d hd
      exact False.elim (ne ⟨by omega,by omega,by omega⟩)
    · intro r off _
      rfl

#axiomsClean host_result
#specNonCircular InputABI avoiding [invoke,runPrograms,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.AttentionKernelAligned.Whole
