import bench.tritonbench_g.attn_fwd_causal.Proofs.NativeReference
import bench.tritonbench_g.attn_fwd_causal.Proofs.NativeOperands
namespace VeriTile.Bench.Repairs.AttnFwdCausal.NativeReference
open VeriTile Triton FP.Structural Native Source Whole
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

theorem inject_read (s : State Word) (r : RegionName) (off : Nat) (owner : Fin 4) :
    lower (((injectState s).mem r off).read .real owner)=.bits ((s.mem r off).read .real) := by
  change lower ((match s.mem r off with
    | .mk d v=>(Cell.mk d (inject d v) : Cell Expr)).read .real owner)=_
  generalize s.mem r off=cell
  cases cell with
  | mk dtype value=>cases dtype <;> rfl

def rowView (row : Fin 128) (a : Acc Expr) : Row :=
  ⟨fun owner=>lower (a.maximum row owner),fun owner=>lower (a.normalizer row owner),
   fun feature owner=>lower (a.value (row,feature,()) owner)⟩

def ctaInputs (p : Parameters) (s : State Word) : Inputs :=
  inputs p s (s.pids 1/p.heads) (s.pids 1%p.heads)

theorem query_row (s : State Word) (row : Fin 128) :
    (s.pids 0*128+row.val)%128=row.val ∧ (s.pids 0*128+row.val)/128=s.pids 0 := by
  have:=row.isLt
  omega

theorem score_lower (p : Parameters) (s : State Word) (r : Fin 128)
    (active : s.pids 0*128+r.val<p.length) (start : Nat) (j : Fin 64) (owner : Fin 4) :
    lower (Source.scores Native.algebra Native.dotModel p (injectState s) start (r,j,()) owner)=
      score p.length (s.pids 0*128+r.val) start (ctaInputs p s) j := by
  have hhead : s.pids 1/p.heads*p.heads+s.pids 1%p.heads=s.pids 1 := by
    simpa [Nat.mul_comm] using Nat.div_add_mod (s.pids 1) p.heads
  have pids : (injectState s).pids=s.pids := rfl
  by_cases hj : j.val<p.length-start ∧ start+j.val≤s.pids 0*128+r.val
  · simp only [Source.scores,show (injectState s).pids=s.pids from rfl,if_pos hj]
    simp [score,hj,hj.1,Native.algebra,Native.dotModel,Native.dotValues,ofFloat,toFloat,
      Source.query,Source.queryScale,Source.key,Source.keyScale,Source.zero,
      lower,literal,inject_read,ctaInputs,inputs,base,pids,
      active,(query_row s r).1,(query_row s r).2,hhead]
  · simp [Source.scores,score,hj,Native.algebra,lower,pids]

theorem reduction_lower (op : Graph → Graph → Graph) (xs : Values Expr .real [128,64])
    (r : Fin 128) (owner : Fin 4) :
    lower (Native.reduction op (some .fp32) ⟨1,by decide⟩ Bool.false xs (r,()) owner)=
      (plan owner).value op (fun j=>lower (xs (r,j,()) (physicalOwner j.val))) := by
  change (plan owner).value op (fun j=>if h : j.val<64 then
    lower (xs (r,⟨j.val,h⟩,()) (physicalOwner j.val)) else zero)=_
  congr 1
  funext j
  simp only [dif_pos j.isLt]

theorem maximum_lower (p : Parameters) (s : State Word) (r : Fin 128)
    (active : s.pids 0*128+r.val<p.length) (start : Nat) (a : Acc Expr) (owner : Fin 4) :
    lower (Source.maximum Native.algebra Native.dotModel p (injectState s) start a r owner)=
      maximum p.length (s.pids 0*128+r.val) start (ctaInputs p s) (rowView r a) owner := by
  change Graph.maxNum32 (lower (a.maximum r owner))
    (lower (Native.reduction Graph.maxNum32 (some .fp32) ⟨1,by decide⟩ Bool.false
      (Source.scores Native.algebra Native.dotModel p (injectState s) start) (r,()) owner))=_
  rw [reduction_lower]
  simp only [score_lower p s r active,maximum,rowView]

theorem probability_lower (p : Parameters) (s : State Word) (r : Fin 128)
    (active : s.pids 0*128+r.val<p.length) (start : Nat) (a : Acc Expr) (j : Fin 64) (owner : Fin 4) :
    lower (Source.probability Native.algebra Native.dotModel p (injectState s) start a (r,j,()) owner)=
      probability p.length (s.pids 0*128+r.val) start (ctaInputs p s) (rowView r a) owner j := by
  change Graph.exp2ApproxFTZ (Graph.sub32
    (lower (Source.scores Native.algebra Native.dotModel p (injectState s) start (r,j,()) owner))
    (lower (Source.maximum Native.algebra Native.dotModel p (injectState s) start a r owner)))=_
  rw [score_lower p s r active,maximum_lower p s r active]
  rfl

theorem rescale_lower (p : Parameters) (s : State Word) (r : Fin 128)
    (active : s.pids 0*128+r.val<p.length) (start : Nat) (a : Acc Expr) (owner : Fin 4) :
    lower (Source.rescale Native.algebra Native.dotModel p (injectState s) start a r owner)=
      alpha p.length (s.pids 0*128+r.val) start (ctaInputs p s) (rowView r a) owner := by
  change Graph.exp2ApproxFTZ (Graph.sub32 (lower (a.maximum r owner))
    (lower (Source.maximum Native.algebra Native.dotModel p (injectState s) start a r owner)))=_
  rw [maximum_lower p s r active]
  rfl

theorem value_lower (p : Parameters) (s : State Word) (start : Nat)
    (j : Fin 64) (feature : Fin 128) (owner : Fin 4) :
    lower (Source.value Native.algebra p (injectState s) start (j,feature,()) owner)=
      .narrow16 (if j.val<p.length-start ∧ feature.val<128 then
        .bits ((ctaInputs p s).value (start+j.val) feature.val) else Native.zero) := by
  have pids : (injectState s).pids=s.pids := rfl
  by_cases active : j.val<p.length-start
  all_goals simp [Source.value,Native.algebra,lower,Source.zero,literal,inject_read,
    ctaInputs,inputs,base,pids,active]

theorem contribution_lower (p : Parameters) (s : State Word) (r : Fin 128)
    (active : s.pids 0*128+r.val<p.length) (start : Nat) (a : Acc Expr)
    (feature : Fin 128) (owner : Fin 4) :
    lower (Source.contribution Native.algebra Native.dotModel p (injectState s) start a (r,feature,()) owner)=
      contribution p.length (s.pids 0*128+r.val) start (ctaInputs p s) (rowView r a) feature := by
  change Graph.widen16 (dotHalf 64 r.val feature.val
    (fun j=>.narrow16 (lower (Source.probability Native.algebra Native.dotModel p (injectState s)
      start a (r,j,()) (physicalOwner j.val))))
    (fun j=>lower (Source.value Native.algebra p (injectState s) start (j,feature,())
      (physicalOwner feature.val))))=_
  simp only [probability_lower p s r active,contribution,(query_row s r).1]
  congr 2
  funext j
  exact value_lower p s start j feature (physicalOwner feature.val)

theorem step_binding (p : Parameters) (s : State Word) (r : Fin 128)
    (active : s.pids 0*128+r.val<p.length) (start : Nat) (a : Acc Expr) :
    rowView r (Source.update Native.algebra Native.dotModel p (injectState s) start a)=
      step p.length (s.pids 0*128+r.val) (ctaInputs p s) start (rowView r a) := by
  simp only [rowView,Source.update,step,Row.mk.injEq]
  refine ⟨?_,?_,?_⟩
  · funext owner
    exact maximum_lower p s r active start a owner
  · funext owner
    change Graph.fma32 (lower (a.normalizer r owner))
      (lower (Source.rescale Native.algebra Native.dotModel p (injectState s) start a r owner))
      (lower (Native.reduction Graph.add32 (some .fp32) ⟨1,by decide⟩ Bool.false
        (Source.probability Native.algebra Native.dotModel p (injectState s) start a) (r,()) owner))=_
    rw [rescale_lower p s r active,reduction_lower]
    simp only [probability_lower p s r active]
    rfl
  · funext feature owner
    change Graph.fma32 (lower (a.value (r,feature,()) owner))
      (lower (Source.rescale Native.algebra Native.dotModel p (injectState s) start a r owner))
      (lower (Source.contribution Native.algebra Native.dotModel p (injectState s) start a
        (r,feature,()) owner))=_
    rw [rescale_lower p s r active,contribution_lower p s r active]
    rfl

theorem initial_binding (r : Fin 128) : rowView r (initialAcc Native.algebra)=initial := by
  simp [rowView,initialAcc,Source.zero,Native.algebra,literal,lower,initial]

theorem loop_binding (p : Parameters) (s : State Word) (r : Fin 128)
    (active : s.pids 0*128+r.val<p.length) (cur finish : Nat) (a : Acc Expr) :
    rowView r (foldRange (Source.update Native.algebra Native.dotModel p (injectState s)) cur finish 64 a)=
      foldRange (step p.length (s.pids 0*128+r.val) (ctaInputs p s)) cur finish 64 (rowView r a) := by
  by_cases h : cur<finish
  · conv_lhs => rw [foldRange]
    conv_rhs => rw [foldRange]
    simp only [show ¬(64:Nat)=0 by omega,h,dite_false,dite_true]
    rw [loop_binding p s r active (cur+64) finish,step_binding p s r active]
  · simp [foldRange,h]
termination_by finish-cur
decreasing_by omega

theorem answer_binding (p : Parameters) (s : State Word) (r : Fin 128)
    (active : s.pids 0*128+r.val<p.length) (feature : Fin 128) :
    lower (Whole.answer Native.algebra Native.dotModel p (injectState s)
      (r,feature,()) (physicalOwner feature.val))=
      attention p.length (s.pids 0*128+r.val) (ctaInputs p s) feature := by
  have loop : rowView r (Source.accumulated Native.algebra Native.dotModel p (injectState s))=
      foldRange (step p.length (s.pids 0*128+r.val) (ctaInputs p s))
        (s.pids 0*128) (min ((s.pids 0+1)*128) p.length) 64
        (foldRange (step p.length (s.pids 0*128+r.val) (ctaInputs p s))
          0 (s.pids 0*128) 64 initial) := by
    rw [Source.accumulated,loop_binding p s r active,Source.prefixAcc,
      loop_binding p s r active,initial_binding]
    rfl
  have h:=congrArg (fun result : Row=>Graph.narrow16 (Graph.divFull32
    (result.value feature (physicalOwner feature.val))
    (result.normalizer (physicalOwner feature.val)))) loop
  simpa only [rowView,Whole.answer,Source.outputValue,Native.algebra,lower,attention,(query_row s r).2] using h

theorem inject_start (g : Whole.Grid) (pid : Program) (s : State Word) :
    injectState (Whole.start g pid s)=Whole.start g pid (injectState s) := by rfl

theorem target_binding (instructions : Native.Instructions) (p : Parameters) (s : State Word)
    (b h n d : Nat) (hh : h<p.heads) (hn : n<p.length) (hd : d<128) :
    Native.outputWord instructions p s b h n d hd=target instructions p s b h n d hd := by
  have positive : 0<p.heads := by omega
  have hdiv : (b*p.heads+h)/p.heads=b := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ positive,Nat.div_eq_of_lt hh]
    omega
  have hmod : (b*p.heads+h)%p.heads=h := by simp [Nat.add_mod,Nat.mod_eq_of_lt hh]
  have hrow : n/128*128+n%128=n := by omega
  have active : (Whole.start (grid p) (n/128,b*p.heads+h,0) s).pids 0*128+(lane n d hd).1.val<p.length := by
    change n/128*128+n%128<p.length
    omega
  unfold Native.outputWord Native.outputGraph Native.reference
  rw [←inject_start,answer_binding p _ _ active]
  simp [target,ctaInputs,inputs,Whole.start,ids,lane,hdiv,hmod,hrow]

theorem source_load_word (instructions : Native.Instructions) (s : State Word)
    (r : RegionName) (off : Nat) (owner : Fin 4) :
    (lower ((Native.algebra.fp32Load (((injectState s).mem r off).read .real)) owner)).eval instructions.ops=
      (s.mem r off).read .real := by
  simp only [Native.algebra,lower,inject_read,Graph.eval,instructions.bitsIdentity]

/-- The actual evaluated FP32 source operand is reconstructed from its four
physical input bytes. The existing typed readback proof supplies this fact. -/
def SourceReadback (instructions : Native.Instructions) (p : Parameters)
    (storage : TensorStorage p) (placement : Placement p storage)
    (s : State Word) (flat : Fin placement.arena → BitVec 8) : Prop :=
  ∀ r : Buffer,∀ hr : r≠.output,∀ off,∀ ho : off<extent storage r,∀ owner : Fin 4,
    (lower ((Native.algebra.fp32Load
      (((injectState s).mem (region p r) off).read .real)) owner)).eval instructions.ops=
      Native.read32 p storage placement flat r hr off ho

theorem source_readback (instructions : Native.Instructions) (p : Parameters)
    (storage : TensorStorage p) (placement : Placement p storage)
    (s : State Word) (typed : Native.RawInputs p s) (flat : Fin placement.arena → BitVec 8)
    (binding : Native.WordBinding p storage placement flat s) :
    SourceReadback instructions p storage placement s flat := by
  intro r hr off ho owner
  rw [source_load_word,Native.input_from_current_bytes p storage placement flat s binding typed r hr off ho]
  rfl

#axiomsClean inject_read
#axiomsClean score_lower
#axiomsClean reduction_lower
#axiomsClean maximum_lower
#axiomsClean probability_lower
#axiomsClean rescale_lower
#axiomsClean value_lower
#axiomsClean contribution_lower
#axiomsClean step_binding
#axiomsClean loop_binding
#axiomsClean answer_binding
#axiomsClean target_binding
#axiomsClean source_load_word
#axiomsClean source_readback
end
end VeriTile.Bench.Repairs.AttnFwdCausal.NativeReference
