import bench.tritonbench_g.token_attn_mistral.IO
import bench.tritonbench_g.token_attn_mistral.Proofs.SourceFormula
import bench.tritonbench_g.token_attn_mistral.Proofs.AccessSafety

namespace VeriTile.Bench.Repairs.TokenAttnMistralGridYChecked.Binding
open VeriTile Triton FP.Structural TokenAttnMistral
noncomputable section
set_option maxRecDepth 10000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem count_le (p : Parameters) (work : List Grid.PID) (counts : Fin work.length → Nat)
    (j : Fin work.length) : counts j≤∑ k,counts k :=
  Finset.single_le_sum (fun _ _=>Nat.zero_le _) (Finset.mem_univ j)

theorem lane_bound (p : Parameters) (work : List Grid.PID) (counts : Fin work.length → Nat)
    (j : Fin work.length) (n : Nat) (live : n<counts j) (lane : Fin p.block) :
    n+lane.val<Ports.span p work counts := by
  have := count_le p work counts j
  have := lane.isLt
  unfold Ports.span
  omega

theorem visited_lane (p : Parameters) (n : Nat) (aligned : n%p.block=0)
    (lane : Fin p.block) : (n+lane.val)/p.block*p.block=n := by
  have h := Nat.mod_add_div (n+lane.val) p.block
  have hm : (n+lane.val)%p.block=lane.val := by
    simp [Nat.add_mod,aligned,Nat.mod_eq_of_lt lane.isLt]
  rw [hm,Nat.mul_comm] at h
  omega

structure Facts (p : Parameters) (work : List Grid.PID) (counts : Fin work.length → Nat)
    (x : Ports.Logical α work.length (Ports.span p work counts) p.width) (m : Grid.Memory α) : Prop where
  sequenceWord : ∀ j,(m p.length.cast (work[j].1*p.sequenceStride)).read .int=x.sequenceWord j
  startWord : ∀ j,(m p.start.cast (work[j].1*p.attentionStartStride)).read .int=x.startWord j
  requestWord : ∀ j,(m p.request.cast (work[j].1*p.requestStride)).read .int=x.requestWord j
  attentionWord : ∀ j,(m p.attentionLength.cast (work[j].1*p.attentionLengthStride)).read .int=(counts j : Int)
  probability : ∀ j n,0<p.block → n.val<counts j →
    (m p.prob (work[j].2*p.probHeadStride+(x.start j+n.val)*p.probTokenStride)).read .fp32=x.probability j n
  tokenWord : ∀ j n,Ports.visited p (counts j) n.val → n.val+Ports.window p x j<x.sequence j →
    (m p.tokens.cast (x.request j*p.requestBatchStride+(Ports.window p x j+n.val)*p.requestTokenStride)).read .int=x.tokenWord j n
  value : ∀ j n,Ports.visited p (counts j) n.val → n.val+Ports.window p x j<x.sequence j → ∀ d,
    (m p.value (work[j].2/p.kvGroup*p.valueHeadStride+d.val*p.valueDimStride+
      x.token j n*p.valueBatchStride)).read .fp32=x.value j n d
  domain : Ports.Admissible p counts x

section Projections
variable {p : Parameters} {work : List Grid.PID} {counts : Fin work.length → Nat}
  {x : Ports.Logical α work.length (Ports.span p work counts) p.width} {m : Grid.Memory α}

theorem Facts.sequence (f : Facts p work counts x m) (j : Fin work.length) :
    ((m p.length.cast (work[j].1*p.sequenceStride)).read .int).toNat = x.sequence j :=
  congrArg Int.toNat (f.sequenceWord j)

theorem Facts.start (f : Facts p work counts x m) (j : Fin work.length) :
    ((m p.start.cast (work[j].1*p.attentionStartStride)).read .int).toNat = x.start j :=
  congrArg Int.toNat (f.startWord j)

theorem Facts.request (f : Facts p work counts x m) (j : Fin work.length) :
    ((m p.request.cast (work[j].1*p.requestStride)).read .int).toNat = x.request j :=
  congrArg Int.toNat (f.requestWord j)

theorem Facts.attention (f : Facts p work counts x m) (j : Fin work.length) :
    ((m p.attentionLength.cast (work[j].1*p.attentionLengthStride)).read .int).toNat=counts j := by
  simpa only [Int.toNat_natCast] using congrArg Int.toNat (f.attentionWord j)

theorem Facts.tokens (f : Facts p work counts x m) (j : Fin work.length)
    (n : Fin (Ports.span p work counts)) (visited : Ports.visited p (counts j) n.val)
    (live : n.val+Ports.window p x j<x.sequence j) :
    ((m p.tokens.cast (x.request j*p.requestBatchStride+(Ports.window p x j+n.val)*p.requestTokenStride)).read .int).toNat=x.token j n :=
  congrArg Int.toNat (f.tokenWord j n visited live)
end Projections

 theorem facts (M : Algebra α) (R : SumPrimitive α) (p : Parameters)
    (work : List Grid.PID) (counts : Fin work.length → Nat) (vals : Ports.Values α p work counts)
    (domain : Ports.Domain p work counts vals) (s : State α)
    (bound : (IO.program M R p work counts).InputBinding vals s) :
    Facts p work counts (Ports.logical p work counts vals) s.mem := by
  constructor
  · intro j
    simpa [IO.program,Ports.memory,Ports.inputs,Ports.logical] using
      bound ⟨0,by simp [IO.program,Ports.memory,Ports.inputs]⟩ (j,()) (by change True; trivial)
  · intro j
    simpa [IO.program,Ports.memory,Ports.inputs,Ports.logical] using
      bound ⟨1,by simp [IO.program,Ports.memory,Ports.inputs]⟩ (j,()) (by change True; trivial)
  · intro j
    simpa [IO.program,Ports.memory,Ports.inputs,Ports.logical] using
      bound ⟨2,by simp [IO.program,Ports.memory,Ports.inputs]⟩ (j,()) (by change True; trivial)
  · intro j
    have h := bound ⟨3,by simp [IO.program,Ports.memory,Ports.inputs]⟩ (j,()) (by change True; trivial)
    have hc := domain.attention j
    simpa [IO.program,Ports.memory,Ports.inputs,Ports.logical,mapValue,id_eq] using h.trans hc
  · intro j n pos live
    simpa [IO.program,Ports.memory,Ports.inputs,Ports.logical,mapValue,id_eq] using
      bound ⟨5,by simp [IO.program,Ports.memory,Ports.inputs]⟩ (j,n,()) (by simpa [IO.program,Ports.memory,Ports.inputs,Ports.outputs,Fin.cases] using And.intro pos live)
  · intro j n visited live
    simpa [IO.program,Ports.memory,Ports.inputs,Ports.logical,mapValue,id_eq] using
      bound ⟨4,by simp [IO.program,Ports.memory,Ports.inputs]⟩ (j,n,()) (by simpa [IO.program,Ports.memory,Ports.inputs,Ports.outputs,Fin.cases] using And.intro visited live)
  · intro j n visited live d
    simpa [IO.program,Ports.memory,Ports.inputs,Ports.logical,mapValue,id_eq] using
      bound ⟨6,by simp [IO.program,Ports.memory,Ports.inputs]⟩ (j,n,d,()) (by simpa [IO.program,Ports.memory,Ports.inputs,Ports.outputs,Fin.cases] using And.intro visited live)
  · exact domain


 theorem Facts.congr {p : Parameters} {work : List Grid.PID} {counts : Fin work.length → Nat}
    {x : Ports.Logical α work.length (Ports.span p work counts) p.width} {a b : Grid.Memory α}
    (f : Facts p work counts x a) (agree : Grid.Agree p b a) : Facts p work counts x b := by
  have hp := agree p.prob (by simp [Grid.Reads])
  have hv := agree p.value (by simp [Grid.Reads])
  have ht := agree p.tokens.cast (by simp [Grid.Reads])
  have hr := agree p.request.cast (by simp [Grid.Reads])
  have hs := agree p.start.cast (by simp [Grid.Reads])
  have hl := agree p.length.cast (by simp [Grid.Reads])
  have ha := agree p.attentionLength.cast (by simp [Grid.Reads])
  constructor <;> try simp only [hp,hv,ht,hr,hs,hl,ha]
  · exact f.sequenceWord
  · exact f.startWord
  · exact f.requestWord
  · exact f.attentionWord
  · exact f.probability
  · exact f.tokenWord
  · exact f.value
  · exact f.domain

end
end VeriTile.Bench.Repairs.TokenAttnMistralGridYChecked.Binding
