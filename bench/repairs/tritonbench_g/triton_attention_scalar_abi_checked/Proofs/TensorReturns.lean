import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.TensorStorage
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.TensorStorage
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.TritonAttention.Whole001
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 12000000

def inputAt {α : Type} (c : Shape) (t : Objects c) (read : Nat → α) (r : RegionName) (o : Nat) : α:=read (address c t r o)
def initialReal (c : Shape) (t : Objects c) (read : Nat → ℝ) : Grid.Memory Erased.R:=
  fun r o=>.mk .fp32 (some (if r="DQ" then 0 else inputAt c t read r o))
def initialWords (c : Shape) (t : Objects c) (read : Nat → Native.Word) : Grid.Memory Native.Word:=
  fun r o=>.mk .fp32 (if r="DQ" then 0 else inputAt c t read r o)

def realInput (c : Shape) (t : Objects c) (read : Nat → ℝ) (r : RegionName) (h : Fin c.groups) : Math.Tensor c.rows c.dim:=
  fun i d=>logical read (t.get r) (offset c (h,i,d))

theorem tensor_numel (c : Shape) (t : Objects c) (r : RegionName) (hr : r∈tensorRegions) :
    (t.get r).metadata.numel=c.elements := by
  simp [Torch.TensorMetadata.numel,t.tensorShape r hr,Shape.elements,Shape.groups,Nat.mul_assoc]

theorem headTensor_logical (c : Shape) (t : Objects c) (read : Nat → ℝ) (r : RegionName) (hr : r∈tensorRegions) (h : Fin c.groups) :
    Erased.headTensor c (inputAt c t read) r h=realInput c t read r h := by
  funext i d
  symm
  exact logical_direct c t read r (offset c (h,i,d)) (by rw [tensor_numel c t r hr];exact offset_lt c (h,i,d))

theorem wordInput_logical (c : Shape) (t : Objects c) (read : Nat → Native.Word) (r : RegionName) (hr : r∈tensorRegions) (i : Coord c) :
    inputAt c t read r (offset c i)=
      (Torch.FP32.logicalValue (fun a=>⟨read a⟩) (t.get r) (t.laws r) (offset c i)).bits := by
  rw [native_logical,logical_direct c t read r _ (by rw [tensor_numel c t r hr];exact offset_lt c i)]
  rfl

theorem result_readback {α : Type} [Inhabited α] (c : Shape) (t : Objects c) (before : Nat → α) (final : Grid.Memory α)
    (r : RegionName) (hr : r∈(["Out","DQ","DK","DV"] : List RegionName)) (i : Coord c) :
    logical (result c t before final).read (t.get r) (offset c i)=(final r (offset c i)).read .fp32 := by
  have ht : r∈tensorRegions := by simp only [List.mem_cons,List.not_mem_nil,or_false] at hr;rcases hr with rfl|rfl|rfl|rfl <;> simp [tensorRegions]
  rw [logical_direct c t _ r _ (by rw [tensor_numel c t r ht];exact offset_lt c i)]
  apply publish_readback
  left
  refine ⟨?_,offset_lt c i⟩
  simp only [List.mem_cons,List.not_mem_nil,or_false] at hr ⊢
  tauto

def invokeReal (c : Shape) (t : Objects c) (read : Nat → ℝ) (scale : ℝ) (fw pw bw : List Grid.Program) : Option (Result Erased.R):=
  (Grid.wholeRun Erased.algebra Erased.model Erased.model Erased.model c scale fw pw bw (initialReal c t read)).map
    (result c t (fun a=>some (read a)))
def invokeNative (c : Shape) (t : Objects c) (read : Nat → Native.Word) (scalar : Checked.ScalarABI)
    (owners : TritonAttention.Whole001.Native.Owners) (I : Native.Prior.Instructions) (literal : ℝ → Native.Word)
    (fw pw bw : List Grid.Program) : Option (Result Native.Word):=
  (Native.Selected.nativeRun (Native.constexprScale scalar) c (scalar.project scalar.raw) owners I literal fw pw bw (initialWords c t read)).map
    (result c t read)

def RealValues (c : Shape) (t : Objects c) (read : Nat → ℝ) (scale : ℝ) (r : Result Erased.R) : Prop:=
  ∀ i : Coord c,
    logical r.read r.output (offset c i)=some (Math.attention (realInput c t read "Q" i.1)
      (realInput c t read "K" i.1) (realInput c t read "V" i.1) scale i.2.1 i.2.2) ∧
    logical r.read r.dq (offset c i)=some (Math.gradientQ (realInput c t read "Q" i.1)
      (realInput c t read "K" i.1) (realInput c t read "V" i.1) (realInput c t read "DO" i.1) scale i.2.1 i.2.2) ∧
    logical r.read r.dk (offset c i)=some (Math.gradientK (realInput c t read "Q" i.1)
      (realInput c t read "K" i.1) (realInput c t read "V" i.1) (realInput c t read "DO" i.1) scale i.2.1 i.2.2) ∧
    logical r.read r.dv (offset c i)=some (Math.gradientV (realInput c t read "Q" i.1)
      (realInput c t read "K" i.1) (realInput c t read "DO" i.1) scale i.2.1 i.2.2)

def Identities {α : Type} (c : Shape) (t : Objects c) (r : Result α) : Prop:=
  r.output=t.get "Out" ∧ r.dq=t.get "DQ" ∧ r.dk=t.get "DK" ∧ r.dv=t.get "DV" ∧ r.scaleGradient=none

theorem real_tensors (c : Shape) (t : Objects c) (read : Nat → ℝ) (scale : ℝ) (fw pw bw : List Grid.Program)
    (hf : Grid.FullSchedule (Grid.forwardGrid c) fw) (hp : Grid.FullSchedule (Grid.preprocessGrid c) pw)
    (hb : Grid.FullSchedule (Grid.backwardGrid c) bw) :
    ∃ r,invokeReal c t read scale fw pw bw=some r ∧ Identities c t r ∧ RealValues c t read scale r ∧
      (∀ a,¬(∃ region o,Grid.Written c region o ∧ address c t region o=a) → r.read a=some (read a)) := by
  obtain ⟨final,he,hv,hframe⟩:=Erased.whole_execution_tensors c scale fw pw bw hf hp hb (initialReal c t read) (inputAt c t read)
    (by intro r hr o;simp only [List.mem_cons,List.not_mem_nil,or_false] at hr;rcases hr with rfl|rfl|rfl|rfl <;> simp [initialReal,Cell.read])
    (by intro i;simp [initialReal])
  refine ⟨result c t (fun a=>some (read a)) final,by simp only [invokeReal,he,Option.map_some],⟨rfl,rfl,rfl,rfl,rfl⟩,?_,?_⟩
  · intro i
    have out:=hv.output i
    have dq:=hv.dq i
    have dk:=hv.dk i
    have dv:=hv.dv i
    rw [headTensor_logical c t read "Q" (by simp [tensorRegions]),headTensor_logical c t read "K" (by simp [tensorRegions]),
      headTensor_logical c t read "V" (by simp [tensorRegions])] at out dq dk
    rw [headTensor_logical c t read "DO" (by simp [tensorRegions])] at dq dk
    rw [headTensor_logical c t read "Q" (by simp [tensorRegions]),headTensor_logical c t read "K" (by simp [tensorRegions]),
      headTensor_logical c t read "DO" (by simp [tensorRegions])] at dv
    change logical _ (t.get "Out") _=_ ∧ logical _ (t.get "DQ") _=_ ∧ logical _ (t.get "DK") _=_ ∧ logical _ (t.get "DV") _=_
    rw [result_readback c t _ final "Out" (by simp),result_readback c t _ final "DQ" (by simp),
      result_readback c t _ final "DK" (by simp),result_readback c t _ final "DV" (by simp),out,dq,dk,dv]
    exact ⟨rfl,rfl,rfl,rfl⟩
  · exact publish_frame c t _ final

def NativeValues (c : Shape) (t : Objects c) (read : Nat → Native.Word) (scalar : Checked.ScalarABI)
    (owners : TritonAttention.Whole001.Native.Owners) (I : Native.Prior.Instructions) (literal : ℝ → Native.Word) (r : Result Native.Word) : Prop:=
  let one:=Native.constexprScale scalar
  let scale:=scalar.project scalar.raw
  let x:=Native.Selected.wordInput c (inputAt c t read)
  let back:=Native.Target.prepared c one scale x owners
  ∀ i : Coord c,
    logical r.read r.output (offset c i)=((Native.Target.forward c one scale x owners).output i.1.val i.2.1.val i.2.2.val).eval I literal ∧
    logical r.read r.dq (offset c i)=(Native.Target.query c one scale back i.1.val i.2.1.val i.2.2.val c.blocks).eval I literal ∧
    logical r.read r.dk (offset c i)=(Native.Target.key c one scale back i.1.val i.2.1.val i.2.2.val).eval I literal ∧
    logical r.read r.dv (offset c i)=(Native.Target.value c one scale back i.1.val i.2.1.val i.2.2.val).eval I literal

theorem native_tensors (c : Shape) (t : Objects c) (read : Nat → Native.Word) (scalar : Checked.ScalarABI)
    (owners : TritonAttention.Whole001.Native.Owners) (I : Native.Prior.Instructions) (literal : ℝ → Native.Word)
    (fw pw bw : List Grid.Program) (hf : Grid.FullSchedule (Grid.forwardGrid c) fw)
    (hp : Grid.FullSchedule (Grid.preprocessGrid c) pw) (hb : Grid.FullSchedule (Grid.backwardGrid c) bw) :
    ∃ r,invokeNative c t read scalar owners I literal fw pw bw=some r ∧ Identities c t r ∧ NativeValues c t read scalar owners I literal r ∧
      (∀ a,¬(∃ region o,Grid.Written c region o ∧ address c t region o=a) → r.read a=read a) := by
  obtain ⟨final,he,hv,hframe⟩:=Native.Selected.whole_execution_words (Native.constexprScale scalar) c (scalar.project scalar.raw)
    owners I literal fw pw bw hf hp hb (initialWords c t read) (inputAt c t read)
    (by intro r hr o;simp only [List.mem_cons,List.not_mem_nil,or_false] at hr;rcases hr with rfl|rfl|rfl|rfl <;> simp [initialWords])
    (by intro i;simp [initialWords])
  refine ⟨result c t read final,by simp only [invokeNative,he,Option.map_some],⟨rfl,rfl,rfl,rfl,rfl⟩,?_,?_⟩
  · intro i
    change logical _ (t.get "Out") _=_ ∧ logical _ (t.get "DQ") _=_ ∧ logical _ (t.get "DK") _=_ ∧ logical _ (t.get "DV") _=_
    rw [result_readback c t _ final "Out" (by simp),result_readback c t _ final "DQ" (by simp),
      result_readback c t _ final "DK" (by simp),result_readback c t _ final "DV" (by simp),
      (hv i).1,(hv i).2.1,(hv i).2.2.1,(hv i).2.2.2]
    exact ⟨rfl,rfl,rfl,rfl⟩
  · exact publish_frame c t _ final

#axiomsClean real_tensors
#axiomsClean native_tensors
#specNonCircular RealValues avoiding [invokeReal,Grid.wholeRun,FP.Structural.exec]
#specNonCircular NativeValues avoiding [invokeNative,Native.Selected.nativeRun,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.TensorStorage
