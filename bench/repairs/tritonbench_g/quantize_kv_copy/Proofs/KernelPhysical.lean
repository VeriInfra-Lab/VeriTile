import bench.repairs.tritonbench_g.quantize_kv_copy.Proofs.KernelTensor

namespace VeriTile.Bench.Repairs.QuantizeKvCopy.KernelPhysical
open VeriTile.Triton FP.Structural Kernel KernelModels KernelWhole KernelTensor
noncomputable section
open Classical
local instance : Inhabited Word := ⟨KernelModels.zero⟩
set_option maxHeartbeats 2500000

def byteOfBits (x : BitVec 8) : Byte := ⟨x.toNat,x.isLt⟩
def byteOfInt (x : Int) : Byte := byteOfBits (BitVec.ofInt 8 x)

 theorem byte_roundtrip (x : BitVec 8) : byteOfInt x.toInt=byteOfBits x := by simp [byteOfInt]

 theorem decode_scale (x : Word) : decodeWord (scaleByte x)=x := by
  unfold decodeWord
  rw [scale_bytes_exact]
  cases x
  simp [FP.Value.ofBits]

 theorem encode_decode (bytes : Fin 4 → Byte) (b : Fin 4) : scaleByte (decodeWord bytes) b=bytes b := by
  have h0 := (bytes ⟨0,by decide⟩).isLt
  have h1 := (bytes ⟨1,by decide⟩).isLt
  have h2 := (bytes ⟨2,by decide⟩).isLt
  have h3 := (bytes ⟨3,by decide⟩).isLt
  have hn : (bytes ⟨0,by decide⟩).val+256*(bytes ⟨1,by decide⟩).val+
      65536*(bytes ⟨2,by decide⟩).val+16777216*(bytes ⟨3,by decide⟩).val<2^32 := by omega
  apply Fin.ext
  fin_cases b <;>
    simp [scaleByte,decodeWord,FP.Value.ofBits,BitVec.toNat_ofNat,FP.Format.width,FP.Format.fractionBits,Nat.mod_eq_of_lt hn] <;> omega

 def CodeTouched (m : Inputs) (w : Nat → MetadataWord) (a : Nat) : Prop :=
  ∃ pid g c,pid<(request m w).programs ∧ g<(request m w).groups ∧ c<8 ∧
    a=base m .output+codeOffset (parameters m w) ((request m w).destination pid) g c
 def ScaleTouched (m : Inputs) (w : Nat → MetadataWord) (a : Nat) : Prop :=
  ∃ pid g,∃ b : Fin 4,pid<(request m w).programs ∧ g<(request m w).groups ∧
    a=base m .scale+4*scaleOffset (parameters m w) ((request m w).destination pid) g+b.val
 def Touched (m : Inputs) (w : Nat → MetadataWord) (a : Nat) : Prop := CodeTouched m w a ∨ ScaleTouched m w a

/-- Lower the canonical source's final typed stores into the one caller byte
heap. Both outputs use the same final state; bytes outside the exact source
store footprint retain their initial value. This is the store codec boundary. -/
 def post (m : Inputs) (w : Nat → MetadataWord) (before : Heap) (final : State Word) : Heap := fun a =>
  if CodeTouched m w a then byteOfInt ((final.mem "O" (a-base m .output)).read .int)
  else if ScaleTouched m w a then
    scaleByte ((final.mem "S" ((a-base m .scale)/4)).read .real) ⟨(a-base m .scale)%4,Nat.mod_lt _ (by decide)⟩
  else before a

 theorem output_bound (m : Inputs) (w : Nat → MetadataWord) (h : InputContract m w)
    (pid g c : Nat) (hp : pid<(request m w).programs) (hg : g<(request m w).groups) (hc : c<8) :
    codeOffset (parameters m w) ((request m w).destination pid) g c<(m .output).numel := by
  have hb := RW.linear_bound ((request m w).outputPrograms*(request m w).groups) 8
    ((request m w).destination pid*(request m w).groups+g) c
    (RW.linear_bound (request m w).outputPrograms (request m w).groups _ g
      ((checked_domain m w h).program_destination_bound pid hp) hg) hc
  have sz := sizes m w h .output
  simp only [elements,elementBytes,groupWidth,Nat.one_mul] at sz
  rw [←sz]
  simpa [codeOffset,parameters,Nat.add_mul,Nat.mul_assoc] using hb

 theorem scale_bound (m : Inputs) (w : Nat → MetadataWord) (h : InputContract m w)
    (pid g : Nat) (hp : pid<(request m w).programs) (hg : g<(request m w).groups) :
    scaleOffset (parameters m w) ((request m w).destination pid) g<(m .scale).numel := by
  have hb := RW.linear_bound (request m w).outputPrograms (request m w).groups _ g
    ((checked_domain m w h).program_destination_bound pid hp) hg
  have sz := sizes m w h .scale
  simp only [elements,elementBytes] at sz
  change (request m w).destination pid*(request m w).groups+g<(m .scale).numel
  omega

 theorem code_span (m : Inputs) (w : Nat → MetadataWord) (h : InputContract m w) (a : Nat)
    (ha : CodeTouched m w a) : base m .output≤a ∧ a<base m .output+(m .output).numel := by
  obtain ⟨pid,g,c,hp,hg,hc,rfl⟩ := ha
  have hb := output_bound m w h pid g c hp hg hc
  omega

 theorem scale_span (m : Inputs) (w : Nat → MetadataWord) (h : InputContract m w) (a : Nat)
    (ha : ScaleTouched m w a) : base m .scale≤a ∧ a<base m .scale+4*(m .scale).numel := by
  obtain ⟨pid,g,b,hp,hg,rfl⟩ := ha
  have hb := scale_bound m w h pid g hp hg
  have hi := b.isLt
  omega

 theorem disjoint (m : Inputs) (w : Nat → MetadataWord) (h : InputContract m w) (a : Nat)
    (hs : ScaleTouched m w a) : ¬CodeTouched m w a := by
  intro ho
  have hc := code_span m w h a ho
  have hb := scale_span m w h a hs
  have sep := h.separate .output .scale (by decide)
  simp only [elementBytes] at sep
  omega

 theorem post_code (m : Inputs) (w : Nat → MetadataWord) (before : Heap) (final : State Word)
    (pid g c : Nat) (hp : pid<(request m w).programs) (hg : g<(request m w).groups) (hc : c<8) :
    post m w before final (base m .output+codeOffset (parameters m w) ((request m w).destination pid) g c)=
      byteOfInt ((final.mem "O" (codeOffset (parameters m w) ((request m w).destination pid) g c)).read .int) := by
  have ht : CodeTouched m w (base m .output+codeOffset (parameters m w) ((request m w).destination pid) g c) := ⟨pid,g,c,hp,hg,hc,rfl⟩
  simp [post,ht]

 theorem post_scale (m : Inputs) (w : Nat → MetadataWord) (h : InputContract m w) (before : Heap) (final : State Word)
    (pid g : Nat) (b : Fin 4) (hp : pid<(request m w).programs) (hg : g<(request m w).groups) :
    post m w before final (base m .scale+4*scaleOffset (parameters m w) ((request m w).destination pid) g+b.val)=
      scaleByte ((final.mem "S" (scaleOffset (parameters m w) ((request m w).destination pid) g)).read .real) b := by
  have ht : ScaleTouched m w (base m .scale+4*scaleOffset (parameters m w) ((request m w).destination pid) g+b.val) := ⟨pid,g,b,hp,hg,rfl⟩
  have hn := disjoint m w h _ ht
  have hb := b.isLt
  have div : (base m .scale+4*scaleOffset (parameters m w) ((request m w).destination pid) g+b.val-base m .scale)/4=
      scaleOffset (parameters m w) ((request m w).destination pid) g := by omega
  have mod : (base m .scale+4*scaleOffset (parameters m w) ((request m w).destination pid) g+b.val-base m .scale)%4=b.val := by omega
  simp only [post,if_neg hn,if_pos ht,div]
  congr 1
  exact Fin.ext mod

 theorem post_frame (m : Inputs) (w : Nat → MetadataWord) (before : Heap) (final : State Word)
    (a : Nat) (ha : ¬Touched m w a) : post m w before final a=before a := by
  simp only [Touched,not_or] at ha
  simp [post,ha.1,ha.2]

/-- The physical target uses independent indexed numerical functions. It
contains neither an execution nor a source-final-state premise. -/
 def Result (m : Inputs) (w : Nat → MetadataWord) (before after : Heap)
    (codes : Nat → Nat → Nat → BitVec 8) (scales : Nat → Nat → Word) : Prop :=
  (∀ pid g c,pid<(request m w).programs → g<(request m w).groups → c<8 →
    after (base m .output+codeOffset (parameters m w) ((request m w).destination pid) g c)=byteOfBits (codes pid g c)) ∧
  (∀ pid g,pid<(request m w).programs → g<(request m w).groups → ∀ b : Fin 4,
    after (base m .scale+4*scaleOffset (parameters m w) ((request m w).destination pid) g+b.val)=scaleByte (scales pid g) b) ∧
  (∀ a,¬Touched m w a → after a=before a)

 theorem of_source (m : Inputs) (w : Nat → MetadataWord) (h : InputContract m w) (before : Heap)
    (initial final : State Word) (codes : Nat → Nat → Nat → BitVec 8) (scales : Nat → Nat → Word)
    (result : KernelWhole.Result m w (fun pid g c => (codes pid g c).toInt) scales initial final) :
    Result m w before (post m w before final) codes scales := by
  refine ⟨?_,?_,post_frame m w before final⟩
  · intro pid g c hp hg hc
    rw [post_code m w before final pid g c hp hg hc,result.1 pid g c hp hg hc,Cell.read_same,byte_roundtrip]
  · intro pid g hp hg b
    rw [post_scale m w h before final pid g b hp hg,result.2.1 pid g hp hg,Cell.read_same]

 theorem logical_outputs (t : Tensors) (laws : ∀ r,Laws (t r)) (w : Nat → MetadataWord)
    (h : InputContract (metadata t) w) (before after : Heap)
    (codes : Nat → Nat → Nat → BitVec 8) (scales : Nat → Nat → Word)
    (result : Result (metadata t) w before after codes scales) :
    (∀ pid g c,pid<(request (metadata t) w).programs → g<(request (metadata t) w).groups → c<8 →
      logicalInt8 after (t .output) (codeOffset (parameters (metadata t) w) ((request (metadata t) w).destination pid) g c)=codes pid g c) ∧
    (∀ pid g,pid<(request (metadata t) w).programs → g<(request (metadata t) w).groups →
      logicalFP32 after (t .scale) (scaleOffset (parameters (metadata t) w) ((request (metadata t) w).destination pid) g)=scales pid g) := by
  constructor
  · intro pid g c hp hg hc
    rw [logicalInt8,direct_address (t .output) (laws _) (h.contiguous _) _ (output_bound _ _ h pid g c hp hg hc)]
    simp only [elementBytes,Nat.one_mul]
    change BitVec.ofNat 8 (after (base (metadata t) .output+codeOffset (parameters (metadata t) w) ((request (metadata t) w).destination pid) g c)).val=_
    rw [result.1 pid g c hp hg hc]
    simp [byteOfBits]
  · intro pid g hp hg
    rw [logicalFP32,direct_address (t .scale) (laws _) (h.contiguous _) _ (scale_bound _ _ h pid g hp hg)]
    change decodeWord (fun b => after (base (metadata t) .scale+4*scaleOffset (parameters (metadata t) w) ((request (metadata t) w).destination pid) g+b.val))=_
    simp only [result.2.1 pid g hp hg]
    exact decode_scale _

#axiomsClean decode_scale
#axiomsClean encode_decode
#axiomsClean of_source
#axiomsClean logical_outputs
#axiomsClean post_frame
#specNonCircular Result avoiding [Kernel.kernel,FP.Structural.exec,KernelGrid.run,post]
end
end VeriTile.Bench.Repairs.QuantizeKvCopy.KernelPhysical
