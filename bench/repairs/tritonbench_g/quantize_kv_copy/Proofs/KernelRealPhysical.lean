import bench.repairs.tritonbench_g.quantize_kv_copy.Proofs.KernelPublic

namespace VeriTile.Bench.Repairs.QuantizeKvCopy.KernelRealPhysical
open VeriTile.Triton FP.Structural Kernel KernelModels KernelWhole KernelTensor KernelPublic
noncomputable section
open Classical
set_option maxHeartbeats 2500000

def ScaleCell (m : Inputs) (w : Nat → MetadataWord) (a : Nat) : Prop :=
  ∃ pid g,pid<(request m w).programs ∧ g<(request m w).groups ∧
    a=base m .scale+4*scaleOffset (parameters m w) ((request m w).destination pid) g

/-- Real scale cells remain unrestricted Reals. Code bytes use the same source
INT8 store representation. Both maps are projections of one source-final state. -/
def codePost (m : Inputs) (w : Nat → MetadataWord) (before : Nat → BitVec 8) (final : State ℝ) (a : Nat) : BitVec 8 :=
  if KernelPhysical.CodeTouched m w a then BitVec.ofInt 8 ((final.mem "O" (a-base m .output)).read .int) else before a

def scalePost (m : Inputs) (w : Nat → MetadataWord) (before : Nat → ℝ) (final : State ℝ) (a : Nat) : ℝ :=
  if ScaleCell m w a then (final.mem "S" ((a-base m .scale)/4)).read .real else before a

/-- Independent arbitrary-Real physical target at actual caller Tensor
coordinates, with the complete code-byte and Real-scale-cell exterior frames. -/
def Result (t : Tensors) (w : Nat → MetadataWord) (realBefore realAfter : Nat → ℝ)
    (codeBefore codeAfter : Nat → BitVec 8) : Prop :=
  let m := metadata t
  let x := realValues m w realBefore
  (∀ pid g c,pid<(request m w).programs → g<(request m w).groups → c<8 →
    codeAfter (address (t .output) (codeOffset (parameters m w) ((request m w).destination pid) g c))=realCode (x pid g) c) ∧
  (∀ pid g,pid<(request m w).programs → g<(request m w).groups →
    realAfter (address (t .scale) (scaleOffset (parameters m w) ((request m w).destination pid) g))=realScale (x pid g)) ∧
  (∀ a,¬KernelPhysical.CodeTouched m w a → codeAfter a=codeBefore a) ∧
  (∀ a,¬ScaleCell m w a → realAfter a=realBefore a)

theorem input_logical (t : Tensors) (laws : ∀ r,Laws (t r)) (w : Nat → MetadataWord)
    (h : InputContract (metadata t) w) (realMemory : Nat → ℝ)
    (pid g c : Nat) (hp : pid<(request (metadata t) w).programs) (hg : g<(request (metadata t) w).groups) (hc : c<8) :
    realValues (metadata t) w realMemory pid g c=
      realMemory (address (t .input) (inputOffset (parameters (metadata t) w) pid g c)) := by
  rw [direct_address (t .input) (laws _) (h.contiguous _) _ (KernelTensor.input_bound _ _ h pid g c hp hg hc)]
  rfl

theorem of_source (t : Tensors) (laws : ∀ r,Laws (t r)) (w : Nat → MetadataWord)
    (h : InputContract (metadata t) w) (realMemory : Nat → ℝ) (codeMemory : Nat → BitVec 8) (final : State ℝ)
    (result : KernelWhole.Result (metadata t) w
      (fun pid g c => (realCode (realValues (metadata t) w realMemory pid g) c).toInt)
      (fun pid g => realScale (realValues (metadata t) w realMemory pid g))
      (realInitial (metadata t) w realMemory codeMemory) final) :
    Result t w realMemory (scalePost (metadata t) w realMemory final)
      codeMemory (codePost (metadata t) w codeMemory final) := by
  refine ⟨?_,?_,?_,?_⟩
  · intro pid g c hp hg hc
    rw [direct_address (t .output) (laws _) (h.contiguous _) _ (KernelPhysical.output_bound _ _ h pid g c hp hg hc)]
    simp only [elementBytes,Nat.one_mul]
    have ht : KernelPhysical.CodeTouched (metadata t) w
      (base (metadata t) .output+codeOffset (parameters (metadata t) w) ((request (metadata t) w).destination pid) g c) :=
      ⟨pid,g,c,hp,hg,hc,rfl⟩
    change codePost (metadata t) w codeMemory final (base (metadata t) .output+codeOffset (parameters (metadata t) w) ((request (metadata t) w).destination pid) g c)=_
    rw [codePost,if_pos ht,Nat.add_sub_cancel_left,result.1 pid g c hp hg hc,Cell.read_same]
    simp
  · intro pid g hp hg
    rw [direct_address (t .scale) (laws _) (h.contiguous _) _ (KernelPhysical.scale_bound _ _ h pid g hp hg)]
    have ht : ScaleCell (metadata t) w
      (base (metadata t) .scale+4*scaleOffset (parameters (metadata t) w) ((request (metadata t) w).destination pid) g) :=
      ⟨pid,g,hp,hg,rfl⟩
    change scalePost (metadata t) w realMemory final (base (metadata t) .scale+4*scaleOffset (parameters (metadata t) w) ((request (metadata t) w).destination pid) g)=_
    rw [scalePost,if_pos ht,Nat.add_sub_cancel_left,Nat.mul_div_cancel_left _ (by decide : 0<4),result.2.1 pid g hp hg,Cell.read_same]
  · intro a ha
    exact if_neg ha
  · intro a ha
    exact if_neg ha

theorem correct (t : Tensors) (laws : ∀ r,Laws (t r)) (before : Heap)
    (h : InputContract (metadata t) (words t before)) (realMemory : Nat → ℝ) (codeMemory : Nat → BitVec 8) :
    let m := metadata t
    let w := words t before
    let initial := realInitial m w realMemory codeMemory
    ∃ final,KernelGrid.run KernelModels.realOps (parameters m w) (List.range (request m w).programs) initial=some final ∧
      KernelWhole.Result m w (fun pid g c => (realCode (realValues m w realMemory pid g) c).toInt)
        (fun pid g => realScale (realValues m w realMemory pid g)) initial final ∧
      Result t w realMemory (scalePost m w realMemory final) codeMemory (codePost m w codeMemory final) := by
  obtain ⟨final,run,result⟩ := KernelPublic.mathematical t before h realMemory codeMemory
  exact ⟨final,run,result,of_source t laws _ h realMemory codeMemory final result⟩

#axiomsClean input_logical
#axiomsClean of_source
#axiomsClean correct
#specNonCircular Result avoiding [realInitial,codePost,scalePost,Kernel.kernel,FP.Structural.exec,KernelGrid.run]
end
end VeriTile.Bench.Repairs.QuantizeKvCopy.KernelRealPhysical
