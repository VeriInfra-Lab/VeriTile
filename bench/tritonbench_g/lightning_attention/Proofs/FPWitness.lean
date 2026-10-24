import bench.tritonbench_g.lightning_attention.Proofs.FPContract
namespace VeriTile.Bench.TritonBenchG.LightningAttention.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
set_option maxHeartbeats 12000000
set_option maxRecDepth 35000
set_option linter.unusedSimpArgs false

def witnessParameters : Parameters := ⟨"Q","K","V","C","O",0,2,2,4,2,2⟩
def witnessInitial : State Word where
  mem := fun r o => .mk .fp32 (FP.Value.ofBits .fp32
    (if r=("Q":RegionName) then
       if o=20 then 0x80000000 else if o=21 then 0x7fc01234 else
       if o=0 ∨ o=1 ∨ o=2 ∨ o=4 ∨ o=5 ∨ o=6 then 0x3f800000 else
        if o=3 ∨ o=7 then 0x40000000 else 0x42c60000
     else if r=("K":RegionName) then
       if o=0 ∨ o=4 ∨ o=7 then 0x3f800000 else
        if o=1 then 0x40000000 else
        if o=2 then 0x40400000 else
        if o=3 then 0x40800000 else 0x0
     else if r=("V":RegionName) then
       if o=0 ∨ o=10 then 0x40000000 else
        if o=1 ∨ o=11 then 0x40400000 else
        if o=2 ∨ o=14 then 0x40800000 else
        if o=3 ∨ o=15 then 0x40a00000 else 0x42c60000
     else if r=("C":RegionName) then
       if o=0 then 0x41200000 else
        if o=1 then 0x41a00000 else
        if o=2 then 0x41f00000 else
        if o=3 then 0x42200000 else 0x0
     else 0x42c60000))
  regs := fun _ _ _ => none
  pids := fun _ => 1
  numPids := fun _ => 2
  undef := fun d _ _ => defaultValue d

def lane00 : TileIndex [2,2] := (⟨0,by decide⟩,⟨0,by decide⟩,())
def lane01 : TileIndex [2,2] := (⟨0,by decide⟩,⟨1,by decide⟩,())
def lane10 : TileIndex [2,2] := (⟨1,by decide⟩,⟨0,by decide⟩,())
def lane11 : TileIndex [2,2] := (⟨1,by decide⟩,⟨1,by decide⟩,())

theorem kv_witness_values :
    wordKV .ieeeValue (softwareDot .ieeeValue) witnessParameters witnessInitial lane00 = FP.Value.ofBits .fp32 0x41a00000 ∧
    wordKV .ieeeValue (softwareDot .ieeeValue) witnessParameters witnessInitial lane01 = FP.Value.ofBits .fp32 0x42040000 ∧
    wordKV .ieeeValue (softwareDot .ieeeValue) witnessParameters witnessInitial lane10 = FP.Value.ofBits .fp32 0x42500000 ∧
    wordKV .ieeeValue (softwareDot .ieeeValue) witnessParameters witnessInitial lane11 = FP.Value.ofBits .fp32 0x428a0000 := by
  decide +kernel

theorem kv_bit_success (rest : Algebra Word) :
    ∃ final, FP.Structural.exec (backend (software rest .ieeeValue) (softwareDot .ieeeValue))
      (kvSource witnessParameters) witnessInitial=some final ∧
      final.mem "O" 0=.mk .fp32 (FP.Value.ofBits .fp32 0x41a00000) ∧
      final.mem "O" 1=.mk .fp32 (FP.Value.ofBits .fp32 0x42040000) ∧
      final.mem "Q" 4=witnessInitial.mem "Q" 4 := by
  obtain ⟨final,he,hp⟩ := kv_software_correctness rest .ieeeValue (softwareDot .ieeeValue)
    witnessParameters witnessInitial
  refine ⟨final,he,?_,?_,hp.2 "Q" 4 (Or.inl (by decide))⟩
  · have hv := hp.1 lane00 (by decide)
    change final.mem "O" 0 = _ at hv
    exact hv.trans (congrArg (Cell.mk .fp32) kv_witness_values.1)
  · have hv := hp.1 lane01 (by decide)
    change final.mem "O" 1 = _ at hv
    exact hv.trans (congrArg (Cell.mk .fp32) kv_witness_values.2.1)
#axiomsClean kv_witness_values
#axiomsClean kv_bit_success

theorem inter_witness_values :
    wordInter (softwareDot .ieeeValue) witnessParameters witnessInitial lane00 = FP.Value.ofBits .fp32 0x42200000 ∧
    wordInter (softwareDot .ieeeValue) witnessParameters witnessInitial lane01 = FP.Value.ofBits .fp32 0x42700000 ∧
    wordInter (softwareDot .ieeeValue) witnessParameters witnessInitial lane10 = FP.Value.ofBits .fp32 0x428c0000 ∧
    wordInter (softwareDot .ieeeValue) witnessParameters witnessInitial lane11 = FP.Value.ofBits .fp32 0x42c80000 := by
  decide +kernel

theorem inter_bit_success (rest : Algebra Word) :
    ∃ final, FP.Structural.exec (backend (software rest .ieeeValue) (softwareDot .ieeeValue))
      (interSource witnessParameters) witnessInitial=some final ∧
      final.mem "O" 0=.mk .fp32 (FP.Value.ofBits .fp32 0x42200000) ∧
      final.mem "O" 1=.mk .fp32 (FP.Value.ofBits .fp32 0x42700000) ∧
      final.mem "Q" 4=witnessInitial.mem "Q" 4 := by
  obtain ⟨final,he,hp⟩ := inter_software_correctness rest .ieeeValue (softwareDot .ieeeValue)
    witnessParameters witnessInitial
  refine ⟨final,he,?_,?_,hp.2 "Q" 4 (Or.inl (by decide))⟩
  · have hv := hp.1 lane00 (by decide)
    change final.mem "O" 0 = _ at hv
    exact hv.trans (congrArg (Cell.mk .fp32) inter_witness_values.1)
  · have hv := hp.1 lane01 (by decide)
    change final.mem "O" 1 = _ at hv
    exact hv.trans (congrArg (Cell.mk .fp32) inter_witness_values.2.1)
#axiomsClean inter_witness_values
#axiomsClean inter_bit_success

theorem body_witness_values :
    wordBody .ieeeValue (softwareDot .ieeeValue) witnessParameters witnessInitial lane00 = FP.Value.ofBits .fp32 0x42280000 ∧
    wordBody .ieeeValue (softwareDot .ieeeValue) witnessParameters witnessInitial lane01 = FP.Value.ofBits .fp32 0x427c0000 ∧
    wordBody .ieeeValue (softwareDot .ieeeValue) witnessParameters witnessInitial lane10 = FP.Value.ofBits .fp32 0x42a00000 ∧
    wordBody .ieeeValue (softwareDot .ieeeValue) witnessParameters witnessInitial lane11 = FP.Value.ofBits .fp32 0x42e20000 := by
  decide +kernel

theorem body_bit_success (rest : Algebra Word) :
    ∃ final, FP.Structural.exec (backend (software rest .ieeeValue) (softwareDot .ieeeValue))
      (bodySource witnessParameters) witnessInitial=some final ∧
      final.mem "O" 10=.mk .fp32 (FP.Value.ofBits .fp32 0x42280000) ∧
      final.mem "O" 11=.mk .fp32 (FP.Value.ofBits .fp32 0x427c0000) ∧
      final.mem "Q" 4=witnessInitial.mem "Q" 4 := by
  obtain ⟨final,he,hp⟩ := body_software_correctness rest .ieeeValue (softwareDot .ieeeValue)
    witnessParameters witnessInitial (by decide)
  refine ⟨final,he,?_,?_,hp.2 "Q" 4 (Or.inl (by decide))⟩
  · have hv := hp.1 lane00 (by decide)
    change final.mem "O" 10 = _ at hv
    exact hv.trans (congrArg (Cell.mk .fp32) body_witness_values.1)
  · have hv := hp.1 lane01 (by decide)
    change final.mem "O" 11 = _ at hv
    exact hv.trans (congrArg (Cell.mk .fp32) body_witness_values.2.1)
#axiomsClean body_witness_values
#axiomsClean body_bit_success

def tailParameters : Parameters := {witnessParameters with length:=1}
def tailInitial : State Word := {witnessInitial with mem:=fun r _ => .mk .fp32 (if r=("C":RegionName) then FP.Value.zero .fp32 else FP.Value.ofBits .fp32 0x3f800000)}
theorem tail_witness_value :
    wordBody .ieeeValue (softwareDot .ieeeValue) tailParameters tailInitial lane00 =
      FP.Value.ofBits .fp32 0x40000000 := by decide +kernel

theorem body_tail_bit_success (rest : Algebra Word) :
    ∃ final, FP.Structural.exec (backend (software rest .ieeeValue) (softwareDot .ieeeValue))
      (bodySource tailParameters) tailInitial=some final ∧
      final.mem "O" 6=.mk .fp32 (FP.Value.ofBits .fp32 0x40000000) ∧
      final.mem "O" 10=tailInitial.mem "O" 10 := by
  obtain ⟨final,he,hp⟩ := body_software_correctness rest .ieeeValue (softwareDot .ieeeValue)
    tailParameters tailInitial (by decide)
  refine ⟨final,he,?_,?_⟩
  · have hv := hp.1 lane00 (by decide)
    change final.mem "O" 6 = _ at hv
    exact hv.trans (congrArg (Cell.mk .fp32) tail_witness_value)
  · apply hp.2 "O" 10
    right
    intro i hi
    simp [bodyOffset,bodyActive,tailParameters,witnessParameters,tailInitial,witnessInitial] at hi ⊢
    have hc := i.2.1.isLt
    change i.2.1.val<2 at hc
    omega
#axiomsClean body_tail_bit_success

def gradParameters : Parameters := {witnessParameters with length:=3}
theorem grad_witness_values :
    wordGrad gradParameters witnessInitial lane00=FP.Value.ofBits .fp32 0x80000000 ∧
    wordGrad gradParameters witnessInitial lane01=FP.Value.ofBits .fp32 0x7fc01234 := by decide +kernel

theorem grad_bit_success (rest : Algebra Word) :
    ∃ final, FP.Structural.exec (backend (software rest .ieeeValue) (softwareDot .ieeeValue))
      (gradSource gradParameters) witnessInitial=some final ∧
      final.mem "O" 20=.mk .fp32 (FP.Value.ofBits .fp32 0x80000000) ∧
      final.mem "O" 21=.mk .fp32 (FP.Value.ofBits .fp32 0x7fc01234) ∧
      final.mem "O" 24=witnessInitial.mem "O" 24 := by
  have hinj : Function.Injective (gradOffset gradParameters witnessInitial) :=
    rowMajor_injective 2 2 4 2 12 0 (by decide)
  obtain ⟨final,he,hp⟩ := grad_software_correctness rest .ieeeValue (softwareDot .ieeeValue)
    gradParameters witnessInitial hinj
  refine ⟨final,he,?_,?_,?_⟩
  · have hv := hp.1 lane00 (by decide)
    change final.mem "O" 20 = _ at hv
    exact hv.trans (congrArg (Cell.mk .fp32) grad_witness_values.1)
  · have hv := hp.1 lane01 (by decide)
    change final.mem "O" 21 = _ at hv
    exact hv.trans (congrArg (Cell.mk .fp32) grad_witness_values.2)
  · apply hp.2 "O" 24
    right
    intro i hi
    simp [gradOffset,gradActive,gradParameters,witnessParameters,witnessInitial] at hi ⊢
    have hc := i.2.1.isLt
    change i.2.1.val<2 at hc
    omega
#axiomsClean grad_witness_values
#axiomsClean grad_bit_success
end
end VeriTile.Bench.TritonBenchG.LightningAttention.FPValue
