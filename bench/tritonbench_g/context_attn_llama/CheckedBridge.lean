import bench.tritonbench_g.context_attn_llama.Checks

namespace VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.Checked
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnLlama Host Formula Program
set_option maxRecDepth 24000
noncomputable section

structure MetadataObservation (c : Call) (s : BlockState)
    (start sequence prompt request pages : Nat → Int) where
  (startWords sequenceWords promptWords requestWords pageWords : Nat → BitVec 32)
  startCPU : ∀ b,b < c.batches → start b=(startWords b).toInt
  sequenceCPU : ∀ b,b < c.batches → sequence b=(sequenceWords b).toInt
  promptCPU : ∀ b,b < c.batches → prompt b=(promptWords b).toInt
  requestCPU : ∀ b,b < c.batches → request b=(requestWords b).toInt
  pagesCPU : ∀ off,pages off=(pageWords off).toInt
  startMemory : ∀ b,b < c.batches → (Program.inputs s c.regions).start b=start b
  sequenceMemory : ∀ b,b < c.batches → (Program.inputs s c.regions).sequence b=sequence b
  promptMemory : ∀ b,b < c.batches → (Program.inputs s c.regions).prompt b=prompt b
  requestMemory : ∀ b,b < c.batches → (Program.inputs s c.regions).request b=request b
  pagesMemory : ∀ off,(Program.inputs s c.regions).pages off=pages off

def Labels (c : Call) : Prop := ∀ rg,rg∈c.inputs → c.regions.out≠rg

theorem bound3 (rows heads dims extent row head dim : Nat)
    (space : rows*heads*dims≤extent) (hr : row < rows) (hh : head < heads) (hd : dim < dims) :
    row*(heads*dims)+head*dims+dim < extent := by
  have hrow := Nat.mul_le_mul_right heads (Nat.succ_le_of_lt hr)
  simp only [Nat.succ_mul] at hrow
  have linear : row*heads+head<rows*heads := by omega
  have hdim := Nat.mul_le_mul_right dims (Nat.succ_le_of_lt linear)
  simp only [Nat.succ_mul,Nat.add_mul,Nat.mul_assoc] at hdim
  simp only [Nat.mul_assoc] at space
  omega

theorem unique3 (rows heads dims row row' head head' dim dim' : Nat)
    (hr : row < rows) (hr' : row' < rows) (hh : head < heads) (hh' : head' < heads)
    (hd : dim < dims) (hd' : dim' < dims)
    (eq : row*(heads*dims)+head*dims+dim=row'*(heads*dims)+head'*dims+dim') :
    row=row' ∧ head=head' ∧ dim=dim' := by
  have hD : 0<dims := by omega
  have hH : 0<heads := by omega
  have form : row*(heads*dims)+head*dims+dim=(row*heads+head)*dims+dim := by ring
  have form' : row'*(heads*dims)+head'*dims+dim'=(row'*heads+head')*dims+dim' := by ring
  rw [form,form'] at eq
  have e : row*heads+head=row'*heads+head' := by
    have divided := congrArg (· / dims) eq
    simpa [Nat.add_div,hD,Nat.ne_of_gt hD,Nat.div_eq_of_lt hd,Nat.div_eq_of_lt hd',
      Nat.mod_eq_of_lt hd,Nat.mod_eq_of_lt hd',Nat.not_le.mpr hd,Nat.not_le.mpr hd'] using divided
  have d : dim=dim' := by
    have reduced := congrArg (· % dims) eq
    simpa [Nat.add_mod,Nat.mod_eq_of_lt hd,Nat.mod_eq_of_lt hd'] using reduced
  have he : head=head' := by
    have reduced := congrArg (· % heads) e
    simpa [Nat.add_mod,Nat.mod_eq_of_lt hh,Nat.mod_eq_of_lt hh'] using reduced
  have rr : row=row' := by
    have divided := congrArg (· / heads) e
    simpa [Nat.add_div,hH,Nat.ne_of_gt hH,Nat.div_eq_of_lt hh,Nat.div_eq_of_lt hh',
      Nat.mod_eq_of_lt hh,Nat.mod_eq_of_lt hh',Nat.not_le.mpr hh,Nat.not_le.mpr hh'] using divided
  exact ⟨rr,he,d⟩

theorem valid (c : Call) (d : Dimensions) (A : FlatAlloc) (s : BlockState)
    (start sequence prompt request pages : Nat → Int)
    (g : geometryChecks c d A.extent) (m : metadataChecks c start sequence prompt request pages)
    (o : MetadataObservation c s start sequence prompt request pages) (labels : Labels c) : Valid c A s := by
  rcases g with ⟨strides,reqRow,reqToken,kvPositive,groupPositive,grouped,capacity,packed,qSpace,kSpace,metaSpace,pageSpace⟩
  rcases strides with ⟨qr,qh,qd,orr,oh,od,kr,kh,kd,vr,vh,vd⟩
  have kv_bound (h : Nat) (hh : h < c.layout.heads) : h/c.layout.kvGroup < c.kvHeads := by
    apply (Nat.div_lt_iff_lt_mul groupPositive).mpr
    simpa only [grouped] using hh
  have row_bound (b row : Nat) (hb : b < c.batches) (hr : row < c.length b) : c.start b+row < d.queryRows := by
    have capacity := (capacity ⟨b,hb⟩).1
    change c.start b+c.length b≤d.queryRows at capacity
    omega
  constructor
  · intro b hb; exact (o.startMemory b hb).trans (m.1 ⟨b,hb⟩).1
  · intro b hb; exact (o.sequenceMemory b hb).trans (m.1 ⟨b,hb⟩).2.1
  · intro b hb; exact (o.promptMemory b hb).trans (m.1 ⟨b,hb⟩).2.2.1
  · intro b hb; exact (o.requestMemory b hb).trans (m.1 ⟨b,hb⟩).2.2.2.1
  · intro b hb j hj; exact (o.pagesMemory _).trans ((m.1 ⟨b,hb⟩).2.2.2.2 ⟨j,hj⟩)
  · exact metaSpace _ (by simp)
  · exact metaSpace _ (by simp)
  · intro b hb j hj
    have cap := capacity ⟨b,hb⟩
    have one := Nat.mul_le_mul_right d.tokens (Nat.succ_le_of_lt cap.2.2.2.1)
    simp only [Nat.succ_mul] at one
    rw [reqRow,reqToken,Nat.one_mul]
    have token : j < d.tokens := lt_of_lt_of_le hj cap.2.2.1
    have request : c.request b*d.tokens+j < d.requests*d.tokens := by omega
    simpa only [Nat.mul_comm d.tokens] using lt_of_lt_of_le request pageSpace
  · exact metaSpace _ (by simp)
  · exact metaSpace _ (by simp)
  · exact groupPositive
  · exact grouped
  · intro b hb h hh row hr dim hd
    simp only [qr,qh,qd,Nat.mul_one]
    exact bound3 _ _ _ _ _ _ _ (qSpace _ (by simp)) (row_bound b row hb hr) hh hd
  · intro b hb h hh row hr dim hd
    simp only [kr,kh,kd,Nat.mul_one]
    exact bound3 _ _ _ _ _ _ _ (kSpace _ (by simp)) ((capacity ⟨b,hb⟩).2.2.2.2 ⟨row,hr⟩) (kv_bound h hh) hd
  · intro b hb h hh row hr dim hd
    simp only [vr,vh,vd,Nat.mul_one]
    exact bound3 _ _ _ _ _ _ _ (kSpace _ (by simp)) ((capacity ⟨b,hb⟩).2.2.2.2 ⟨row,hr⟩) (kv_bound h hh) hd
  · intro b hb h hh row hr dim hd
    simp only [Call.output,orr,oh,od,Nat.mul_one]
    exact bound3 _ _ _ _ _ _ _ (qSpace _ (by simp)) (row_bound b row hb hr) hh hd
  · intro b b' row row' hb hb' hr hr' eq
    by_cases beq : b=b'
    · exact ⟨beq,by subst b'; omega⟩
    · have sep := packed ⟨b,hb⟩ ⟨b',hb'⟩ (Nat.zero_lt_of_lt hr) (Nat.zero_lt_of_lt hr')
        (fun h => beq (congrArg Fin.val h))
      change c.start b+c.length b≤c.start b' ∨ c.start b'+c.length b'≤c.start b at sep
      rcases sep with sep|sep <;> omega
  · intro b b' h h' row row' dim dim' hb hb' hh hh' hr hr' hd hd' eq
    simp only [Call.output,orr,oh,od,Nat.mul_one] at eq
    exact unique3 _ _ _ _ _ _ _ _ _ (row_bound b row hb hr) (row_bound b' row' hb' hr') hh hh' hd hd' eq
  · exact labels

#axiomsClean valid
end
end VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.Checked
