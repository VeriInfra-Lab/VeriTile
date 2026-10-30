import bench.repairs.tritonbench_g.softmax_reducev_neginf_checked.IO
import bench.repairs.tritonbench_g.softmax_reducev_neginf_checked.Proofs.IOSourceSafety

namespace VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.PublicIO
open VeriTile Triton FP.Structural FPValue
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false
set_option linter.unusedSectionVars false
variable {A B : Type} [Inhabited A] [Inhabited B]
variable (M : Host.Job → Algebra B) (encode : A → B) (c : Host.Call)
  (observe : Fin c.batch → Fin c.heads → Fin c.dimensions → B → A)
  (counts : Fin c.batch → Nat) (work : List Host.Job)

theorem length_binding (xs : Inputs A c counts) (s : State B)
    (bound : (sourceIO M encode c observe counts work).InputBinding xs s)
    (b : Fin c.batch) (heads : 0 < c.heads) :
    (s.mem "Lengths" b.val).read .nat=length xs b := by
  simpa [sourceIO,memory,inputs,read,inputMask,mapValue,length] using
    bound (0 : Fin 5) (b,()) heads

theorem start_binding (xs : Inputs A c counts) (s : State B)
    (bound : (sourceIO M encode c observe counts work).InputBinding xs s)
    (b : Fin c.batch) (heads : 0 < c.heads) :
    (s.mem "Starts" b.val).read .nat=start xs b := by
  simpa [sourceIO,memory,inputs,read,inputMask,mapValue,start] using
    bound (1 : Fin 5) (b,()) heads

theorem page_binding (xs : Inputs A c counts) (s : State B)
    (bound : (sourceIO M encode c observe counts work).InputBinding xs s)
    (b : Fin c.batch) (heads : 0 < c.heads) (k : Fin (capacity counts)) (hk : k.val < counts b) :
    (s.mem "Locations" (b.val*c.maximum+(c.maximum-counts b)+k.val)).read .int=page xs b k := by
  simpa [sourceIO,memory,inputs,read,inputMask,mapValue,page] using
    bound (2 : Fin 5) (b,k,()) ⟨heads,hk⟩

theorem logit_binding (xs : Inputs A c counts) (s : State B)
    (bound : (sourceIO M encode c observe counts work).InputBinding xs s)
    (b : Fin c.batch) (h : Fin c.heads) (k : Fin (capacity counts)) (hk : k.val < counts b) :
    (s.mem "Logics" (h.val*c.tokens+(start xs b+k.val))).read .fp32=encode (logit xs b h k) := by
  simpa [sourceIO,memory,inputs,read,inputMask,mapValue,logit] using
    bound (3 : Fin 5) (b,h,k,()) hk

theorem value_binding (xs : Inputs A c counts) (s : State B)
    (bound : (sourceIO M encode c observe counts work).InputBinding xs s)
    (b : Fin c.batch) (h : Fin c.heads) (k : Fin (capacity counts)) (d : Fin c.dimensions)
    (hk : k.val < counts b) :
    (s.mem "V" (h.val*c.dimensions+d.val+(page xs b k*(c.heads*c.dimensions : Nat)).toNat)).read .fp32=
      encode (value xs b h k d) := by
  simpa [sourceIO,memory,inputs,read,inputMask,mapValue,value] using
    bound (4 : Fin 5) (b,h,k,d,()) hk

theorem source_length (xs : Inputs A c counts) (s : State B)
    (bound : (sourceIO M encode c observe counts work).InputBinding xs s)
    (hm : LengthsMatch xs) (b : Fin c.batch) (h : Fin c.heads) :
    sequenceLength (Host.parameters c) ((invocation (M ⟨b.val,h.val⟩) c ⟨b.val,h.val⟩).initial s)=counts b := by
  change (s.mem "Lengths" b.val).read .nat=counts b
  rw [length_binding M encode c observe counts work xs s bound b (Nat.zero_lt_of_lt h.isLt),hm b (Nat.zero_lt_of_lt h.isLt)]

theorem source_start (xs : Inputs A c counts) (s : State B)
    (bound : (sourceIO M encode c observe counts work).InputBinding xs s)
    (b : Fin c.batch) (h : Fin c.heads) :
    sequenceStart (Host.parameters c) ((invocation (M ⟨b.val,h.val⟩) c ⟨b.val,h.val⟩).initial s)=start xs b := by
  exact start_binding M encode c observe counts work xs s bound b (Nat.zero_lt_of_lt h.isLt)

theorem source_location (xs : Inputs A c counts) (s : State B)
    (bound : (sourceIO M encode c observe counts work).InputBinding xs s)
    (hm : LengthsMatch xs) (b : Fin c.batch) (h : Fin c.heads)
    (n : Nat) (i : Fin 64) (hi : n+i.val < counts b) :
    location (Host.parameters c) ((invocation (M ⟨b.val,h.val⟩) c ⟨b.val,h.val⟩).initial s) n i=
      page xs b ⟨n+i.val,lt_of_lt_of_le hi (count_le_capacity counts b)⟩ := by
  have len := source_length M encode c observe counts work xs s bound hm b h
  simp only [location,active,len,if_pos hi,locationBase,len]
  simpa [Host.parameters,invocation,Invocation.initial,bindRegisters,Region.cast,len,Nat.add_assoc] using
    page_binding M encode c observe counts work xs s bound b (Nat.zero_lt_of_lt h.isLt)
      ⟨n+i.val,lt_of_lt_of_le hi (count_le_capacity counts b)⟩ hi

theorem source_logit (xs : Inputs A c counts) (s : State B)
    (bound : (sourceIO M encode c observe counts work).InputBinding xs s)
    (b : Fin c.batch) (h : Fin c.heads) (n : Nat) (i : Fin 64) (hi : n+i.val < counts b) :
    let initial := (invocation (M ⟨b.val,h.val⟩) c ⟨b.val,h.val⟩).initial s
    (initial.mem "Logics" (logicOffset (Host.parameters c) initial n i)).read .fp32=
      encode (logitAt xs b h (n+i.val)) := by
  have st := source_start M encode c observe counts work xs s bound b h
  have hc := lt_of_lt_of_le hi (count_le_capacity counts b)
  simp only [logicOffset,st]
  simpa [Host.parameters,invocation,Invocation.initial,bindRegisters,Region.cast,
    st,logitAt,hc,Nat.add_assoc] using
    logit_binding M encode c observe counts work xs s bound b h ⟨n+i.val,hc⟩ hi

theorem source_value (xs : Inputs A c counts) (s : State B)
    (bound : (sourceIO M encode c observe counts work).InputBinding xs s)
    (hm : LengthsMatch xs) (b : Fin c.batch) (h : Fin c.heads)
    (n : Nat) (i : Fin 64) (d : Fin c.dimensions) (hi : n+i.val < counts b) :
    let initial := (invocation (M ⟨b.val,h.val⟩) c ⟨b.val,h.val⟩).initial s
    (initial.mem "V" (valueOffset (Host.parameters c) initial n (i,d,()))).read .fp32=
      encode (valueAt xs b h (n+i.val) d) := by
  have loc := source_location M encode c observe counts work xs s bound hm b h n i hi
  have hc := lt_of_lt_of_le hi (count_le_capacity counts b)
  simp only [valueOffset,loc]
  simpa [Host.parameters,invocation,Invocation.initial,bindRegisters,Region.cast,
    valueAt,hc] using value_binding M encode c observe counts work xs s bound b h ⟨n+i.val,hc⟩ d hi

theorem source_bounds (xs : Inputs A c counts) (s : State B)
    (bound : (sourceIO M encode c observe counts work).InputBinding xs s)
    (hm : LengthsMatch xs) (b : Fin c.batch) (h : Fin c.heads)
    (bounds : RegionName → Nat) (hb : (sourceIO M encode c observe counts work).InBounds bounds xs) :
    IOSourceSafety.Bounds (Host.parameters c)
      ((invocation (M ⟨b.val,h.val⟩) c ⟨b.val,h.val⟩).initial s) bounds := by
  have len := source_length M encode c observe counts work xs s bound hm b h
  have st := source_start M encode c observe counts work xs s bound b h
  constructor
  · exact hb.2.1 (0 : Fin 5) (b,()) (Nat.zero_lt_of_lt h.isLt)
  · exact hb.2.1 (1 : Fin 5) (b,()) (Nat.zero_lt_of_lt h.isLt)
  · intro n i hi
    have hi' : n+i.val < counts b := by simpa [active,len] using hi
    have hc := lt_of_lt_of_le hi' (count_le_capacity counts b)
    simp only [locationBase,len]
    simpa [Host.parameters,sourceIO,memory,inputs,read,
      invocation,Invocation.initial,bindRegisters,Region.cast,Nat.add_assoc] using
      hb.2.1 (2 : Fin 5) (b,⟨n+i.val,hc⟩,()) ⟨Nat.zero_lt_of_lt h.isLt,hi'⟩
  · intro n i hi
    have hi' : n+i.val < counts b := by simpa [active,len] using hi
    have hc := lt_of_lt_of_le hi' (count_le_capacity counts b)
    simp only [logicOffset,st]
    simpa [Host.parameters,sourceIO,memory,inputs,read,
      invocation,Invocation.initial,bindRegisters,Region.cast,Nat.add_assoc] using
      hb.2.1 (3 : Fin 5) (b,h,⟨n+i.val,hc⟩,()) hi'
  · intro n i hi
    have hi' : n+i.1.val < counts b := by simpa [active,len] using hi
    have hc := lt_of_lt_of_le hi' (count_le_capacity counts b)
    have loc := source_location M encode c observe counts work xs s bound hm b h n i.1 hi'
    simp only [valueOffset,loc]
    simpa [Host.parameters,sourceIO,memory,inputs,read,
      invocation,Invocation.initial,bindRegisters,Region.cast] using
      hb.2.1 (4 : Fin 5) (b,h,⟨n+i.1.val,hc⟩,i.2.1,()) hi'
  · intro d
    simpa [outputOffset,Host.parameters,sourceIO,memory,outputs,
      invocation,Invocation.initial,bindRegisters,Nat.add_mul,Nat.mul_assoc] using
      hb.2.2.1 (0 : Fin 1) (b,h,d.1,()) trivial

end
end VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.PublicIO
