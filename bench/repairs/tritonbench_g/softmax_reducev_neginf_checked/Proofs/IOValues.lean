import bench.repairs.tritonbench_g.softmax_reducev_neginf_checked.Proofs.IOBindings
import bench.repairs.tritonbench_g.softmax_reducev_neginf_checked.Targets

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

theorem logits_logical (xs : Inputs A c counts) (s : State B)
    (bound : (sourceIO M encode c observe counts work).InputBinding xs s)
    (hm : LengthsMatch xs) (b : Fin c.batch) (h : Fin c.heads) (n : Nat) :
    logits (M ⟨b.val,h.val⟩) (Host.parameters c)
      ((invocation (M ⟨b.val,h.val⟩) c ⟨b.val,h.val⟩).initial s) n=
    (fun i : TileIndex [64] =>
      (M ⟨b.val,h.val⟩).cast (some .fp32) .fp32 .real
        (if n+i.1.val < counts b then encode (logitAt xs b h (n+i.1.val))
         else (M ⟨b.val,h.val⟩).cast none .real .fp32 (M ⟨b.val,h.val⟩).negInf)) := by
  have len := source_length M encode c observe counts work xs s bound hm b h
  funext i
  simp only [logits,active,len]
  split
  · rename_i hi
    exact congrArg ((M ⟨b.val,h.val⟩).cast (some .fp32) .fp32 .real)
      (source_logit M encode c observe counts work xs s bound b h n i.1 hi)
  · rfl

theorem values_logical (xs : Inputs A c counts) (s : State B)
    (bound : (sourceIO M encode c observe counts work).InputBinding xs s)
    (hm : LengthsMatch xs) (b : Fin c.batch) (h : Fin c.heads) (n : Nat) :
    values (M ⟨b.val,h.val⟩) (Host.parameters c)
      ((invocation (M ⟨b.val,h.val⟩) c ⟨b.val,h.val⟩).initial s) n=
    (fun i : TileIndex [64,c.dimensions] =>
      (M ⟨b.val,h.val⟩).cast (some .fp32) .fp32 .real
        (if n+i.1.val < counts b then encode (valueAt xs b h (n+i.1.val) i.2.1)
         else (M ⟨b.val,h.val⟩).cast none .real .fp32
           ((M ⟨b.val,h.val⟩).literal none .real 0.0))) := by
  have len := source_length M encode c observe counts work xs s bound hm b h
  funext i
  simp only [values,active,len]
  split
  · rename_i hi
    exact congrArg ((M ⟨b.val,h.val⟩).cast (some .fp32) .fp32 .real)
      (source_value M encode c observe counts work xs s bound hm b h n i.1 i.2.1 hi)
  · rfl

theorem update_logical (xs : Inputs A c counts) (s : State B)
    (bound : (sourceIO M encode c observe counts work).InputBinding xs s)
    (hm : LengthsMatch xs) (b : Fin c.batch) (h : Fin c.heads) :
    update (M ⟨b.val,h.val⟩) (Host.parameters c)
      ((invocation (M ⟨b.val,h.val⟩) c ⟨b.val,h.val⟩).initial s)=
    orderedStep (M ⟨b.val,h.val⟩) c.dimensions (counts b)
      (fun k => encode (logitAt xs b h k)) (fun k d => encode (valueAt xs b h k d)) := by
  funext n a
  simp only [update,logits_logical M encode c observe counts work xs s bound hm,
    values_logical M encode c observe counts work xs s bound hm]
  rfl

theorem row_logical (xs : Inputs A c counts) (s : State B)
    (bound : (sourceIO M encode c observe counts work).InputBinding xs s)
    (hm : LengthsMatch xs) (b : Fin c.batch) (h : Fin c.heads) (d : Fin c.dimensions) :
    outputValues (M ⟨b.val,h.val⟩) (Host.parameters c)
      (finalAccumulator (M ⟨b.val,h.val⟩) (Host.parameters c)
        ((invocation (M ⟨b.val,h.val⟩) c ⟨b.val,h.val⟩).initial s)) (d,())=
    orderedRow (M ⟨b.val,h.val⟩) c.dimensions (counts b)
      (fun k => encode (logitAt xs b h k)) (fun k d => encode (valueAt xs b h k d)) d := by
  simp only [finalAccumulator,update_logical M encode c observe counts work xs s bound hm,
    source_length M encode c observe counts work xs s bound hm]
  rfl

theorem invocation_offset (j : Host.Job) (s : State B) (d : Index (Host.parameters c)) :
    outputOffset (Host.parameters c) ((invocation (M j) c j).initial s) d=Host.address c j d.1.val := by
  simp [outputOffset,Host.parameters,invocation,Invocation.initial,bindRegisters,
    Host.address,Nat.add_mul,Nat.mul_assoc]

theorem invocation_injective (j : Host.Job) (s : State B) :
    Function.Injective (outputOffset (Host.parameters c) ((invocation (M j) c j).initial s)) := by
  intro a b hab
  rw [invocation_offset,invocation_offset] at hab
  have eq : a.1=b.1 := Fin.ext (Nat.add_left_cancel hab)
  cases a; cases b; simp_all

theorem binding_of_frame (xs : Inputs A c counts) (s t : State B)
    (bound : (sourceIO M encode c observe counts work).InputBinding xs s)
    (frame : ∀ (r : RegionName) a, r≠"Out" → t.mem r a=s.mem r a) :
    (sourceIO M encode c observe counts work).InputBinding xs t := by
  intro port i hi
  rw [frame]
  · exact bound port i hi
  · change (inputs c counts)[port].buf≠"Out"
    fin_cases port <;> simp [inputs]

#axiomsClean row_logical
end
end VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.PublicIO
