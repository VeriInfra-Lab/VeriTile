import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.F8ConversionUtils.ByteStore
open VeriTile.Triton
noncomputable section
open Classical
abbrev Bytes := Nat → BitVec 8
theorem fold_value {β : Type} (step : Bytes → β → Bytes) (target : Bytes) (covers : β → Nat → Prop)
    (law : ∀ s c i, step s c i=if covers c i then target i else s i)
    (cs : List β) (scratch : Bytes) (i : Nat) :
    cs.foldl step scratch i=if ∃ c∈cs,covers c i then target i else scratch i := by
  induction cs generalizing scratch with
  | nil => simp
  | cons c cs ih =>
    simp only [List.foldl_cons,ih,law]
    by_cases tail : ∃ d∈cs,covers d i
    · simp [tail,List.mem_cons]
    · by_cases head : covers c i <;> simp [tail,head,List.mem_cons]

def writeWord {width : Nat} (address : Nat) (value : Fin width → BitVec 8) (before : Bytes) : Bytes :=
  (List.finRange width).foldl (fun mem k => Function.update mem (address+k.val) (value k)) before

theorem writeWord_value {width : Nat} (address : Nat) (value : Fin width → BitVec 8)
    (before : Bytes) (k : Fin width) : writeWord address value before (address+k)=value k := by
  classical
  let target : Bytes := fun q => if h : q-address<width then value ⟨q-address,h⟩ else 0
  have law (memory : Bytes) (j : Fin width) (q : Nat) :
      Function.update memory (address+j.val) (value j) q=
        if address+j.val=q then target q else memory q := by
    by_cases h : address+j.val=q
    · subst q
      simp [target,j.isLt]
    · simp only [Function.update_of_ne (Ne.symm h),if_neg h]
  rw [writeWord,fold_value _ target (fun (j : Fin width) q => address+j.val=q)
    (by intro memory j q; by_cases h : address+j.val=q <;>
        simpa only [h,if_true,if_false] using law memory j q)]
  have hit : ∃ j∈List.finRange width,address+j.val=address+k.val := ⟨k,List.mem_finRange k,rfl⟩
  rw [if_pos hit]
  simp only [target,Nat.add_sub_cancel_left,dif_pos k.isLt]

theorem writeWord_frame {width : Nat} (address : Nat) (value : Fin width → BitVec 8)
    (before : Bytes) (q : Nat) (outside : ∀ k : Fin width,q≠address+k.val) :
    writeWord address value before q=before q := by
  unfold writeWord
  generalize List.finRange width=ks
  induction ks generalizing before with
  | nil => rfl
  | cons k ks ih =>
    rw [List.foldl_cons,ih,Function.update_of_ne (outside k)]

#axiomsClean writeWord_value
#axiomsClean writeWord_frame
end
end VeriTile.Bench.Repairs.F8ConversionUtils.ByteStore
