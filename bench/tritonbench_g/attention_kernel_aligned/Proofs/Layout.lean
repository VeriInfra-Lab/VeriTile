import bench.tritonbench_g.attention_kernel_aligned.Proofs.Parameters
namespace VeriTile.Bench.Repairs.AttentionKernelAligned.Layout
open VeriTile.Triton
set_option maxHeartbeats 4000000

def offset {ι : Type} (stride : ι → Nat) (axes : List ι) (x : ι → Nat) : Nat :=
  (axes.map fun a=>x a*stride a).sum

def width {ι : Type} (extent stride : ι → Nat) (axes : List ι) : Nat :=
  1+(axes.map fun a=>(extent a-1)*stride a).sum

/-- Reversing the Python sorted-axis scan gives this identical span guard. -/
def Guarded {ι : Type} (extent stride : ι → Nat) : List ι → Prop
  | []=>True
  | a::rest=>(1<extent a → width extent stride rest≤stride a) ∧ Guarded extent stride rest

theorem bounded {ι : Type} (extent stride : ι → Nat) (axes : List ι) (x : ι → Nat)
    (hx : ∀ a∈axes,x a<extent a) : offset stride axes x<width extent stride axes := by
  have hsum : (axes.map fun a=>x a*stride a).sum≤(axes.map fun a=>(extent a-1)*stride a).sum := by
    induction axes with
    | nil=>simp
    | cons a rest ih=>
      simp only [List.map_cons,List.sum_cons]
      have ha:=Nat.mul_le_mul_right (stride a) (show x a≤extent a-1 by have:=hx a (by simp);omega)
      have ht:=ih (fun b hb=>hx b (by simp [hb]))
      omega
  simp only [offset,width]
  omega

theorem injective {ι : Type} (extent stride : ι → Nat) (axes : List ι)
    (guard : Guarded extent stride axes) (x y : ι → Nat)
    (hx : ∀ a∈axes,x a<extent a) (hy : ∀ a∈axes,y a<extent a)
    (he : offset stride axes x=offset stride axes y) : ∀ a∈axes,x a=y a := by
  induction axes with
  | nil=>simp
  | cons a rest ih=>
    obtain ⟨ha,hrest⟩:=guard
    have hxa:=hx a (by simp)
    have hya:=hy a (by simp)
    have hxr : ∀ b∈rest,x b<extent b := fun b hb=>hx b (by simp [hb])
    have hyr : ∀ b∈rest,y b<extent b := fun b hb=>hy b (by simp [hb])
    have hhead : x a=y a := by
      by_cases hext : 1<extent a
      · have hxb:=lt_of_lt_of_le (bounded extent stride rest x hxr) (ha hext)
        have hyb:=lt_of_lt_of_le (bounded extent stride rest y hyr) (ha hext)
        have hs : 0<stride a := by omega
        have dx : (x a*stride a+offset stride rest x)/stride a=x a := by
          rw [Nat.add_comm,Nat.add_mul_div_right _ _ hs,Nat.div_eq_of_lt hxb]
          omega
        have dy : (y a*stride a+offset stride rest y)/stride a=y a := by
          rw [Nat.add_comm,Nat.add_mul_div_right _ _ hs,Nat.div_eq_of_lt hyb]
          omega
        have hd:=congrArg (fun n=>n/stride a) he
        change (x a*stride a+offset stride rest x)/stride a=(y a*stride a+offset stride rest y)/stride a at hd
        simpa only [dx,dy] using hd
      · omega
    have htail : offset stride rest x=offset stride rest y := by
      change x a*stride a+offset stride rest x=y a*stride a+offset stride rest y at he
      rw [hhead] at he
      omega
    intro b hb
    rcases List.mem_cons.mp hb with rfl|hb
    · exact hhead
    · exact ih hrest hxr hyr htail b hb

#axiomsClean injective
end VeriTile.Bench.Repairs.AttentionKernelAligned.Layout
