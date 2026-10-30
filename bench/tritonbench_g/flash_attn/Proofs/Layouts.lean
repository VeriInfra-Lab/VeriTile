import bench.tritonbench_g.flash_attn.Proofs.FlashAttn_Grid
namespace VeriTile.Bench.Repairs.FlashAttn.Layouts
set_option maxHeartbeats 5000000

def extent (sizes strides : Fin 4 → Nat) (axes : List (Fin 4)) : Nat :=
  1+(axes.map (fun i=>(sizes i-1)*strides i)).sum

def packed (strides coords : Fin 4 → Nat) (axes : List (Fin 4)) : Nat :=
  (axes.map (fun i=>coords i*strides i)).sum

/-- Reverse of the host's sorted positive-stride span scan. Singleton axes
are deliberately ignored by the span comparison, just as in `_tensor`. -/
def Guard (sizes strides : Fin 4 → Nat) : List (Fin 4) → Prop
  | []=>True
  | i::is=>Guard sizes strides is ∧ (sizes i≤1 ∨ extent sizes strides is≤strides i)

theorem packed_bound (sizes strides coords : Fin 4 → Nat) (axes : List (Fin 4))
    (hc : ∀ i∈axes,coords i<sizes i) : packed strides coords axes<extent sizes strides axes := by
  unfold packed extent
  have hh : (axes.map (fun i=>coords i*strides i)).sum≤(axes.map (fun i=>(sizes i-1)*strides i)).sum := by
    apply List.sum_le_sum
    intro i hi
    exact Nat.mul_le_mul_right _ (show coords i≤sizes i-1 by have:=hc i hi;omega)
  omega

theorem packed_injective (sizes strides a b : Fin 4 → Nat) (axes : List (Fin 4))
    (guard : Guard sizes strides axes) (ha : ∀ i∈axes,a i<sizes i) (hb : ∀ i∈axes,b i<sizes i)
    (he : packed strides a axes=packed strides b axes) : ∀ i∈axes,a i=b i := by
  induction axes with
  | nil=>simp
  | cons j js ih=>
    have haj:=ha j (by simp)
    have hbj:=hb j (by simp)
    have hat : ∀ i∈js,a i<sizes i :=fun i hi=>ha i (by simp [hi])
    have hbt : ∀ i∈js,b i<sizes i :=fun i hi=>hb i (by simp [hi])
    have atbound:=packed_bound sizes strides a js hat
    have btbound:=packed_bound sizes strides b js hbt
    have eq : a j*strides j+packed strides a js=b j*strides j+packed strides b js := by
      simpa [packed] using he
    have hj : a j=b j := by
      rcases guard.2 with single|gap
      · omega
      · have hap : packed strides a js<strides j := by omega
        have hbp : packed strides b js<strides j := by omega
        have aa:=flatten_decode (strides j) (a j) (packed strides a js) hap
        have bb:=flatten_decode (strides j) (b j) (packed strides b js) hbp
        have hh:=congrArg (fun x=>x/strides j) eq
        simpa only [aa.1,bb.1] using hh
    have tail : packed strides a js=packed strides b js := by rw [hj] at eq;omega
    intro i hi
    rcases List.mem_cons.mp hi with rfl|hi
    · exact hj
    · exact ih guard.1 hat hbt tail i hi

def sizes (s : Shape) : Fin 4 → Nat := ![s.batches,s.heads,s.length,s.dim]
def strides (s : Strides) : Fin 4 → Nat := ![s.batch,s.head,s.row,s.dim]
def coords (b h n d : Nat) : Fin 4 → Nat := ![b,h,n,d]

def Admitted (s : Shape) (t : Strides) : Prop :=
  ∃ axes,axes.Perm [0,1,2,3] ∧ Guard (sizes s) (strides t) axes

theorem address_packed (t : Strides) (b h n d : Nat) (axes : List (Fin 4))
    (perm : axes.Perm [0,1,2,3]) : packed (strides t) (coords b h n d) axes=address t b h n d := by
  have hh:= (perm.map (fun i=>coords b h n d i*strides t i)).sum_eq
  simpa [packed,coords,strides,address,add_assoc] using hh

theorem active_coordinates (s : Shape) (b h n d : Nat) (active : (Work.output b h n d).active s) :
    ∀ i : Fin 4,coords b h n d i<sizes s i := by
  intro i
  rcases active with ⟨hb,hh,hn,hd⟩
  fin_cases i <;> simp [coords,sizes] <;> assumption

theorem output_injective (x : Layout) (admitted : Admitted x.shape x.o) : OutputInjective x := by
  obtain ⟨axes,perm,guard⟩:=admitted
  intro b h n d b' h' n' d' ha hb he
  have aa:=active_coordinates x.shape b h n d ha
  have bb:=active_coordinates x.shape b' h' n' d' hb
  have eq : packed (strides x.o) (coords b h n d) axes=packed (strides x.o) (coords b' h' n' d') axes := by
    rw [address_packed _ _ _ _ _ _ perm,address_packed _ _ _ _ _ _ perm,he]
  have same:=packed_injective (sizes x.shape) (strides x.o) (coords b h n d) (coords b' h' n' d') axes
    guard (fun i _=>aa i) (fun i _=>bb i) eq
  have c0:=same 0 (perm.mem_iff.mpr (by simp))
  have c1:=same 1 (perm.mem_iff.mpr (by simp))
  have c2:=same 2 (perm.mem_iff.mpr (by simp))
  have c3:=same 3 (perm.mem_iff.mpr (by simp))
  exact ⟨c0,c1,c2,c3⟩
#axiomsClean packed_bound
#axiomsClean packed_injective
#axiomsClean output_injective
end VeriTile.Bench.Repairs.FlashAttn.Layouts
