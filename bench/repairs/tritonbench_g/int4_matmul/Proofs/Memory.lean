import bench.repairs.tritonbench_g.int4_matmul.Spec
import bench.repairs.tritonbench_g.int4_matmul.Proofs.Geometry
namespace VeriTile.Bench.Repairs.Int4Matmul
set_option maxHeartbeats 4000000

def Readonly (initial current : State α) :=
  (∀ r a,r≠.output → current.values r a=initial.values r a) ∧ current.packed=initial.packed

theorem store_readonly (initial current : State α) (h : Readonly initial current)
    (a : Nat) (v : α) : Readonly initial (store current .output a v) := by
  exact ⟨fun r b hr => by simp [store,hr,h.1 r b hr],h.2⟩
theorem zero_readonly (s : Shape) (p : Ops α) (m : State α) : Readonly m (zeroOutput s p m) :=
  ⟨fun r a hr => by simp [zeroOutput,hr],rfl⟩
theorem partial_reference (s : Shape) (t : Tile) (p : Ops α) (initial current : State α)
    (w : Work) (active : s.Active t w) (same : Readonly initial current) :
    partialValue s t p current w=referencePartial s t p initial w := by
  have ha : current.values .a=initial.values .a := funext (fun a => same.1 .a a (by decide))
  have hs : current.values .scale=initial.values .scale := funext (fun a => same.1 .scale a (by decide))
  simp only [partialValue,blockStep,referencePartial,rounds,operandA,operandB,
    Nat.mod_eq_of_lt active.1,Nat.mod_eq_of_lt active.2.1,ha,hs,same.2,
    difference,dequantized,logicalQ,logicalZ,logicalA,logicalScale,aAddress,qAddress,sAddress,zAddress]
  congr 1
  funext seed block
  congr 1
  funext acc chunk
  congr 1 <;> funext i <;> congr 1 <;> unfold kIndex <;> ring

theorem step_frame (s : Shape) (t : Tile) (p : Ops α) (w : Work) (m : State α)
    (r : Region) (a : Nat) (h : ¬(r=.output ∧ a=cAddress s w.row w.col)) :
    (step s t p w m).values r a=m.values r a := by simp [step,store,h]

theorem execute_readonly (s : Shape) (t : Tile) (p : Ops α) (ws : List Work)
    (initial current : State α) (h : Readonly initial current) :
    Readonly initial (execute s t p ws current) := by
  induction ws generalizing current with
  | nil => exact h
  | cons w ws ih => exact ih _ (store_readonly initial current h _ _)

theorem execute_frame (s : Shape) (t : Tile) (p : Ops α) (ws : List Work) (m : State α)
    (r : Region) (a : Nat) (h : ∀ w∈ws,¬(r=.output ∧ a=cAddress s w.row w.col)) :
    (execute s t p ws m).values r a=m.values r a := by
  induction ws generalizing m with
  | nil => rfl
  | cons w ws ih =>
    change (execute s t p ws (step s t p w m)).values r a=_
    rw [ih _ (fun v hv => h v (by simp [hv]))]
    exact step_frame s t p w m r a (h w (by simp))

theorem execute_result (s : Shape) (t : Tile) (p : Ops α) (ws : List Work)
    (initial current : State α) (same : Readonly initial current)
    (active : ∀ w∈ws,s.Active t w) (r c : Nat) (hc : c<s.n) :
    (execute s t p ws current).values .output (cAddress s r c)=
      referenceOutput s t p initial r c ws (current.values .output (cAddress s r c)) := by
  induction ws generalizing current with
  | nil => rfl
  | cons w ws ih =>
    have hw := active w (by simp)
    have address : cAddress s r c=cAddress s w.row w.col ↔ w.row=r ∧ w.col=c := by
      constructor
      · intro h
        have he := address_injective s ⟨r,c,0⟩ w hc hw.2.1 h
        exact ⟨he.1.symm,he.2.symm⟩
      · rintro ⟨hr,hc⟩;simp [hr,hc]
    change (execute s t p ws (step s t p w current)).values .output (cAddress s r c)=_
    have hs : Readonly initial (step s t p w current) := by
      unfold step
      exact store_readonly initial current same _ _
    rw [ih (step s t p w current) hs (fun v hv => active v (by simp [hv]))]
    have hp := partial_reference s t p initial current w hw same
    simp only [referenceOutput,step,store,true_and,hp,address]
    split_ifs with he hsplit <;> simp_all [cAddress]
#axiomsClean partial_reference
#axiomsClean execute_readonly
#axiomsClean execute_frame
#axiomsClean execute_result
end VeriTile.Bench.Repairs.Int4Matmul
