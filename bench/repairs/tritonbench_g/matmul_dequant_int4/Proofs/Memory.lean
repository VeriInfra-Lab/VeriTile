import bench.repairs.tritonbench_g.matmul_dequant_int4.Proofs.Grid
namespace VeriTile.Bench.Repairs.MatmulDequantInt4
set_option maxHeartbeats 5000000
namespace Writes

def Readonly (destination : Region) (initial current : State α) :=
  (∀ r a,r≠destination → current.values r a=initial.values r a) ∧ current.packed=initial.packed
def Stable (destination : Region) (compute : State α → Work → α) :=
  ∀ initial current,Readonly destination initial current → ∀ w,compute current w=compute initial w
def step (destination : Region) (columns : Nat) (compute : State α → Work → α)
    (w : Work) (m : State α) := store m destination (w.row*columns+w.col) (compute m w)
def run (destination : Region) (columns : Nat) (compute : State α → Work → α) : List Work → State α → State α
  | [],m => m
  | w::ws,m => run destination columns compute ws (step destination columns compute w m)

theorem reflexive (destination : Region) (m : State α) : Readonly destination m m :=
  ⟨fun _ _ _ => rfl,rfl⟩
theorem step_readonly (d : Region) (columns : Nat) (compute : State α → Work → α)
    (initial current : State α) (h : Readonly d initial current) (w : Work) :
    Readonly d initial (step d columns compute w current) :=
  ⟨fun r a hr => by simp [step,store,hr,h.1 r a hr],h.2⟩
theorem run_readonly (d : Region) (columns : Nat) (compute : State α → Work → α)
    (order : List Work) (initial current : State α) (h : Readonly d initial current) :
    Readonly d initial (run d columns compute order current) := by
  induction order generalizing current with
  | nil => exact h
  | cons w ws ih => exact ih _ (step_readonly d columns compute initial current h w)

theorem result (d : Region) (columns rows : Nat) (compute : State α → Work → α)
    (stable : Stable d compute) (order : List Work) (initial current : State α)
    (same : Readonly d initial current) (active : ∀ w∈order,w.Active rows columns)
    (w : Work) (hw : w.Active rows columns) :
    (run d columns compute order current).values d (w.row*columns+w.col)=
      if w∈order then compute initial w else current.values d (w.row*columns+w.col) := by
  induction order generalizing current with
  | nil => simp [run]
  | cons v vs ih =>
    have hv := active v (by simp)
    have he : w.row*columns+w.col=v.row*columns+v.col ↔ w=v :=
      ⟨address_injective columns w v hw.2 hv.2,fun h => by rw [h]⟩
    change (run d columns compute vs (step d columns compute v current)).values d _=_
    rw [ih _ (step_readonly d columns compute initial current same v)
      (fun x hx => active x (by simp [hx]))]
    simp only [List.mem_cons,step,store,true_and,he,stable initial current same v]
    by_cases hvw : w=v
    · subst v;simp
    · by_cases hmem : w∈vs <;> simp [hvw,hmem]

theorem frame (d : Region) (columns : Nat) (compute : State α → Work → α)
    (order : List Work) (initial : State α) (r : Region) (a : Nat)
    (outside : ∀ w∈order,¬(r=d ∧ a=w.row*columns+w.col)) :
    (run d columns compute order initial).values r a=initial.values r a := by
  induction order generalizing initial with
  | nil => rfl
  | cons w ws ih =>
    change (run d columns compute ws (step d columns compute w initial)).values r a=_
    rw [ih _ (fun v hv => outside v (by simp [hv]))]
    simp [step,store,outside w (by simp)]

theorem complete_result (d : Region) (compute : State α → Work → α) (stable : Stable d compute)
    (g : Grid rows columns) (order : List Work) (complete : g.Complete order)
    (initial : State α) (w : Work) (hw : w.Active rows columns) :
    (run d columns compute order initial).values d (w.row*columns+w.col)=compute initial w := by
  rw [result d columns rows compute stable order initial initial (reflexive _ _)
    (fun v hv => (g.complete_mem order complete v).mp hv) w hw]
  simp [(g.complete_mem order complete w).mpr hw]
theorem complete_frame (d : Region) (compute : State α → Work → α)
    (g : Grid rows columns) (order : List Work) (complete : g.Complete order)
    (initial : State α) (r : Region) (a : Nat) (outside : r≠d ∨ rows*columns≤a) :
    (run d columns compute order initial).values r a=initial.values r a := by
  apply frame
  intro w hw ⟨hr,ha⟩
  have active := (g.complete_mem order complete w).mp hw
  have bound := rowMajor_bound rows columns w.row w.col active.1 active.2
  rcases outside with h|h
  · exact h hr
  · omega

end Writes

theorem dequant_stable (s : Shape) (p : DequantOps α) : Writes.Stable .dequant (dequantValue s p) := by
  intro initial current same w
  have scales := same.1 .scale (sAddress s w.row w.col) (by decide)
  simp [dequantValue,difference,same.2,scales]
theorem dequant_run_writes (s : Shape) (p : DequantOps α) (order : List Work) (initial : State α) :
    dequantRun s p order initial=Writes.run .dequant s.n (dequantValue s p) order initial := by
  induction order generalizing initial with
  | nil => rfl
  | cons w ws ih => exact ih _
theorem dequant_complete (s : Shape) (p : DequantOps α) (g : Grid s.k s.n)
    (order : List Work) (complete : g.Complete order) (initial : State α)
    (k n : Nat) (hk : k<s.k) (hn : n<s.n) :
    (dequantRun s p order initial).values .dequant (k*s.n+n)=dequantValue s p initial ⟨k,n⟩ := by
  rw [dequant_run_writes]
  exact Writes.complete_result .dequant _ (dequant_stable s p) g order complete initial ⟨k,n⟩ ⟨hk,hn⟩
theorem dequant_frame (s : Shape) (p : DequantOps α) (g : Grid s.k s.n)
    (order : List Work) (complete : g.Complete order) (initial : State α)
    (r : Region) (a : Nat) (outside : r≠.dequant ∨ s.k*s.n≤a) :
    (dequantRun s p order initial).values r a=initial.values r a := by
  rw [dequant_run_writes]
  exact Writes.complete_frame .dequant _ g order complete initial r a outside

theorem fused_stable (s : Shape) (t : Tile) (p : FusedOps α β)
    (ng : Bool) (fusion : Nat → Nat → Fusion) : Writes.Stable .output (fusedValue s t p ng fusion) := by
  intro initial current same w
  have a : current.values .a=initial.values .a := funext (fun a => same.1 .a a (by decide))
  have scales : current.values .scale=initial.values .scale := funext (fun a => same.1 .scale a (by decide))
  simp [fusedValue,fusedBlock,fusedOperandB,same.2,a,scales]
theorem fused_run_writes (s : Shape) (t : Tile) (p : FusedOps α β)
    (ng : Bool) (fusion : Nat → Nat → Fusion) (order : List Work) (initial : State α) :
    fusedRun s t p ng fusion order initial=
      Writes.run .output s.n (fusedValue s t p ng fusion) order initial := by
  induction order generalizing initial with
  | nil => rfl
  | cons w ws ih => exact ih _
theorem fused_complete (s : Shape) (t : Tile) (p : FusedOps α β)
    (ng : Bool) (fusion : Nat → Nat → Fusion) (g : Grid s.m s.n)
    (order : List Work) (complete : g.Complete order) (initial : State α)
    (r n : Nat) (hr : r<s.m) (hn : n<s.n) :
    (fusedRun s t p ng fusion order initial).values .output (r*s.n+n)=fusedValue s t p ng fusion initial ⟨r,n⟩ := by
  rw [fused_run_writes]
  exact Writes.complete_result .output _ (fused_stable s t p ng fusion) g order complete initial ⟨r,n⟩ ⟨hr,hn⟩
theorem fused_frame (s : Shape) (t : Tile) (p : FusedOps α β)
    (ng : Bool) (fusion : Nat → Nat → Fusion) (g : Grid s.m s.n)
    (order : List Work) (complete : g.Complete order) (initial : State α)
    (r : Region) (a : Nat) (outside : r≠.output ∨ s.m*s.n≤a) :
    (fusedRun s t p ng fusion order initial).values r a=initial.values r a := by
  rw [fused_run_writes]
  exact Writes.complete_frame .output _ g order complete initial r a outside

#axiomsClean Writes.result
#axiomsClean Writes.complete_result
#axiomsClean Writes.complete_frame
#axiomsClean dequant_complete
#axiomsClean dequant_frame
#axiomsClean fused_complete
#axiomsClean fused_frame
end VeriTile.Bench.Repairs.MatmulDequantInt4
