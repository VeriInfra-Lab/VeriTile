import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.Stages
import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.Split

namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.Preparation
open Checked Binding Machine
noncomputable section

def coordinates (c : Shape) (r : Region) (p t j : Nat) : List Nat :=
  let h := p/c.tiles
  let k := (p%c.tiles/c.VT)*64+j/16
  let v := (p%c.tiles%c.VT)*16+j%16
  match r with
  | .x | .incoming => [h/c.H,h%c.H,t,k,v]
  | .gate => [p/c.H,p%c.H,t]
  | .seed => [h/c.H,h%c.H,k,v]
  | _ => []

/-- Actual caller data pointer plus its original, possibly zero or transposed,
strides. No payload is copied by this mathematical reader. -/
def callerKey (c : Shape) (caller : Region → Tensor) (r : Region) (p t j : Nat) :
    VeriTile.Bench.Repairs.SwigluTriton.Memory.Key :=
  ⟨(caller r).device,(caller r).base+4*(((coordinates c r p t j).zip (caller r).strides).map
    (fun pair => pair.1*pair.2)).sum⟩

def callerMemory {M α β : Type} (api : API M α β) (c : Shape)
    (caller : Region → Tensor) (m : M) : Memory α :=
  fun r p t j => api.read m (callerKey c caller r p t j)

def ReadRole (forward : Bool) (r : Region) : Prop :=
  if forward then r=.x ∨ r=.gate ∨ r=.seed else r=.incoming ∨ r=.gate

/-- Local element-copy/view semantics of the three original contiguous calls.
The seed uses its original view. These are ordinary input API laws and contain
no recurrence, reduction, saved-output formula, or completed kernel result. -/
structure Copies {M α β : Type} (api : API M α β) (c : Shape) (seed forward : Bool)
    (runtime : Runtime c seed) (caller : Region → Tensor) (before prepared : M) : Prop where
  element : ∀ r p t j,ReadRole forward r → Active c seed r p t j →
    api.read prepared (key c seed runtime ⟨r,p,t,j⟩)=
      api.read before (callerKey c caller r p t j)
  /-- Contiguous preparation only writes fresh storage, preserving every byte
  of each original caller backing, including gaps and storage-offset prefixes. -/
  frame : ∀ r,ReadRole forward r → present c seed r=Bool.true → ∀ owner address,
    owner=(caller r).device → (caller r).allocationBase ≤ address →
    address<(caller r).allocationBase+(caller r).allocationBytes →
      api.observe prepared owner address=api.observe before owner address

/-- Original input preparation preserves each caller byte. Fresh copy and
allocation semantics supply this local frame, independently of all kernels. -/
def Preserves {M α β : Type} (api : API M α β) (before prepared : M)
    (owner address : Nat) : Prop := api.observe prepared owner address=api.observe before owner address

theorem forward_congr (ops : Ops α) (c : Shape) (seed : Bool) (a b : Inputs α)
    (hx : ∀ p t j,p<c.programs → t<c.N → j<1024 → a.x p t j=b.x p t j)
    (hg : ∀ h t,h<c.heads → t<c.N → a.gate h t=b.gate h t)
    (hs : ∀ p j,p<c.programs → j<1024 → seed=Bool.true → 0<c.N → a.seed p j=b.seed p j)
    (p t j : Nat) (hp : p<c.programs) (ht : t<c.N) (hj : j<1024) :
    Reference.forward ops c a seed p t j=Reference.forward ops c b seed p t j := by
  induction t with
  | zero =>
    cases hseed : seed
    · rfl
    · simpa [Reference.forward] using hs p j hp hj hseed (by omega)
  | succ t ih =>
    simp only [Reference.forward]
    rw [ih (by omega),hg _ t (Safety.head_bound c p hp) (by omega),
      hx p t j hp (by omega) hj]

theorem reverse_congr (ops : Ops α) (c : Shape) (a b : Inputs α)
    (hi : ∀ p t j,p<c.programs → t<c.N → j<1024 → a.incoming p t j=b.incoming p t j)
    (hg : ∀ h t,h<c.heads → t<c.N → a.gate h t=b.gate h t)
    (p n j : Nat) (hp : p<c.programs) (hn : n ≤ c.N) (hj : j<1024) :
    Reference.reverse ops c a p n j=Reference.reverse ops c b p n j := by
  induction n with
  | zero => rfl
  | succ n ih =>
    simp only [Reference.reverse]
    rw [ih (by omega),hg _ _ (Safety.head_bound c p hp) (by omega),
      hi p _ j hp (by omega) hj]

theorem forward_reads {M α β : Type} (api : API M α β) (ops : Ops α) (c : Shape) (seed : Bool)
    (runtime : Runtime c seed) (caller : Region → Tensor) (before prepared : M)
    (copies : Copies api c seed Bool.true runtime caller before prepared) (p t j : Nat)
    (hp : p<c.programs) (ht : t<c.N) (hj : j<1024) :
    Reference.forward ops c (inputsOf (mirror api c seed runtime prepared)) seed p t j=
      Reference.forward ops c (inputsOf (callerMemory api c caller before)) seed p t j := by
  apply forward_congr ops c seed _ _ _ _ _ p t j hp ht hj
  · intro p t j hp ht hj
    exact copies.element .x p t j (Or.inl rfl) ⟨hp,ht,hj⟩
  · intro h t hh ht
    exact copies.element .gate h t 0 (Or.inr (Or.inl rfl)) ⟨hh,ht,rfl⟩
  · intro p j hp hj hs hn
    exact copies.element .seed p 0 j (Or.inr (Or.inr rfl)) ⟨hs,hn,hp,rfl,hj⟩

theorem reverse_reads {M α β : Type} (api : API M α β) (ops : Ops α) (c : Shape) (seed : Bool)
    (runtime : Runtime c seed) (caller : Region → Tensor) (before prepared : M)
    (copies : Copies api c seed Bool.false runtime caller before prepared) (p n j : Nat)
    (hp : p<c.programs) (hn : n ≤ c.N) (hj : j<1024) :
    Reference.reverse ops c (inputsOf (mirror api c seed runtime prepared)) p n j=
      Reference.reverse ops c (inputsOf (callerMemory api c caller before)) p n j := by
  apply reverse_congr ops c _ _ _ _ p n j hp hn hj
  · intro p t j hp ht hj
    exact copies.element .incoming p t j (Or.inl rfl) ⟨hp,ht,hj⟩
  · intro h t hh ht
    exact copies.element .gate h t 0 (Or.inr rfl) ⟨hh,ht,rfl⟩

#axiomsClean forward_reads
#axiomsClean reverse_reads
#specNonCircular Copies avoiding [Source.run,Source.backwardRun,Reference.forward,Reference.reverse]
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.Preparation
