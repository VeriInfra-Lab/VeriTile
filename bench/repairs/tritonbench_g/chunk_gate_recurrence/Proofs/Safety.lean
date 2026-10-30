import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.Machine
import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.Whole
namespace VeriTile.Bench.Repairs.ChunkGateRecurrence
noncomputable section
namespace Safety

theorem head_bound (c : Shape) (p : Nat) (hp : p<c.programs) : p/c.tiles<c.heads :=
  (Nat.div_lt_iff_lt_mul c.tiles_pos).mpr hp

theorem mixed_bound (a n m b : Nat) (ha : a<m) (hb : b<n) : a*n+b<m*n := by
  have h := Nat.mul_le_mul_right n (Nat.succ_le_of_lt ha)
  simp only [Nat.succ_mul] at h
  omega

theorem digit_decode (n a b : Nat) (hn : n>0) (hb : b<n) :
    (a*n+b)/n=a ∧ (a*n+b)%n=b := by
  have h := digit_injective n a b ((a*n+b)/n) ((a*n+b)%n) hb (Nat.mod_lt _ hn)
    (by simpa [Nat.mul_comm] using (Nat.div_add_mod (a*n+b) n).symm)
  exact ⟨h.1.symm,h.2.symm⟩

/-- The real three-dimensional launch and normalized program index are a
bijection. All 64x16 lanes are retained, including both tile axes. -/
theorem program_grid (c : Shape) (h tk tv : Nat) (hh : h<c.heads) (hk : tk<c.KT) (hv : tv<c.VT) :
    let p := h*c.tiles+(tk*c.VT+tv)
    p<c.programs ∧ p/c.tiles=h ∧ (p%c.tiles)/c.VT=tk ∧ (p%c.tiles)%c.VT=tv := by
  have hq : tk*c.VT+tv<c.tiles := mixed_bound tk c.VT c.KT tv hk hv
  have hd := digit_decode c.tiles h (tk*c.VT+tv) c.tiles_pos hq
  have ht := digit_decode c.VT tk tv c.vpos hv
  exact ⟨c.program_bound h _ hh hq,hd.1,by rw [hd.2]; exact ht.1,by rw [hd.2]; exact ht.2⟩

theorem lane_grid (e j : Nat) (he : e<64) (hj : j<16) :
    e*16+j<1024 ∧ (e*16+j)/16=e ∧ (e*16+j)%16=j := by omega

/-- Exact active access families of the complete Python/typed source and host
postprocessing. Negative reverse pointers are formed but have no constructor
when their loop is empty; source integer formation is covered by Machine. -/
inductive Access (c : Shape) (seed : Bool) : Region → Nat → Nat → Nat → Prop where
  | fSeed (p j : Nat) (hp : p<c.programs) (hj : j<1024) (hn : c.N>0) (hs : seed=Bool.true) : Access c seed .seed p 0 j
  | fInitial (p j : Nat) (hp : p<c.programs) (hj : j<1024) (hn : c.N>0) : Access c seed .output p 0 j
  | fInput (p t j : Nat) (hp : p<c.programs) (ht : t+1<c.N) (hj : j<1024) : Access c seed .x p t j
  | fGate (p t : Nat) (hp : p<c.programs) (ht : t+1<c.N) : Access c seed .gate (p/c.tiles) t 0
  | fOutput (p t j : Nat) (hp : p<c.programs) (ht : t+1<c.N) (hj : j<1024) : Access c seed .output p (t+1) j
  | bSaved (p t j : Nat) (hp : p<c.programs) (ht : t+1<c.N) (hj : j<1024) : Access c seed .output p t j
  | bIncoming (p t j : Nat) (hp : p<c.programs) (ht : t<c.N) (hj : j<1024) : Access c seed .incoming p t j
  | bGate (p t : Nat) (hp : p<c.programs) (ht : t<c.N) : Access c seed .gate (p/c.tiles) t 0
  | bInput (p t j : Nat) (hp : p<c.programs) (ht : t+1<c.N) (hj : j<1024) : Access c seed .dx p t j
  | bTileGate (p t : Nat) (hp : p<c.programs) (ht : t+1<c.N) : Access c seed .tileGate p t 0
  | bSeed (p j : Nat) (hp : p<c.programs) (hj : j<1024) (hn : c.N>0) : Access c seed .dseed p 0 j
  | tailInput (p j : Nat) (hp : p<c.programs) (hj : j<1024) (hn : c.N>0) : Access c seed .dx p (c.N-1) j
  | tailGate (p : Nat) (hp : p<c.programs) (hn : c.N>0) : Access c seed .tileGate p (c.N-1) 0
  | sumRead (h t q : Nat) (hh : h<c.heads) (ht : t<c.N) (hq : q<c.tiles) : Access c seed .tileGate (h*c.tiles+q) t 0
  | sumWrite (h t : Nat) (hh : h<c.heads) (ht : t<c.N) : Access c seed .dgate h t 0
  | emptySeed (p j : Nat) (hp : p<c.programs) (hj : j<1024) (hn : c.N=0) (hs : seed=Bool.true) : Access c seed .dseed p 0 j

theorem active (c : Shape) (seed : Bool) (r : Region) (p t j : Nat) (access : Access c seed r p t j) :
    Machine.Active c seed r p t j := by
  cases access with
  | fSeed p j hp hj hn hs => simp [Machine.Active,hp,hj,hn,hs]
  | fInitial p j hp hj hn => exact ⟨hp,hn,hj⟩
  | fInput p t j hp ht hj => exact ⟨hp,by omega,hj⟩
  | fGate p t hp ht => exact ⟨head_bound c p hp,by omega,rfl⟩
  | fOutput p t j hp ht hj => exact ⟨hp,ht,hj⟩
  | bSaved p t j hp ht hj => exact ⟨hp,by omega,hj⟩
  | bIncoming p t j hp ht hj => exact ⟨hp,ht,hj⟩
  | bGate p t hp ht => exact ⟨head_bound c p hp,ht,rfl⟩
  | bInput p t j hp ht hj => exact ⟨hp,by omega,hj⟩
  | bTileGate p t hp ht => exact ⟨hp,by omega,rfl⟩
  | bSeed p j hp hj hn => exact ⟨Or.inl hn,hp,rfl,hj⟩
  | tailInput p j hp hj hn => exact ⟨hp,by omega,hj⟩
  | tailGate p hp hn => exact ⟨hp,by omega,rfl⟩
  | sumRead h t q hh ht hq => exact ⟨c.program_bound h q hh hq,ht,rfl⟩
  | sumWrite h t hh ht => exact ⟨hh,ht,rfl⟩
  | emptySeed p j hp hj hn hs => exact ⟨Or.inr hs,hp,rfl,hj⟩

theorem all_accesses_safe (c : Shape) (seed : Bool) (abi : Machine.ABI) (hd : Machine.Domain c seed abi)
    (r : Region) (p t j : Nat) (access : Access c seed r p t j) :
    (abi.allocation r).Contains (Machine.offset c abi r p t j) ∧
    0≤(abi.allocation r).dataPointer+4*Machine.offset c abi r p t j ∧
    (abi.allocation r).dataPointer+4*Machine.offset c abi r p t j+4≤2^64 :=
  ⟨hd.words r p t j (active c seed r p t j access),hd.pointers r p t j (active c seed r p t j access)⟩

#axiomsClean program_grid
#axiomsClean lane_grid
#axiomsClean active
#axiomsClean all_accesses_safe
#guard_msgs (drop info) in
#auditModuleAxioms
end Safety
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence
