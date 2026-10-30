import bench.tritonbench_g.rms_rbe_matmul.Proofs.KernelGrid

namespace VeriTile.Bench.Repairs.RmsRbeMatmul.KernelGrid
open VeriTile.Triton FP.Structural
open KernelSource (Operations Parameters ByteMemory)
noncomputable section
set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

abbrev Coordinate (B : Nat) (p : Parameters) := Fin B × Fin p.M × Fin p.N
def owner (p : Parameters) (x : Coordinate B p) : Program :=
  (x.1.val,x.2.1.val/16*tilesN p+x.2.2.val/64)
def lane (x : Coordinate B p) : TileIndex [16,64] :=
  (⟨x.2.1.val%16,by omega⟩,⟨x.2.2.val%64,by omega⟩,PUnit.unit)
def flat (p : Parameters) (x : Coordinate B p) :=
  x.1.val*(p.M*p.N)+x.2.1.val*p.N+x.2.2.val

theorem owner_indices (p : Parameters) (x : Coordinate B p) :
    (owner p x).2/tilesN p=x.2.1.val/16 ∧ (owner p x).2%tilesN p=x.2.2.val/64 := by
  have hn := x.2.2.isLt
  have positive : 0<tilesN p := by unfold tilesN; omega
  have bound : x.2.2.val/64<tilesN p := by unfold tilesN; omega
  simp only [owner,Prod.snd]
  constructor
  · rw [Nat.mul_comm (x.2.1.val/16) (tilesN p),Nat.mul_add_div positive,
      Nat.div_eq_of_lt bound,Nat.add_zero]
  · simp [Nat.add_mod,Nat.mod_eq_of_lt bound]

theorem owner_mem (p : Parameters) (x : Coordinate B p) : owner p x∈programs B p := by
  have hm := x.2.1.isLt
  have hn := x.2.2.isLt
  have row : x.2.1.val/16<tilesM p := by unfold tilesM; omega
  have col : x.2.2.val/64<tilesN p := by unfold tilesN; omega
  have bound : x.2.1.val/16*tilesN p+x.2.2.val/64<tilesM p*tilesN p := by
    have le := Nat.succ_le_of_lt row
    nlinarith
  simpa [programs,owner] using And.intro x.1.isLt bound

theorem owner_coordinate (p : Parameters) (x : Coordinate B p) :
    row p (owner p x) (lane x).1=x.2.1.val ∧
    col p (owner p x) (lane x).2.1=x.2.2.val := by
  obtain ⟨hm,hn⟩ := owner_indices p x
  simp only [row,col,hm,hn,lane]
  constructor <;> omega
theorem owner_active (p : Parameters) (x : Coordinate B p) : active p (owner p x) (lane x) := by
  obtain ⟨hm,hn⟩ := owner_coordinate p x
  exact ⟨by simpa [hm] using x.2.1.isLt,by simpa [hn] using x.2.2.isLt⟩
theorem owner_offset (p : Parameters) (x : Coordinate B p) : offset p (owner p x) (lane x)=flat p x := by
  unfold offset
  rw [(owner_coordinate p x).1,(owner_coordinate p x).2]
  rfl

theorem covers (p : Parameters) (q : Program) (hq : q∈programs B p)
    (i : TileIndex [16,64]) (hi : active p q i) :
    ∃ x : Coordinate B p, q=owner p x ∧ i=lane x := by
  have hb : q.1<B := by simpa [programs] using (List.mem_product.mp hq).1
  let x : Coordinate B p := (⟨q.1,hb⟩,⟨row p q i.1,hi.1⟩,⟨col p q i.2.1,hi.2⟩)
  have he : offset p q i=offset p (owner p x) (lane x) := by
    rw [owner_offset]
    rfl
  exact ⟨x,offset_unique p q (owner p x) i (lane x) hi (owner_active p x) he⟩

variable {α : Type} [Inhabited α]
theorem logical_complete (a : Operations α) (fp8 : Bool) (B : Nat) (p : Parameters)
    (s : State α) (bytes : ByteMemory s) :
    ∃ t, launch a fp8 B p s=some t ∧
      (∀ x : Coordinate B p, t.mem "OUT" (flat p x)=result a fp8 p s (owner p x) (lane x)) ∧
      (∀ (r : RegionName) (off : Nat), (r≠"OUT" ∨ ∀ x : Coordinate B p,off≠flat p x) → t.mem r off=s.mem r off) := by
  obtain ⟨t,run,e⟩ := complete a fp8 B p s bytes
  refine ⟨t,run,?_,?_⟩
  · intro x
    simpa only [owner_offset] using e.output (owner p x) (owner_mem p x) (lane x) (owner_active p x)
  · intro r off outside
    apply e.frame r off
    rcases outside with hr|ho
    · exact .inl hr
    · right
      intro q hq i hi
      obtain ⟨x,rfl,rfl⟩ := covers p q hq i hi
      simpa only [owner_offset] using ho x

#axiomsClean covers
#axiomsClean logical_complete
end
end VeriTile.Bench.Repairs.RmsRbeMatmul.KernelGrid
