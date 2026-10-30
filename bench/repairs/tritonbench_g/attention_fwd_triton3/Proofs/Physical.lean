import bench.repairs.tritonbench_g.attention_fwd_triton3.Proofs.NativeSchedule
import bench.repairs.tritonbench_g.attention_fwd_triton3.Proofs.RawStores

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3.Physical
open VeriTile.Triton Host Ownership Raw RawStores
noncomputable section

abbrev Bytes := Nat → Byte

/-- Four little-endian storage bytes; arbitrary NaN payloads and signed zeros
are retained. All six operands are read from this single initial memory. -/
def word (before : Bytes) (base : Nat) : Word := BitVec.ofNat 32
  ((before base).toNat+256*(before (base+1)).toNat+
    65536*(before (base+2)).toNat+16777216*(before (base+3)).toNat)

def snapshot (A : FlatAlloc) (before : Bytes) : Memory :=
  fun rg off=>word before (4*A.addr rg off)

def address (c : Call) (A : FlatAlloc) (idx : GridIndex c.grid)
    (i : Lane c.BM c.D) (byte : Fin 4) : Nat :=
  4*(Flat.address c A idx i).2+byte.val

theorem unique (c : Call) (h : Valid c) (A : FlatAlloc) (a : Allocated c A.extent)
    (hd : A.Disjoint) (hc : ∀ rg,rg∉A.regions → A.extent rg=0)
    (idx jdx : GridIndex c.grid) (i j : Lane c.BM c.D) (b e : Fin 4)
    (he : address c A idx i b=address c A jdx j e) : idx=jdx ∧ i=j ∧ b=e := by
  have hb:=b.isLt
  have he':=e.isLt
  unfold address at he
  have hoff : (Flat.address c A idx i).2=(Flat.address c A jdx j).2 := by omega
  have hu:=Flat.unique c h A a hd hc idx jdx i j (Prod.ext rfl hoff)
  exact ⟨hu.1,hu.2,Fin.ext (by omega)⟩

/-- Execute every active ordinary byte store. The unique-writer theorem above
is derived from tensor extents, stride hierarchy and allocation disjointness. -/
def execute (c : Call) (A : FlatAlloc) (ops : Ops) (schedule : Schedule c)
    (before : Bytes) (scale : Word) : Bytes := by
  classical
  exact fun addr=>if hw : ∃ idx i b,active c i ∧ address c A idx i b=addr then
    let idx:=Classical.choose hw
    let i:=Classical.choose (Classical.choose_spec hw)
    let b:=Classical.choose (Classical.choose_spec (Classical.choose_spec hw))
    byte (output c ops schedule (snapshot A before) scale idx i) b
  else before addr

theorem readback (c : Call) (h : Valid c) (A : FlatAlloc) (a : Allocated c A.extent)
    (hd : A.Disjoint) (hc : ∀ rg,rg∉A.regions → A.extent rg=0)
    (ops : Ops) (schedule : Schedule c) (before : Bytes) (scale : Word)
    (idx : GridIndex c.grid) (i : Lane c.BM c.D) (b : Fin 4) (hi : active c i) :
    execute c A ops schedule before scale (address c A idx i b)=
      byte (output c ops schedule (snapshot A before) scale idx i) b := by
  have hex : ∃ jdx j e,active c j ∧ address c A jdx j e=address c A idx i b := ⟨idx,i,b,hi,rfl⟩
  rw [execute,dif_pos hex]
  have hh:=Classical.choose_spec (Classical.choose_spec (Classical.choose_spec hex))
  obtain ⟨hidx,hi',hb⟩:=unique c h A a hd hc _ idx _ i _ b hh.2
  exact congrArg₂ byte (congrArg₂ (output c ops schedule (snapshot A before) scale) hidx hi') hb

theorem frame (c : Call) (A : FlatAlloc) (ops : Ops) (schedule : Schedule c)
    (before : Bytes) (scale : Word) (addr : Nat)
    (hn : ¬∃ idx i b,active c i ∧ address c A idx i b=addr) :
    execute c A ops schedule before scale addr=before addr := by rw [execute,dif_neg hn]

theorem region_frame (c : Call) (h : Valid c) (A : FlatAlloc) (a : Allocated c A.extent)
    (hd : A.Disjoint) (hc : ∀ rg,rg∉A.regions → A.extent rg=0)
    (ops : Ops) (schedule : Schedule c) (before : Bytes) (scale : Word)
    (rg : RegionName) (off : Nat) (b : Fin 4) (hb : off<A.extent rg)
    (hr : rg≠"O" ∧ rg≠"M" ∧ (c.finish≠0 ∨ rg≠"L")) :
    execute c A ops schedule before scale (4*A.addr rg off+b.val)=before (4*A.addr rg off+b.val) := by
  apply frame
  rintro ⟨idx,i,e,hi,he⟩
  have heb:=e.isLt
  have hbb:=b.isLt
  unfold address at he
  have hoff : (Flat.address c A idx i).2=A.addr rg off := by omega
  have hbi:=Flat.output_bound c h a idx i
  have hx:=A.decode_addr hd (Flat.member A hc hbi) hbi
  have hy:=A.decode_addr hd (Flat.member A hc hb) hb
  change A.decode (Flat.address c A idx i).2=some (Ownership.address c idx i) at hx
  rw [hoff,hy] at hx
  have heq:=(Option.some.inj hx).symm
  cases i <;> simp [Ownership.address,active] at heq hi
  · exact hr.2.1 heq.1.symm
  · rcases hr.2.2 with hf|hn
    · exact hf hi
    · exact hn heq.1.symm
  · exact hr.1 heq.1.symm

theorem end_L (c : Call) (h : Valid c) (A : FlatAlloc) (a : Allocated c A.extent)
    (hd : A.Disjoint) (hc : ∀ rg,rg∉A.regions → A.extent rg=0)
    (ops : Ops) (schedule : Schedule c) (before : Bytes) (scale : Word)
    (hf : c.finish≠0) (off : Nat) (b : Fin 4) (hb : off<A.extent "L") :
    execute c A ops schedule before scale (4*A.addr "L" off+b.val)=before (4*A.addr "L" off+b.val) :=
  region_frame c h A a hd hc ops schedule before scale "L" off b hb
    ⟨by decide,by decide,Or.inl hf⟩

structure View (c : Call) where
  shape : List Nat
  output : Nat → Nat → Nat → Nat → Fin 4 → Byte
  center : Nat → Fin 4 → Byte
  denominator : Nat → Fin 4 → Byte
  sameM : Bool
  sameL : Bool
  guaranteesSameO : Bool

def returned (c : Call) (A : FlatAlloc) (after : Bytes) : View c where
  shape := [c.Z,c.H,if c.finish=0 then c.rounded else c.NQ,c.D]
  output := fun z h r d b=>after (4*A.addr "O" (c.o.at z h r d)+b.val)
  center := fun r b=>after (4*A.addr "M" r+b.val)
  denominator := fun r b=>after (4*A.addr "L" r+b.val)
  sameM := Bool.true
  sameL := Bool.true
  guaranteesSameO := decide (c.finish=0)

/-- The returned view reads the completed physical memory. On END, the
slice/contiguous/FP32 conversion preserves these bytes at every logical row. -/
theorem public_output (c : Call) (h : Valid c) (A : FlatAlloc) (a : Allocated c A.extent)
    (hd : A.Disjoint) (hc : ∀ rg,rg∉A.regions → A.extent rg=0)
    (ops : Ops) (schedule : Schedule c) (before : Bytes) (scale : Word)
    (z head row dim : Nat) (hz : z<c.Z) (hh : head<c.H) (hr : row<c.NQ) (hdi : dim<c.D)
    (b : Fin 4) :
    ∃ idx i,active c i ∧ Ownership.address c idx i=(("O" : RegionName),c.o.at z head row dim) ∧
      (returned c A (execute c A ops schedule before scale)).output z head row dim b=
      byte (output c ops schedule (snapshot A before) scale idx i) b := by
  obtain ⟨idx,i,hi,ha⟩:=Return.output_covered c h z head row dim hz hh hr hdi
  refine ⟨idx,i,hi,ha,?_⟩
  have he:=readback c h A a hd hc ops schedule before scale idx i b hi
  simpa only [address,Flat.address,ha,returned] using he

theorem whole_equiv (c : Call) (A : FlatAlloc) (actual reference : Ops)
    (laws : LocalLaws actual reference) (schedule : Schedule c) (before : Bytes) (scale : Word) :
    execute c A actual schedule before scale=execute c A reference schedule before scale ∧
    returned c A (execute c A actual schedule before scale)=
      returned c A (execute c A reference schedule before scale) := by
  rw [operations_eq actual reference laws]
  exact ⟨rfl,rfl⟩

/-- Continuation consumes the exact first final bytes, including arbitrary
M/L/O carry. No condition reinterprets END results as an unnormalized state. -/
def twoCalls (c d : Call) (A : FlatAlloc) (ops : Ops) (sc : Schedule c) (sd : Schedule d)
    (before : Bytes) (scaleC scaleD : Word) : Bytes :=
  execute d A ops sd (execute c A ops sc before scaleC) scaleD

theorem two_calls_equiv (c d : Call) (A : FlatAlloc) (actual reference : Ops)
    (laws : LocalLaws actual reference) (sc : Schedule c) (sd : Schedule d)
    (before : Bytes) (scaleC scaleD : Word) :
    twoCalls c d A actual sc sd before scaleC scaleD=twoCalls c d A reference sc sd before scaleC scaleD := by
  rw [operations_eq actual reference laws]

#print axioms readback
#print axioms public_output
#print axioms whole_equiv
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3.Physical
