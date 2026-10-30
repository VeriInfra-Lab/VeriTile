import bench.repairs.tritonbench_g.rms_rbe_matmul.Proofs.KernelExecution
import bench.repairs.tritonbench_g.rms_rbe_matmul.Proofs.KernelSourceGraph

namespace VeriTile.Bench.Repairs.RmsRbeMatmul.KernelGrid
open VeriTile.Triton FP.Structural
open KernelSource (Operations Parameters ByteMemory)
noncomputable section
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]
abbrev Program := Nat × Nat
def atProgram (s : State α) (q : Program) : State α :=
  {s with pids:=fun dim => if dim=0 then q.1 else if dim=1 then q.2 else 0}
def tilesN (p : Parameters) := (p.N+63)/64
def tilesM (p : Parameters) := (p.M+15)/16
def tile (p : Parameters) (q : Program) : KernelGraph.Tile := ⟨q.1,q.2/tilesN p,q.2%tilesN p⟩
def row (p : Parameters) (q : Program) (i : Fin 16) := q.2/tilesN p*16+i.val
def col (p : Parameters) (q : Program) (j : Fin 64) := q.2%tilesN p*64+j.val
def active (p : Parameters) (q : Program) (i : TileIndex [16,64]) : Prop :=
  row p q i.1<p.M ∧ col p q i.2.1<p.N
def offset (p : Parameters) (q : Program) (i : TileIndex [16,64]) :=
  q.1*(p.M*p.N)+row p q i.1*p.N+col p q i.2.1
def result (a : Operations α) (fp8 : Bool) (p : Parameters) (s : State α)
    (q : Program) (i : TileIndex [16,64]) : Cell α :=
  .mk (KernelSource.outputDType fp8).toTileDType
    (ofFloat (KernelSource.outputDType fp8)
      (KernelGraph.output a fp8 p (KernelSource.inputs p s) (tile p q) i))
def Outside (p : Parameters) (qs : List Program) (region : RegionName) (off : Nat) : Prop :=
  region≠"OUT" ∨ ∀ q, q∈qs → ∀ i, active p q i → off≠offset p q i
structure Effect (a : Operations α) (fp8 : Bool) (p : Parameters)
    (qs : List Program) (before after : State α) : Prop where
  output : ∀ q, q∈qs → ∀ i, active p q i → after.mem "OUT" (offset p q i)=result a fp8 p before q i
  frame : ∀ r off, Outside p qs r off → after.mem r off=before.mem r off

def runPrograms (a : Operations α) (fp8 : Bool) (p : Parameters) : List Program → State α → Option (State α)
  | [],s => some s
  | q::qs,s => do
    let next ← exec (KernelSource.engine a) (KernelSource.kernel fp8 p) (atProgram s q)
    runPrograms a fp8 p qs next
def programs (B : Nat) (p : Parameters) : List Program :=
  (List.range B).product (List.range (tilesM p*tilesN p))
def launch (a : Operations α) (fp8 : Bool) (B : Nat) (p : Parameters) (s : State α) :=
  runPrograms a fp8 p (programs B p) s

theorem one_complete_reads (a : Operations α) (fp8 : Bool) (p : Parameters) (s : State α)
    (q : Program) (bytes : ∀ cur, cur<KernelSource.steps p →
      KernelSource.BlockBytes fp8 p (atProgram s q) cur) :
    ∃ t, exec (KernelSource.engine a) (KernelSource.kernel fp8 p) (atProgram s q)=some t ∧
      Effect a fp8 p [q] s t := by
  obtain ⟨t,ran,out,frame⟩ := KernelSource.source_complete_reads a fp8 p (atProgram s q) bytes
  refine ⟨t,ran,?_,?_⟩
  · intro r hr i hi
    have same : r=q := by simpa using hr
    subst r
    have hactive : KernelSource.active p (atProgram s q) i=Bool.true := by
      simpa [KernelSource.active,KernelSource.row,KernelSource.col,KernelSource.pidM,
        KernelSource.pidN,atProgram,active,row,col,tilesN] using hi
    have h := out i hactive
    rw [KernelSource.result,KernelSource.output_graph] at h
    simpa [KernelSource.outputOffset,KernelSource.row,KernelSource.col,KernelSource.pidM,
      KernelSource.pidN,atProgram,offset,row,col,tilesN,result,KernelSource.inputs,KernelSource.tile,tile] using h
  · intro r off ho
    apply frame r off
    rcases ho with hr|ho
    · exact .inl hr
    · right
      intro i hi
      simpa [KernelSource.outputOffset,KernelSource.row,KernelSource.col,KernelSource.pidM,
        KernelSource.pidN,atProgram,offset,row,col,tilesN] using ho q (by simp) i
          (by simpa [KernelSource.active,KernelSource.row,KernelSource.col,KernelSource.pidM,
            KernelSource.pidN,atProgram,active,row,col,tilesN] using hi)

theorem one_complete (a : Operations α) (fp8 : Bool) (p : Parameters) (s : State α)
    (bytes : ByteMemory s) (q : Program) :
    ∃ t, exec (KernelSource.engine a) (KernelSource.kernel fp8 p) (atProgram s q)=some t ∧
      Effect a fp8 p [q] s t :=
  one_complete_reads a fp8 p s q
    (fun cur _ => KernelSource.blockBytes_of_byteMemory fp8 p (atProgram s q) cur bytes)

theorem radix_unique {base a b c d : Nat} (hc : c<base) (hd : d<base)
    (he : a*base+c=b*base+d) : a=b ∧ c=d := by
  have same : a=b := by
    by_contra hn
    rcases lt_or_gt_of_ne hn with h|h
    · have := Nat.succ_le_of_lt h; nlinarith
    · have := Nat.succ_le_of_lt h; nlinarith
  exact ⟨same,by rw [same] at he; omega⟩

theorem offset_unique (p : Parameters) (q r : Program) (i j : TileIndex [16,64])
    (hi : active p q i) (hj : active p r j) (he : offset p q i=offset p r j) : q=r ∧ i=j := by
  have hre : (q.1*p.M+row p q i.1)*p.N+col p q i.2.1=
      (r.1*p.M+row p r j.1)*p.N+col p r j.2.1 := by
    simpa [offset,Nat.add_mul,Nat.mul_assoc] using he
  obtain ⟨hbr,hcol⟩ := radix_unique hi.2 hj.2 hre
  obtain ⟨hbatch,hrow⟩ := radix_unique hi.1 hj.1 hbr
  obtain ⟨hm,hii⟩ := radix_unique i.1.isLt j.1.isLt hrow
  obtain ⟨hn,hjj⟩ := radix_unique i.2.1.isLt j.2.1.isLt hcol
  have hprog : q.2=r.2 := by
    have hq := Nat.mod_add_div q.2 (tilesN p)
    have hr := Nat.mod_add_div r.2 (tilesN p)
    rw [hm,hn] at hq
    exact hq.symm.trans hr
  refine ⟨Prod.ext hbatch hprog,?_⟩
  rcases i with ⟨i,k,⟨⟩⟩
  rcases j with ⟨j,l,⟨⟩⟩
  simp only [Fin.ext_iff] at hii hjj
  exact Prod.ext (Fin.ext hii) (Prod.ext (Fin.ext hjj) rfl)

theorem Effect.inputs {a : Operations α} {fp8 : Bool} {p : Parameters}
    {qs : List Program} {s t : State α}
    (e : Effect a fp8 p qs s t) : KernelSource.inputs p t=KernelSource.inputs p s := by
  unfold KernelSource.inputs
  congr 1
  all_goals funext
  all_goals first
    | rw [e.frame "X" _ (.inl (by decide))]
    | rw [e.frame "W" _ (.inl (by decide))]
    | rw [e.frame "RMS" _ (.inl (by decide))]

theorem Effect.bytes {a : Operations α} {fp8 : Bool} {p : Parameters}
    {qs : List Program} {s t : State α}
    (e : Effect a fp8 p qs s t) (bytes : ByteMemory s) : ByteMemory t := by
  intro r hr off
  rw [e.frame r off (.inl (by rcases hr with h|h <;> simp [h]))]
  exact bytes r hr off

/-- Input memory does not change between scheduled programs, including the
signed-byte views used only by the actual FP8 loads. -/
theorem Effect.blockBytes {a : Operations α} {fp8 : Bool} {p : Parameters}
    {qs : List Program} {s t : State α} (e : Effect a fp8 p qs s t)
    (q : Program) (cur : Nat)
    (bytes : KernelSource.BlockBytes fp8 p (atProgram s q) cur) :
    KernelSource.BlockBytes fp8 p (atProgram t q) cur := by
  intro hf
  have h := bytes hf
  constructor
  · intro i hi
    change Float8E5Bits.isInt8 ((t.mem "RMS" _).read .int)=Bool.true
    rw [e.frame "RMS" _ (.inl (by decide))]
    exact h.1 i hi
  · intro i hi
    change Float8E5Bits.isInt8 ((t.mem "W" (KernelSource.wOffset p (atProgram s q) cur i)).read .int)=Bool.true
    rw [e.frame "W" _ (.inl (by decide))]
    exact h.2 i hi

theorem programs_complete_reads (a : Operations α) (fp8 : Bool) (p : Parameters)
    (qs : List Program) (s : State α)
    (bytes : ∀ q, q∈qs → ∀ cur, cur<KernelSource.steps p →
      KernelSource.BlockBytes fp8 p (atProgram s q) cur) :
    ∃ t, runPrograms a fp8 p qs s=some t ∧ Effect a fp8 p qs s t := by
  induction qs generalizing s with
  | nil => exact ⟨s,rfl,⟨by simp,fun _ _ _ => rfl⟩⟩
  | cons q qs ih =>
    obtain ⟨u,hu,eu⟩ := one_complete_reads a fp8 p s q (bytes q (by simp))
    obtain ⟨t,ht,et⟩ := ih u (fun r hr cur hcur => eu.blockBytes r cur
      (bytes r (List.mem_cons_of_mem q hr) cur hcur))
    refine ⟨t,?_,?_,?_⟩
    · simp [runPrograms,hu,ht]
    · intro r hr i hi
      by_cases tail : r∈qs
      · simpa only [result,eu.inputs] using et.output r tail i hi
      · have same : r=q := by simpa [tail] using hr
        subst r
        rw [et.frame "OUT" _ (.inr ?_)]
        · exact eu.output q (by simp) i hi
        · intro r hr j hj equal
          exact tail ((offset_unique p q r i j hi hj equal).1.symm ▸ hr)
    · intro r off ho
      have hoq : Outside p [q] r off := by
        rcases ho with hr|ho
        · exact .inl hr
        · right
          intro r hr i hi
          have same : r=q := by simpa using hr
          subst r
          exact ho q (by simp) i hi
      have hot : Outside p qs r off := by
        rcases ho with hr|ho
        · exact .inl hr
        · exact .inr (fun r hr i hi => ho r (List.mem_cons_of_mem _ hr) i hi)
      exact (et.frame r off hot).trans (eu.frame r off hoq)

theorem programs_complete (a : Operations α) (fp8 : Bool) (p : Parameters)
    (qs : List Program) (s : State α) (bytes : ByteMemory s) :
    ∃ t, runPrograms a fp8 p qs s=some t ∧ Effect a fp8 p qs s t :=
  programs_complete_reads a fp8 p qs s
    (fun q _ cur _ => KernelSource.blockBytes_of_byteMemory fp8 p (atProgram s q) cur bytes)

theorem complete (a : Operations α) (fp8 : Bool) (B : Nat) (p : Parameters)
    (s : State α) (bytes : ByteMemory s) :
    ∃ t, launch a fp8 B p s=some t ∧ Effect a fp8 p (programs B p) s t :=
  programs_complete a fp8 p (programs B p) s bytes

theorem complete_reads (a : Operations α) (fp8 : Bool) (B : Nat) (p : Parameters)
    (s : State α) (bytes : ∀ q, q∈programs B p → ∀ cur, cur<KernelSource.steps p →
      KernelSource.BlockBytes fp8 p (atProgram s q) cur) :
    ∃ t, launch a fp8 B p s=some t ∧ Effect a fp8 p (programs B p) s t :=
  programs_complete_reads a fp8 p (programs B p) s bytes

#axiomsClean one_complete_reads
#axiomsClean programs_complete_reads
#axiomsClean complete_reads

#axiomsClean offset_unique
#axiomsClean complete
end
end VeriTile.Bench.Repairs.RmsRbeMatmul.KernelGrid
