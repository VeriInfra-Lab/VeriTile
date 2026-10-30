import bench.repairs.tritonbench_g.fifth_order_sph_harmonics.Proofs.Readback

namespace VeriTile.Bench.Repairs.FifthOrderSphHarmonics.Grid
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 7000000
set_option maxRecDepth 48000
variable {α : Type} [Inhabited α]

inductive Direction where | forward | backward deriving DecidableEq
def channels : Direction → Nat | .forward=>11 | .backward=>3
def output : Direction → RegionName | .forward=>outputRegion | .backward=>gradientRegion
def source : Direction → Parameters → ComputeKernel
  | .forward=>forwardKernel
  | .backward=>backwardKernel

def atProgram (s : State α) (pid : Nat) : State α :=
  {s with pids := fun dimension => if dimension=0 then pid else s.pids dimension}

def stride (p : Parameters) : Direction → Nat | .forward=>p.outputWidth | .backward=>3
def first (p : Parameters) : Direction → Nat | .forward=>p.columnOffset | .backward=>0
def offset (p : Parameters) (d : Direction) (r c : Nat) := r*stride p d+first p d+c

theorem row_mask (rows width col r c : Nat) (fit : col+11≤width) (hc : c<11) :
    r*width+col+c<width*rows ↔ r<rows := by
  constructor
  · intro active
    by_contra inactive
    have lower := Nat.mul_le_mul_right width (Nat.le_of_not_gt inactive)
    have positive : 0<width := by omega
    rw [Nat.mul_comm rows width] at lower
    omega
  · intro active
    have upper := Nat.mul_le_mul_right width (Nat.succ_le_of_lt active)
    rw [Nat.succ_mul,Nat.mul_comm rows width] at upper
    omega

def value (M : Algebra α) (d : Direction) (p : Parameters) (s : State α) (r : Nat) : Fin (channels d) → α :=
  match d with
  | .forward=>Reference.forwardValues M
      (M.fp32Load ((s.mem coordinateRegion (r*3)).read .real))
      (M.fp32Load ((s.mem coordinateRegion (r*3+1)).read .real))
      (M.fp32Load ((s.mem coordinateRegion (r*3+2)).read .real))
  | .backward=>Readback.backwardExpected M p s r

theorem address_unique (width r t c e : Nat) (hc : c<width) (he : e<width)
    (eq : r*width+c=t*width+e) : r=t ∧ c=e := by
  have positive : 0<width := by omega
  have quot:=congrArg (fun x=>x/width) eq
  have rows : r=t := by
    simpa [Nat.mul_comm r width,Nat.mul_comm t width,Nat.mul_add_div positive,
      Nat.div_eq_of_lt hc,Nat.div_eq_of_lt he] using quot
  exact ⟨rows,by rw [rows] at eq;omega⟩

theorem offset_unique (p : Parameters) (d : Direction)
    (fit : p.columnOffset+11≤p.outputWidth) (r t : Nat) (c e : Fin (channels d))
    (eq : offset p d r c.val=offset p d t e.val) : r=t ∧ c.val=e.val := by
  have hc := c.isLt
  have he := e.isLt
  cases d <;> simp only [channels] at hc he <;>
    simp only [offset,stride,first] at eq
  · apply address_unique p.outputWidth r t c.val e.val (by omega) (by omega)
    omega
  · exact address_unique 3 r t c.val e.val hc he (by simpa using eq)

theorem forward_tile (M : Algebra α) (p : Parameters) (s : State α)
    (fit : p.columnOffset+11≤p.outputWidth) :
    ∃ t,FP.Structural.exec M (forwardKernel p) s=some t ∧
      (∀ (c : Fin 11) q,Readback.row p s q<p.rows → t.mem outputRegion (Readback.row p s q*p.outputWidth+p.columnOffset+c.val)=
        .mk .real (value M .forward p s (Readback.row p s q) c)) ∧
      (∀ region index,(region≠outputRegion ∨ ∀ (c : Fin 11) q,
        Readback.row p s q<p.rows → index≠Readback.row p s q*p.outputWidth+p.columnOffset+c.val) →
        t.mem region index=s.mem region index) := by
  let original:=forwardParameters p
  have coord : ∀ q,VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.TypedFP.coordOffset original s q=
      Readback.row p s q*3 := by
    intro q
    simp [original,forwardParameters,
      VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.TypedFP.coordOffset,Readback.row]
    <;> ring
  have offset : ∀ c q,VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.TypedFP.outputOffset original s c q=
      Readback.row p s q*p.outputWidth+p.columnOffset+c.val := by
    intro c q
    simp [original,forwardParameters,
      VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.TypedFP.outputOffset,Readback.row]
    <;> ring
  have cover : VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.TypedFP.InputCover original s := by
    intro q h
    rw [offset] at h
    rw [coord]
    change Readback.row p s q*p.outputWidth+p.columnOffset+0<p.outputWidth*p.rows at h
    change Readback.row p s q*3+2<3*p.rows
    have active := (row_mask p.rows p.outputWidth p.columnOffset (Readback.row p s q) 0 fit (by omega)).mp h
    omega
  obtain ⟨t,run,read,frame⟩:=
    VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.TypedFP.source_correct M original s
      (by simp only [original,forwardParameters]; omega) cover
  refine ⟨t,run,?_,?_⟩
  · intro c q active
    have bound : VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.TypedFP.outputOffset original s c q<original.outputNumel := by
      rw [offset]
      change Readback.row p s q*p.outputWidth+p.columnOffset+c.val<p.outputWidth*p.rows
      exact (row_mask _ _ _ _ _ fit c.isLt).mpr active
    have h:=read c q bound
    rw [offset,VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.TypedFP.rawExpected,coord] at h
    simpa only [original,forwardParameters,value,Reference.forwardValues] using h
  · intro region index outside
    apply frame
    rcases outside with h|h
    · exact Or.inl h
    · right
      intro c q active
      rw [offset] at active ⊢
      have ha : Readback.row p s q<p.rows := by
        change Readback.row p s q*p.outputWidth+p.columnOffset+c.val<p.outputWidth*p.rows at active
        exact (row_mask _ _ _ _ _ fit c.isLt).mp active
      exact h c q ha

theorem tile_correct (M : Algebra α) (d : Direction) (p : Parameters) (s : State α)
    (fit : p.columnOffset+11≤p.outputWidth) :
    ∃ t,FP.Structural.exec M (source d p) s=some t ∧
      (∀ c q,Readback.row p s q<p.rows →
        t.mem (output d) (offset p d (Readback.row p s q) c.val)=
          .mk .real (value M d p s (Readback.row p s q) c)) ∧
      (∀ region index,(region≠output d ∨ ∀ (c : Fin (channels d)) q,
        Readback.row p s q<p.rows → index≠offset p d (Readback.row p s q) c.val) →
        t.mem region index=s.mem region index) := by
  cases d
  · exact forward_tile M p s fit
  · exact Readback.backward_correct M p s fit

theorem at_correct (M : Algebra α) (d : Direction) (p : Parameters) (s : State α) (pid : Nat)
    (fit : p.columnOffset+11≤p.outputWidth) :
    ∃ t,FP.Structural.exec M (source d p) (atProgram s pid)=some t ∧
      (∀ r,r<p.rows → r/p.blockSize=pid → ∀ c,
        t.mem (output d) (offset p d r c.val)=.mk .real (value M d p s r c)) ∧
      (∀ region index,(region≠output d ∨ ∀ r,r<p.rows → r/p.blockSize=pid →
        ∀ c : Fin (channels d),index≠offset p d r c.val) → t.mem region index=s.mem region index) := by
  obtain ⟨t,run,read,frame⟩:=tile_correct M d p (atProgram s pid) fit
  refine ⟨t,run,?_,?_⟩
  · intro r hr hp c
    let q : TileIndex [p.blockSize]:=(⟨r%p.blockSize,Nat.mod_lt _ p.blockPositive⟩,())
    have eq : Readback.row p (atProgram s pid) q=r := by
      simp only [Readback.row,atProgram,
        ↓reduceIte,q]
      rw [←hp]
      exact Nat.mod_add_div _ _
    have values : value M d p (atProgram s pid) r c=value M d p s r c := by cases d <;> rfl
    simpa only [eq,values] using read c q (by simpa only [eq] using hr)
  · intro region index outside
    apply frame
    rcases outside with h|h
    · exact Or.inl h
    · right
      intro c q active
      apply h (Readback.row p (atProgram s pid) q) active
      change (q.1.val+p.blockSize*pid)/p.blockSize=pid
      rw [Nat.add_mul_div_left _ _ p.blockPositive,Nat.div_eq_of_lt q.1.isLt]
      simp

def execute (M : Algebra α) (d : Direction) (p : Parameters) : List Nat → State α → Option (State α)
  | [],s=>some s
  | pid::tail,s=>(FP.Structural.exec M (source d p) (atProgram s pid)).bind (execute M d p tail)

theorem value_stable (M : Algebra α) (d : Direction) (p : Parameters) (s t : State α) (r : Nat)
    (coords : ∀ o,t.mem coordinateRegion o=s.mem coordinateRegion o)
    (upstream : ∀ o,t.mem upstreamRegion o=s.mem upstreamRegion o)
    (gradient : ∀ c : Fin 3,t.mem gradientRegion (r*3+c.val)=s.mem gradientRegion (r*3+c.val)) :
    value M d p t r=value M d p s r := by
  funext c
  cases d <;> simp only [value,Readback.backwardExpected,coords,upstream,gradient]

theorem order_correct (M : Algebra α) (d : Direction) (p : Parameters)
    (order : List Nat) (nodup : order.Nodup) (s : State α)
    (fit : p.columnOffset+11≤p.outputWidth) :
    ∃ t,execute M d p order s=some t ∧
      (∀ r,r<p.rows → r/p.blockSize∈order → ∀ c,
        t.mem (output d) (offset p d r c.val)=.mk .real (value M d p s r c)) ∧
      (∀ region index,(region≠output d ∨ ∀ r,r<p.rows → r/p.blockSize∈order →
        ∀ c : Fin (channels d),index≠offset p d r c.val) → t.mem region index=s.mem region index) := by
  induction order generalizing s with
  | nil=>
    refine ⟨s,rfl,?_,?_⟩
    · intro r hr hm; simp at hm
    · intros; rfl
  | cons pid tail ih=>
    have nd:=List.nodup_cons.mp nodup
    obtain ⟨u,hu,uread,uframe⟩:=at_correct M d p s pid fit
    obtain ⟨t,ht,tread,tframe⟩:=ih nd.2 u
    have same : ∀ r,r/p.blockSize∈tail → value M d p u r=value M d p s r := by
      intro r member
      apply value_stable
      · intro o;apply uframe;left;cases d <;> decide
      · intro o;apply uframe;left;cases d <;> decide
      · intro c
        apply uframe
        cases d with
        | forward=>left;decide
        | backward=>
          right
          intro a ha hp e eq
          have ar:=offset_unique p .backward fit r a c e (by simpa [offset,stride,first] using eq)
          apply nd.1
          simpa [ar.1,hp] using member
    refine ⟨t,by simp [execute,hu,ht],?_,?_⟩
    · intro r hr member c
      by_cases tailMember : r/p.blockSize∈tail
      · rw [tread r hr tailMember c,same r tailMember]
      · have eq : r/p.blockSize=pid := by simpa [tailMember] using member
        rw [tframe]
        · exact uread r hr eq c
        · right
          intro a ha memberA e eqA
          have ar:=offset_unique p d fit r a c e eqA
          exact tailMember (ar.1 ▸ memberA)
    · intro region index outside
      rw [tframe]
      · apply uframe
        exact outside.imp id (fun h r hr hp c=>h r hr (by simp [hp]) c)
      · exact outside.imp id (fun h r hr hp c=>h r hr (by simp [hp]) c)

def blocks (p : Parameters) := (p.rows+p.blockSize-1)/p.blockSize
def launch (M : Algebra α) (d : Direction) (p : Parameters) (s : State α) :=
  execute M d p (List.range (blocks p)) s

theorem row_covered (p : Parameters) (r : Nat) (hr : r<p.rows) : r/p.blockSize<blocks p := by
  have pos:=p.blockPositive
  unfold blocks
  apply Nat.lt_of_succ_le
  apply (Nat.le_div_iff_mul_le pos).mpr
  have bound:=Nat.div_mul_le_self r p.blockSize
  rw [Nat.succ_mul]
  omega

theorem launch_correct (M : Algebra α) (d : Direction) (p : Parameters) (s : State α)
    (fit : p.columnOffset+11≤p.outputWidth) :
    ∃ t,launch M d p s=some t ∧
      (∀ r,r<p.rows → ∀ c,
        t.mem (output d) (offset p d r c.val)=.mk .real (value M d p s r c)) ∧
      (∀ region index,(region≠output d ∨ ∀ r,r<p.rows →
        ∀ c : Fin (channels d),index≠offset p d r c.val) → t.mem region index=s.mem region index) := by
  obtain ⟨t,run,read,frame⟩:=order_correct M d p (List.range (blocks p)) List.nodup_range s fit
  refine ⟨t,run,?_,?_⟩
  · intro r hr c
    exact read r hr (List.mem_range.mpr (row_covered p r hr)) c
  · intro region index outside
    apply frame
    exact outside.imp id (fun h r hr _ c=>h r hr c)

#axiomsClean launch_correct
end
end VeriTile.Bench.Repairs.FifthOrderSphHarmonics.Grid
