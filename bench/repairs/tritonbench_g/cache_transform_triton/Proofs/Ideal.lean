import bench.repairs.tritonbench_g.cache_transform_triton.Proofs.Admission

namespace VeriTile.Bench.Repairs.CacheTransform.Ideal
open VeriTile.Triton FP.Structural
noncomputable section

/-- The four physical byte lanes of each logical FP32 element belong to the
same actual source tile. This is a geometry fact, with no Real encoding. -/
theorem word_lanes (words q b : Nat) (hq : q<words) (hb : b<4) :
    q/64<blocks (words*4) ∧ q%64*4+b<256 ∧
    q/64*256+(q%64*4+b)=q*4+b ∧ q*4+b<words*4 := by
  unfold blocks
  omega

/-- Exact source address of every byte of a word. The byte source therefore
projects to one logical word copy, for any interpretation of those words. -/
theorem source_word_address (p : Parameters) (rows cache : Nat → Nat)
    (width r c b : Nat) (hw : p.rowBytes=width*4) (hc : c<width) (hb : b<4) :
    byteTarget p rows cache ((r*width+c)*4+b)=cache ((rows r*width+c)*4+b) := by
  obtain ⟨coord,cb⟩ := stored_word_coordinate r c b width 4 hc hb (by decide)
  rw [coord,←hw,byteTarget_coordinate p rows cache r (c*4+b) (by omega) (by omega)]
  rw [hw]
  congr 1
  ring

def wordAddress (width : Nat) (rows : Nat → Nat) (q : Nat) : Nat :=
  (rows (q*4/(width*4))*(width*4)+q*4%(width*4))/4

theorem wordAddress_coordinate (width : Nat) (rows : Nat → Nat) (r c : Nat) (hc : c<width) :
    wordAddress width rows (r*width+c)=rows r*width+c := by
  let p : Parameters := ⟨⟨"cache"⟩,⟨"rows"⟩,⟨"output"⟩,0,width*4,0,256⟩
  have geometry := source_word_address p rows id width r c 0 rfl hc (by decide)
  change (rows (((r*width+c)*4)/(width*4))*(width*4)+((r*width+c)*4)%(width*4))=
    (rows r*width+c)*4 at geometry
  unfold wordAddress
  rw [geometry]
  omega

def wordEvents (words : Nat) : List Nat :=
  (List.range words).filter fun q => decide (q/64<blocks (words*4))

theorem wordEvents_eq (words : Nat) : wordEvents words=List.range words := by
  apply List.filter_eq_self.mpr
  intro q hq
  exact decide_eq_true ((word_lanes words q 0 (List.mem_range.mp hq) (by decide)).1)

/-- Logical word projection of the actual byte-copy source grid. The source
row-table lookup and div/mod address calculation are retained. `word_lanes`
and `source_word_address` justify flattening its complete 256-byte tile grid
to this complete word workset. Payloads are arbitrary Reals, without a codec. -/
def run (out : RegionName) (n width : Nat) (rows : Nat → Nat) (input : Nat → ℝ)
    (s : State ℝ) : State ℝ :=
  (wordEvents (n*width)).foldl
    (fun state q => state.write out q (.mk .real (input (wordAddress width rows q)))) s

theorem readback (out : RegionName) (n width : Nat) (rows : Nat → Nat) (input : Nat → ℝ)
    (s : State ℝ) (r c : Nat) (hr : r<n) (hc : c<width) :
    (run out n width rows input s).mem out (r*width+c)=.mk .real (input (rows r*width+c)) := by
  have bound : r*width+c<n*width := by nlinarith
  have h := State.scatter_readback_list out id
    (fun q => Cell.mk .real (input (wordAddress width rows q)))
    (List.range (n*width)) s (r*width+c) List.nodup_range (List.mem_range.mpr bound) (fun _ _ h => h)
  simpa only [run,wordEvents_eq,id_eq,wordAddress_coordinate width rows r c hc] using h

theorem frame (out : RegionName) (n width : Nat) (rows : Nat → Nat) (input : Nat → ℝ)
    (s : State ℝ) (region : RegionName) (q : Nat) (miss : region≠out ∨ n*width ≤ q) :
    (run out n width rows input s).mem region q=s.mem region q := by
  unfold run
  rw [wordEvents_eq]
  apply State.scatter_frame
  exact miss.imp id (fun outside i hi same => by have bound := List.mem_range.mp hi; omega)

def observed (s : State ℝ) (out : RegionName) (n width : Nat) : List (Fin width → ℝ) :=
  List.ofFn (fun r : Fin n => fun c => (s.mem out (r.val*width+c.val)).read .real)

theorem mathematical_readback (out : RegionName) (mode : Mode) (lengths : List Nat)
    (width : Nat) (input : Nat → ℝ) (s : State ℝ) :
    observed (run out (preparedRows mode lengths).length width (tableRows mode lengths) input s)
      out (preparedRows mode lengths).length width=
    mathematical mode lengths (fun r => fun c : Fin width => input (r*width+c.val)) := by
  rw [←preparedRows_math]
  apply List.ext_getElem
  · simp [observed]
  · intro i hi hj
    have bound : i<(preparedRows mode lengths).length := by simpa using hj
    simp only [observed,List.getElem_ofFn,List.getElem_map]
    funext c
    rw [readback out _ width (tableRows mode lengths) input s i c.val bound c.isLt,Cell.read_same,
      tableRows_get mode lengths i bound]

structure Result (width : Nat) where
  state : State ℝ
  value : List (Fin width → ℝ) × List (Fin width → ℝ)
  objects : RegionName × RegionName
  device : Nat

def call (cos sin : RegionName) (mode : Mode) (lengths : List Nat) (width : Nat)
    (cosInput sinInput : Nat → ℝ) (s : State ℝ) (caller : Nat) : Result width :=
  let n := (preparedRows mode lengths).length
  let afterCos := run cos n width (tableRows mode lengths) cosInput s
  let afterSin := run sin n width (tableRows mode lengths) sinInput afterCos
  ⟨afterSin,(observed afterSin cos n width,observed afterSin sin n width),(cos,sin),caller⟩

theorem pair_math (cos sin : RegionName) (different : cos≠sin) (mode : Mode)
    (lengths : List Nat) (width : Nat) (cosInput sinInput : Nat → ℝ)
    (s : State ℝ) (caller : Nat) :
    let result := call cos sin mode lengths width cosInput sinInput s caller
    result.value.1=mathematical mode lengths (fun r => fun c : Fin width => cosInput (r*width+c.val)) ∧
    result.value.2=mathematical mode lengths (fun r => fun c : Fin width => sinInput (r*width+c.val)) ∧
    (∀ region q,(region≠cos ∨ (preparedRows mode lengths).length*width ≤ q) →
      (region≠sin ∨ (preparedRows mode lengths).length*width ≤ q) →
      result.state.mem region q=s.mem region q) ∧ result.objects=(cos,sin) ∧ result.device=caller := by
  refine ⟨?_,?_,?_,rfl,rfl⟩
  · change observed (run sin _ _ _ _ _) cos _ _=_
    have unchanged : observed (run sin (preparedRows mode lengths).length width (tableRows mode lengths) sinInput
        (run cos (preparedRows mode lengths).length width (tableRows mode lengths) cosInput s))
        cos (preparedRows mode lengths).length width=
        observed (run cos (preparedRows mode lengths).length width (tableRows mode lengths) cosInput s)
        cos (preparedRows mode lengths).length width := by
      unfold observed
      congr 1
      funext r c
      rw [frame sin _ _ _ _ _ cos _ (Or.inl different)]
    rw [unchanged]
    exact mathematical_readback cos mode lengths width cosInput s
  · exact mathematical_readback sin mode lengths width sinInput _
  · intro region q hc hs
    change (run sin _ _ _ _ _).mem region q=_
    rw [frame sin _ _ _ _ _ region q hs,frame cos _ _ _ _ _ region q hc]

#axiomsClean word_lanes
#axiomsClean source_word_address
#axiomsClean mathematical_readback
#axiomsClean pair_math
end
end VeriTile.Bench.Repairs.CacheTransform.Ideal
