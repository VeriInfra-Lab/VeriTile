import bench.tritonbench_g.lora_expand_gemv.Proofs.BranchReference
import bench.tritonbench_g.lora_expand_gemv.Proofs.Geometry
namespace VeriTile.Bench.Repairs.LoraExpandGemv
open VeriTile.Triton FP.Structural FPValue
open FP.Equational (ReductionPlan)
noncomputable section
open Classical
attribute [local simp] VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches.inputFormat VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches.weightFormat VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches.effectiveFormat VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches.promote VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches.productFormat VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches.resultFormat
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false
set_option maxRecDepth 30000
abbrev Memory := RegionName → Nat → Cell Word

def entry (g : Geometry) (raw : Nat → Int) : Memory := fun r o =>
  if r="I" then .mk .int (denseIndices g raw o)
  else if r="W" then .mk .fp16 (.leaf r o) else .mk .real (.leaf r o)

def program (g : Geometry) (b n : Nat) (mem : Memory) : State Word where
  mem := mem
  undef := fun d _ _ => defaultValue d
  regs := fun _ _ _ => none
  pids := fun ax => if ax=0 then n else if ax=1 then b else 0
  numPids := fun ax => if ax=0 then g.N else if ax=1 then g.B else 1

def put (mem : Memory) (o : Nat) (v : Cell Word) : Memory :=
  fun r a => if r="Y" ∧ a=o then v else mem r a

/-- Independent logical scalar/tree expression. All padded products and the
chosen tree are explicit. Native interpretation may fuse adjacent half mul/add. -/
def product (g : Geometry) (raw : Nat → Int) (b n : Nat) (k : Fin g.block) : Word :=
  .mul .half
    (castWord .ieeeValue .single .half (if k.val<g.K then .leaf "X" (xoff g b k.val) else .zero))
    (if k.val<g.K then .leaf "W" (woff g (logicalIndex g raw b).toNat n k.val) else .zero)

def value (plans : ∀ n, ReductionPlan n) (g : Geometry) (raw : Nat → Int) (b n : Nat) : Word :=
  let a := castWord .ieeeValue .half .single
    (treeValue .ieeeValue .half (product g raw b n) (plans g.block).tree)
  if g.add then .add .single a (.leaf "Y" (yoff g b n)) else a

def resultCell (plans : ∀ n, ReductionPlan n) (g : Geometry) (raw : Nat → Int) (b n : Nat) : Cell Word :=
  if logicalIndex g raw b=-1 then entry g raw "Y" (yoff g b n)
  else .mk .real (value plans g raw b n)

/-- Current-memory invariant, not independent per-program snapshots. Previously
written Y cells may differ; every source read for this fresh coordinate agrees. -/
def Fresh (g : Geometry) (raw : Nat → Int) (mem : Memory) (b n : Nat) : Prop :=
  (∀ r o,r≠("Y" : RegionName) → mem r o=entry g raw r o) ∧
  mem "Y" (yoff g b n)=entry g raw "Y" (yoff g b n)

private theorem fold_one {ι : Type} (xs : List ι) (P : ι → Prop) [DecidablePred P]
    (o : Nat) (v : Cell Word) (mem : Memory) :
    xs.foldl (fun m i => if P i then put m o v else m) mem=
      if ∃ i∈xs,P i then put mem o v else mem := by
  induction xs generalizing mem with
  | nil => simp
  | cons i xs ih =>
    simp only [List.foldl_cons,ih,List.mem_cons]
    by_cases hi : P i <;> by_cases ht : ∃ j∈xs,P j
    · simp only [hi,ite_true,ht]
      funext r a
      by_cases ho : r="Y" ∧ a=o <;> simp [put,hi,ht,ho]
    · simp [hi,ht]
    · simp [hi,ht]
    · simp [hi,ht]

theorem write_single (plans : ∀ n, ReductionPlan n) (g : Geometry)
    (b n : Nat) (mem : Memory) (hn : n<g.N) :
    Branches.expectedMemory .ieeeValue plans .singleHalf .single g.parameters (program g b n mem)=
      if (mem "I" b).read .int=-1 then mem else
        put mem (yoff g b n)
          (Branches.outputCell .ieeeValue plans .singleHalf .single g.parameters
            (program g b n mem) mem 0) := by
  have hl := length_one g (by omega)
  have hs : Branches.selected g.parameters (program g b n mem)=(mem "I" b).read .int := rfl
  rw [Branches.expectedMemory,hs]
  split <;> try rfl
  rw [hl]
  simp only [Branches.reference,hl,show (0:Nat)<1 by decide,ite_true,Nat.zero_add]
  change Branches.writeBlock .ieeeValue plans .singleHalf .single g.parameters (program g b n mem) mem 0=_
  have hyo : Branches.outputOffset g.parameters (program g b n mem) 0=yoff g b n := by
    unfold Branches.outputOffset Branches.outputBase
    rw [hl]
    simp [program,Geometry.parameters,yoff]
  have hb : ∃ i∈TileShape.allIndices [256], i.1.val=0 :=
    ⟨(⟨0,by decide⟩,PUnit.unit),TileShape.mem_allIndices _ _,rfl⟩
  have hw : Branches.writeBlock .ieeeValue plans .singleHalf .single g.parameters (program g b n mem) mem 0=
      (TileShape.allIndices [256]).foldl (fun m i => if i.1.val=0 then
        put m (yoff g b n) (Branches.outputCell .ieeeValue plans .singleHalf .single g.parameters
          (program g b n mem) mem 0) else m) mem := by
    unfold Branches.writeBlock
    apply congrArg (fun f => List.foldl f mem (TileShape.allIndices [256]))
    funext m i
    by_cases hz : i.1.val=0
    · simp only [hz,Nat.zero_add,hl,show (0:Nat)<1 by decide,ite_true]
      rw [hyo]
      rfl
    · have hnot : ¬i.1.val<1 := by omega
      simp [hl,hz,hnot]
  rw [hw]
  simpa only [hb,ite_true] using fold_one (TileShape.allIndices [256]) (fun i => i.1.val=0)
    (yoff g b n) (Branches.outputCell .ieeeValue plans .singleHalf .single g.parameters
      (program g b n mem) mem 0) mem

theorem source_value_of_reads (plans : ∀ n, ReductionPlan n) (g : Geometry) (raw : Nat → Int)
    (v : Valid g raw) (b n : Nat) (hb : b<g.B) (hn : n<g.N) (mem : Memory)
    (indexRead : (mem "I" b).read .int=logicalIndex g raw b)
    (xRead : ∀ k,k<g.K → mem "X" (xoff g b k)=entry g raw "X" (xoff g b k))
    (wRead : ∀ k,k<g.K → mem "W" (woff g (logicalIndex g raw b).toNat n k)=
      entry g raw "W" (woff g (logicalIndex g raw b).toNat n k))
    (yRead : mem "Y" (yoff g b n)=entry g raw "Y" (yoff g b n))
    (active : logicalIndex g raw b≠-1) :
    Branches.outputCell .ieeeValue plans .singleHalf .single g.parameters (program g b n mem) mem 0=
      .mk .real (value plans g raw b n) := by
  have hl := length_one g (by omega)
  have hi : (mem "I" b).read .int=logicalIndex g raw b := indexRead
  have hindex : 0≤logicalIndex g raw b := (v.ids b hb).resolve_left active |>.1
  have ha : ∀ k : Fin g.block,
      Branches.aValue .ieeeValue .singleHalf g.parameters (program g b n mem) k=
        castWord .ieeeValue .single .half
          (if k.val<g.K then .leaf "X" (xoff g b k.val) else .zero) := by
    intro k
    by_cases hk : k.val<g.K
    · have hx := xRead k.val hk
      dsimp only [xoff] at hx
      simp [Branches.aValue,Branches.aLoaded,Branches.aOffset,Branches.dtype,
        Branches.inputFormat,Branches.weightFormat,Geometry.parameters,program,hk,
        hx,entry,xoff,Cell.read_same,toFloat]
    · have he : g.even=Bool.false := by
        cases h : g.even
        · rfl
        · have := v.even h; have := k.isLt; omega
      simp [Branches.aValue,Branches.aLoaded,Branches.dtype,Branches.inputFormat,
        Branches.weightFormat,Geometry.parameters,program,hk,he]
  have hbindex : ((g.l0 : Int)*logicalIndex g raw b+(n : Int)*1*(g.wn : Int)).toNat=
      (logicalIndex g raw b).toNat*g.l0+n*g.wn := by
    rw [Int.toNat_add (by positivity) (by positivity),Int.toNat_mul (by omega) hindex]
    simp [Int.toNat_mul,Nat.mul_comm]
  have hbo : ∀ k : Nat,Branches.bOffset g.parameters (program g b n mem) 0 k=
      woff g (logicalIndex g raw b).toNat n k := by
    intro k
    unfold Branches.bOffset Branches.weightBase
    rw [hl]
    change ((g.l0 : Int)*(mem "I" b).read .int+(n : Int)*1*(g.wn : Int)).toNat+0*g.wn+k*g.wk=_
    rw [hi,hbindex]
    simp [woff]
  have hbv : ∀ k : Fin g.block,
      Branches.bValue .singleHalf g.parameters (program g b n mem) mem 0 k=
        if k.val<g.K then .leaf "W" (woff g (logicalIndex g raw b).toNat n k.val) else .zero := by
    intro k
    unfold Branches.bValue
    rw [hl,hbo]
    by_cases hk : k.val<g.K
    · simp [hk,Geometry.parameters,Branches.dtype,Branches.weightFormat,toFloat,Cell.read_same,
        wRead k.val hk,entry]
    · simp [hk,Geometry.parameters]
  have hp : Branches.product .ieeeValue .singleHalf g.parameters (program g b n mem) mem 0=
      product g raw b n := by
    funext k
    unfold Branches.product
    rw [ha,hbv]
    change mulWord .ieeeValue .half
      (castWord .ieeeValue .half .half (castWord .ieeeValue .single .half
        (if k.val<g.K then .leaf "X" (xoff g b k.val) else .zero)))
      (castWord .ieeeValue .half .half
        (if k.val<g.K then .leaf "W" (woff g (logicalIndex g raw b).toNat n k.val) else .zero))=_
    simp [product,mulWord,castWord_same]
  have hy : toFloat .real ((mem "Y" (yoff g b n)).read .real)=.leaf "Y" (yoff g b n) := by
    rw [yRead]; simp [entry,Cell.read_same,toFloat]
  have hyo : Branches.outputOffset g.parameters (program g b n mem) 0=yoff g b n := by
    unfold Branches.outputOffset Branches.outputBase
    rw [hl]
    simp [program,Geometry.parameters,yoff]
  simp only [Branches.outputCell,Branches.added,Branches.accumulated,hp,hyo]
  cases hadd : g.add <;>
    simp [Branches.dtype,Branches.resultFormat,Branches.productFormat,Branches.effectiveFormat,
      Branches.promote,Branches.weightFormat,Geometry.parameters,hadd,castWord_same,
      value,Branches.outputOffset,Branches.outputBase,program,hl,yoff,hy,
      yRead,entry,ofFloat,toFloat,Cell.read_same,addWord]

  all_goals simpa only [toFloat,yoff] using hy

/-- Historical whole-memory interface, derived from the finite source reads. -/
theorem source_value (plans : ∀ n, ReductionPlan n) (g : Geometry) (raw : Nat → Int)
    (v : Valid g raw) (b n : Nat) (hb : b<g.B) (hn : n<g.N) (mem : Memory)
    (fresh : Fresh g raw mem b n) (active : logicalIndex g raw b≠-1) :
    Branches.outputCell .ieeeValue plans .singleHalf .single g.parameters (program g b n mem) mem 0=
      .mk .real (value plans g raw b n) := by
  apply source_value_of_reads plans g raw v b n hb hn mem
  · rw [fresh.1 "I" b (by decide)]
    simp [entry,denseIndices,Cell.read_same]
  · intro k _; exact fresh.1 "X" _ (by decide)
  · intro k _; exact fresh.1 "W" _ (by decide)
  · exact fresh.2
  · exact active

#axiomsClean write_single
#axiomsClean source_value_of_reads
#axiomsClean source_value
#specNonCircular value avoiding [Branches.source,FP.Structural.exec,Branches.expectedMemory]
end
end VeriTile.Bench.Repairs.LoraExpandGemv
