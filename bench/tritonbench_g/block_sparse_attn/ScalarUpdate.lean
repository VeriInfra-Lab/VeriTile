import bench.tritonbench_g.block_sparse_attn.ScalarScore
import bench.tritonbench_g.block_sparse_attn.ScalarPrimitives

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def rowCarry (old : Carry α) (i : TileIndex [16,16]) : BlockSparseAttn.Acc α 4 :=
  ⟨old.maximum i.1,old.denominator i.1,old.output i (owner i.2.1.val)⟩

/-- Derive the scalar row step from initial memory and the explicit owner/tree
layout. The inputs contain no interpreter state or whole-graph equality. -/
theorem update_view (a : BlockSparseAttn.Ops α) (literal argument : ℝ → α)
    (zero : literal 0=a.zero) (p : Parameters) (batches : Nat)
    (s : State (Replicas α)) (m : BlockSparseAttn.Memory α) (rows cols : Int → Int)
    (views : Views s m rows cols) (first0 : 0≤first p s) (visit : Nat)
    (column0 : 0≤csrColumn p s (first p s+visit)) (old : Carry α) (i : TileIndex [16,16]) :
    rowCarry (update (Arithmetic.ofLegacy a literal argument) Bool.true p s
        (csrColumn p s (first p s+visit)) old) i=
      blockStep (layoutOf p batches rows cols) a m (argument p.scale)
        (batch p s) (head p s) (row s i.1) (channel s i.2.1) (rowCarry old i) visit := by
  let c := csrColumn p s (first p s+visit)
  let x := layoutOf p batches rows cols
  let ops := Arithmetic.ofLegacy a literal argument
  let sc : Fin 16 → α := score x a m (argument p.scale) (batch p s) (head p s) (row s i.1) visit
  let vals : Nat → α := fun j => vLoad x a m (batch p s) (head p s)
    (x.csr.key x.shape (head p s) (row s i.1) visit j) (channel s i.2.1)
  let bm := fun o => (BlockSparseAttn.SM90B16.tree o).eval a.max sc
  let pr := fun j => a.exp (a.sub (sc j) (bm (owner j.val)))
  let bs := fun o => (BlockSparseAttn.SM90B16.tree o).eval a.add pr
  let nm := fun o => a.max (old.maximum i.1 o) (bm o)
  let al := fun o => a.exp (a.sub (old.maximum i.1 o) (nm o))
  let be := fun o => a.exp (a.sub (bm o) (nm o))
  let nd := fun o => a.fma (old.denominator i.1 o) (al o) (a.mul (bs o) (be o))
  let sp := fun j => a.mul (pr j) (a.div (be (owner j.val)) (nd (owner j.val)))
  let own := owner i.2.1.val
  let seed := a.mul (old.output i own) (a.mul (al own) (a.div (old.denominator i.1 own) (nd own)))
  have hs (j : Fin 16) (o : Fin 4) : Component.scores ops Bool.true p s c (i.1,j,PUnit.unit) o=sc j :=
    congrFun (scores_view a literal argument zero p batches s m rows cols views first0 visit column0 (i.1,j,PUnit.unit)) o
  have hm : Component.maximum ops Bool.true p s c i.1=bm := by
    funext o
    unfold Component.maximum
    rw [reduceLine_selected]
    have same : (fun j : Fin 16 => Component.scores ops Bool.true p s c (i.1,j,PUnit.unit) (owner j.val))=sc :=
      funext (fun j => hs j (owner j.val))
    rw [same]
    rfl
  have hp (j : Fin 16) (o : Fin 4) :
      Component.probability ops Bool.true p s c (i.1,j,PUnit.unit) o=
        a.exp (a.sub (sc j) (bm o)) := by
    simp only [Component.probability,hs,hm]
    rfl
  have hd : Component.denominator ops Bool.true p s c i.1=bs := by
    funext o
    unfold Component.denominator
    rw [reduceLine_selected]
    have same : (fun j : Fin 16 => Component.probability ops Bool.true p s c (i.1,j,PUnit.unit) (owner j.val))=pr :=
      funext (fun j => hp j (owner j.val))
    rw [same]
    rfl
  have hn : Component.nextMaximum ops Bool.true p s c old i.1=nm := by
    funext o
    simp only [Component.nextMaximum,hm]
    rfl
  have ha : Component.alpha ops Bool.true p s c old i.1=al := by
    funext o
    simp only [Component.alpha,hn]
    rfl
  have hb : Component.beta ops Bool.true p s c old i.1=be := by
    funext o
    simp only [Component.beta,hm,hn]
    rfl
  have hl : Component.nextDenominator ops Bool.true p s c old i.1=nd := by
    funext o
    simp only [Component.nextDenominator,ite_true,ha,hb,hd]
    rfl
  have hsp (j : Fin 16) : Component.scaledProbability ops Bool.true p s c old
      (i.1,j,PUnit.unit) (owner j.val)=sp j := by
    simp only [Component.scaledProbability,hp,hb,hl]
    rfl
  have hseed : Component.scaledOutput ops Bool.true p s c old i own=seed := by
    simp only [Component.scaledOutput,Component.scale,ite_true,ha,hl]
    rfl
  have hv : (fun t : Fin 16 => value ops p s c (t,i.2.1,PUnit.unit))=
      (fun t : Fin 16 => broadcast (vals t.val)) := by
    funext t
    rw [value_view a literal argument zero p batches s m rows cols views c column0]
    change broadcast (vLoad x a m (batch p s) (head p s)
      (csrColumn p s (first p s+visit)*16+t.val) (channel s i.2.1))=_
    rw [column_view p batches s m rows cols views first0 i.1 visit]
    rfl
  have hout : Component.output ops Bool.true p s c old i own=
      BlockSparseAttn.dot a .mma16n8k8 16 (fun j => if h : j<16 then sp ⟨j,h⟩ else a.zero) vals seed := by
    simp only [Component.output,ite_true,tensor,broadcast]
    rw [hv]
    rw [chain_selected a literal argument _ vals _]
    rw [hseed]
    congr 2
    funext j
    split_ifs with hj
    · exact hsp ⟨j,hj⟩
    · rfl
  have outputOwner : BlockSparseAttn.owner (channel s i.2.1)=own :=
    owner_channel (s.pids 2) i.2.1
  rw [update_components]
  change (⟨Component.nextMaximum ops Bool.true p s c old i.1,
    Component.nextDenominator ops Bool.true p s c old i.1,
    Component.output ops Bool.true p s c old i own⟩ : BlockSparseAttn.Acc α 4)=_
  rw [hn,hl,hout]
  simp only [blockStep,normalizedStep,layoutOf,BlockSparseAttn.SM90B16.plan,rowCarry,outputOwner]
  rfl

#axiomsClean update_view
end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
