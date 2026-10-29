import bench.repairs.tritonbench_g.bgmv_expand_slice_physical_tile_checked.SignedGrid
import bench.repairs.tritonbench_g.bgmv_expand_slice_physical_tile_checked.Memory
import bench.repairs.tritonbench_g.bgmv_expand_slice_physical_tile_checked.Target
import bench.repairs.tritonbench_g.bgmv_expand_slice_physical_tile_checked.Proofs.Direct.FPScalarTree
namespace VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.Real
open VeriTile Triton FP.Structural Direct FPValues
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 4000000

def algebra : Algebra ℝ where
  literal := fun _ _ x => x
  negInf := 0
  binary := fun _ _ op a b => match op with
    | .add => a+b | .sub => a-b | .mul => a*b | _ => 0
  unary := fun _ _ a => a
  cast := fun _ _ _ a => a
  fromNat := fun _ n => n
  fromInt := fun _ n => n
  fp32Bits := fun _ => 0
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ _ => 0
  reduceSum := fun _ {_} _ _ _ _ => 0

def reduction : Reduction ℝ where
  apply := fun _ dtype {shape} ax keep xs out =>
    ofFloat dtype (∑ k : Fin (TileShape.axisDim shape ax),
      Direct.FPScalarTree.inputs dtype shape ax keep xs out k)

def initial (r : Request) (heap : ByteMemory) (read : Nat → ℝ) : Data ℝ where
  input := fun off => read (r.x.dataPtr.toNat+4*off)
  weights := fun off => read (r.w.dataPtr.toNat+2*off)
  output := fun off => read (r.y.dataPtr.toNat+4*off)
  indices := indices r heap
  other := fun _ _ => .mk .real 0
  padding := fun dtype _ _ => defaultValue dtype

def formula (h : Call) (p : Geometry.Program h) (data : Data ℝ) : ℝ :=
  let total := ∑ k : Fin h.K, data.input (p.batch*h.K+k.val)*
    data.weights ((data.indices p.batch).toNat*(h.N*h.K)+p.row*h.K+k.val)
  if h.add then total+data.output (SourceGrid.address h p) else total

theorem padded_sum (f : Nat → ℝ) (K BK : Nat) (covered : K≤BK) :
    (∑ k : Fin BK,if k.val<K then f k.val else 0)=∑ k : Fin K,f k.val := by
  rw [Fin.sum_univ_eq_sum_range (fun k => if k<K then f k else 0) BK,
    Fin.sum_univ_eq_sum_range f K]
  calc
    (∑ k ∈ Finset.range BK,if k<K then f k else 0)=
        ∑ k ∈ Finset.range K,if k<K then f k else 0 := by
      symm
      apply Finset.sum_subset (Finset.range_mono covered)
      intro k _ hk
      exact if_neg (by simpa only [Finset.mem_range] using hk)
    _=∑ k ∈ Finset.range K,f k := by
      apply Finset.sum_congr rfl
      intro k hk
      simp only [if_pos (Finset.mem_range.mp hk)]

theorem value_formula (h : Call) (p : Geometry.Program h) (data : Data ℝ) :
    SourceGrid.value algebra reduction h p data=formula h p data := by
  have product (k : Fin h.BK) :
      products algebra h p.batch (data.indices p.batch).toNat p.row data (0,k,PUnit.unit)=
        if k.val<h.K then data.input (p.batch*h.K+k.val)*
          data.weights ((data.indices p.batch).toNat*(h.N*h.K)+p.row*h.K+k.val) else 0 := by
    by_cases hk : k.val<h.K <;> norm_num [products,loaded,weight,algebra,hk]
  have total : sums algebra reduction h p.batch (data.indices p.batch).toNat p.row data (0,PUnit.unit)=
      ∑ k : Fin h.K,data.input (p.batch*h.K+k.val)*
        data.weights ((data.indices p.batch).toNat*(h.N*h.K)+p.row*h.K+k.val) := by
    change (∑ k : Fin h.BK,products algebra h p.batch (data.indices p.batch).toNat p.row data (0,k,PUnit.unit))=_
    simp_rw [product]
    exact padded_sum (fun k => data.input (p.batch*h.K+k)*
      data.weights ((data.indices p.batch).toNat*(h.N*h.K)+p.row*h.K+k)) h.K h.BK h.rankCovered
  change (if h.add then sums algebra reduction h p.batch (data.indices p.batch).toNat p.row data (0,PUnit.unit)+
      data.output (SourceGrid.address h p) else
      sums algebra reduction h p.batch (data.indices p.batch).toNat p.row data (0,PUnit.unit))=formula h p data
  rw [total]
  rfl

def execute (r : Request) (heap : ByteMemory) (h : Admitted r heap) (read : Nat → ℝ) : Option (Data ℝ) :=
  Signed.Grid.run algebra reduction (call r heap h) (Geometry.programs (call r heap h)) (initial r heap read)

theorem source_result (r : Request) (heap : ByteMemory) (h : Admitted r heap) (read : Nat → ℝ) :
    ∃ final,execute r heap h read=some final ∧
      (∀ p : Program r heap h,indices r heap p.batch≠-1 →
        final.output (outputIndex r heap h p)=formula (call r heap h) p (initial r heap read)) := by
  have proof := Signed.Grid.complete r.x r.w r.y r.i r.scalars (indices r heap) h
    algebra reduction (initial r heap read) (fun _ _ => rfl)
  refine ⟨_,proof.1,?_⟩
  intro p active
  exact (proof.2 p active).trans (value_formula (call r heap h) p (initial r heap read))

#axiomsClean value_formula
#axiomsClean source_result
#specNonCircular formula avoiding [execute,Signed.Grid.run,SourceGrid.fold,FP.Structural.exec,Surface.source]
end
end VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.Real
