import bench.repairs.tritonbench_g.matmul_dequant_int4_grid_y_checked.Proofs.Tensor
import bench.repairs.tritonbench_g.matmul_dequant_int4.Proofs.DirectFused
import bench.repairs.tritonbench_g.matmul_dequantize_int4.Proofs.KernelTensor

namespace VeriTile.Bench.Repairs.MatmulDequantInt4GridYChecked.Raw
open VeriTile.Triton FP.Structural
open MatmulDequantInt4.Direct
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section
set_option maxHeartbeats 4000000
abbrev Fused := MatmulDequantizeInt4.Call

/-- The original retained raw ABI is unchanged. Production s1 does not call
this kernel. No sibling host/index checks are imported as new premises. -/
abbrev InputContract := RawAccepted

def view (t : MatmulDequantInt4.Host.Tensor) : MatmulDequantizeInt4.View :=
  ⟨t.rows,t.columns,t.strides.row,t.strides.column,t.rows*t.columns⟩

def call (p : RawInfo) : Fused :=
  ⟨(rawShape p).m,(rawShape p).n,(rawShape p).k,p.group,
    view p.input.tensor,view p.weight.tensor,view p.scale.tensor,view p.zero.tensor,
    view p.output.tensor,Bool.true⟩

private theorem view_valid (t : MatmulDequantInt4.Host.Tensor)
    (dtype : MatmulDequantInt4.Host.DType) (r c device : Nat)
    (h : t.Ordinary dtype r c device) (layout : t.strides=MatmulDequantInt4.Strides.canonical c) :
    (view t).Valid := by
  have hr := h.1
  have hc := h.2.1
  have rp := h.2.2.1
  have cp := h.2.2.2.1
  dsimp [view]
  rw [hr,hc,layout]
  refine ⟨cp,by simp [MatmulDequantInt4.Strides.canonical],Or.inl (by simp [MatmulDequantInt4.Strides.canonical]),Or.inr (Or.inr ?_)⟩
  simp only [MatmulDequantInt4.Strides.canonical,Nat.mul_one]
  have rr : r-1+1=r := by omega
  have cc : c-1+1=c := by omega
  nlinarith

theorem valid (p : RawInfo) (h : InputContract p) : (call p).Valid := by
  rcases h with ⟨_,scope,_,ha,hq,hs,hz,hc,la,lq,ls,lz,lc,_⟩
  have oa := ha.2.2.2.2.2.2.1
  have oq := hq.2.2.2.2.2.2.1
  have os := hs.2.2.2.2.2.2.1
  have oz := hz.2.2.2.2.2.2.1
  have oc := hc.2.2.2.2.2.2.1
  refine ⟨scope.shape.mpos,scope.shape.npos,scope.shape.kpos,scope.shape.gpos,
    scope.n256,scope.k128,scope.group128,scope.shape.groups,
    ⟨oa.1,oa.2.1⟩,⟨oq.1,oq.2.1⟩,⟨os.1,os.2.1⟩,⟨oz.1,oz.2.1⟩,⟨oc.1,oc.2.1⟩,
    ?_,view_valid _ _ _ _ _ oa la,view_valid _ _ _ _ _ oq lq,
    view_valid _ _ _ _ _ os ls,view_valid _ _ _ _ _ oz lz,view_valid _ _ _ _ _ oc lc,?_⟩
  · simp [call,view,la,MatmulDequantInt4.Strides.canonical]
  · intro impossible
    simp [call] at impossible

/-- Complete retained raw source quotation, both group branches and loop. -/
abbrev kernel (p : RawInfo) (i : Fin 17) := MatmulDequantizeInt4.KernelSource.kernel (call p) i

/-- The raw Boolean remains the source Boolean; it is not silently recomputed
on a narrowed subdomain. -/
theorem noGroups (p : RawInfo) (h : InputContract p) :
    (MatmulDequantizeInt4.params (call p) (0 : Fin 17)).noGroups=p.noGroups := by
  have eq := h.2.1.noGroupsEq
  change decide (p.group=p.input.tensor.columns)=p.noGroups
  cases hp : p.noGroups <;> simp_all [rawShape]

/-- All source configurations preserve order and literal launch metadata. -/
theorem configurations (i : Fin 17) :
    (MatmulDequantizeInt4.tunings i).BM=(fusedConfig i).tile.bm ∧
    (MatmulDequantizeInt4.tunings i).BN=(fusedConfig i).tile.bn ∧
    (MatmulDequantizeInt4.tunings i).BK=(fusedConfig i).tile.bk ∧
    (MatmulDequantizeInt4.tunings i).warps=(fusedConfig i).warps ∧
    (MatmulDequantizeInt4.tunings i).stages=(fusedConfig i).stages := by
  fin_cases i <;> decide

/-- Complete source Real execution and independent dense-matmul result under
exactly the pre-existing raw input contract. -/
theorem real (p : RawInfo) (h : InputContract p) (i : Fin 17)
    (initial : State ℝ) (packed : MatmulDequantizeInt4.KernelReal.Packed initial) :
    ∃ final,MatmulDequantizeInt4.KernelSource.run MatmulDequantizeInt4.KernelReal.scalar
        MatmulDequantizeInt4.KernelReal.dot (call p) i initial=some final ∧
      GridExecution.Post "C"
        (Geometry.address (MatmulDequantizeInt4.Grid.geometry (MatmulDequantizeInt4.params (call p) i)))
        (fun x : MatmulDequantizeInt4.Grid.Index (MatmulDequantizeInt4.params (call p) i) => .mk .fp16
          (MatmulDequantizeInt4.Real.mathematical (call p) (MatmulDequantizeInt4.RealHost.input (call p) initial)
            x.1.val x.2.val)) (fun _ => True) initial final :=
  MatmulDequantizeInt4.KernelReal.whole_grid (call p) (valid p h) i initial packed

/-- Native whole-kernel result/frame reuses only the full raw kernel theorem;
its fixed-K16 and three scalar ports remain explicit. -/
theorem native (p : RawInfo) (h : InputContract p) (i : Fin 17)
    (actual : MatmulDequantizeInt4.Native.Instructions)
    (mma : MatmulDequantizeInt4.Opcode → MatmulDequantizeInt4.ISA.MMA16)
    (ports : MatmulDequantizeInt4.Native.Obligations actual mma)
    (initial : MatmulDequantizeInt4.Physical.Bytes) :
    ∃ final,MatmulDequantizeInt4.Physical.source actual (call p) i initial=some final ∧
      (∀ x : MatmulDequantizeInt4.Grid.Index (MatmulDequantizeInt4.params (call p) i),∀ byte : Fin 2,
        final "C" (2*Geometry.address (MatmulDequantizeInt4.Grid.geometry
          (MatmulDequantizeInt4.params (call p) i)) x+byte.val)=
        MatmulDequantizeInt4.Physical.encode16 (MatmulDequantizeInt4.Physical.target mma (call p) i initial x) byte) ∧
      (∀ (r : RegionName) b,(r≠"C" ∨ ∀ x : MatmulDequantizeInt4.Grid.Index (MatmulDequantizeInt4.params (call p) i),
        b/2≠Geometry.address (MatmulDequantizeInt4.Grid.geometry (MatmulDequantizeInt4.params (call p) i)) x) →
        final r b=initial r b) :=
  MatmulDequantizeInt4.Physical.whole_bytes actual mma ports (call p) (valid p h) i initial

#axiomsClean valid
#axiomsClean noGroups
#axiomsClean configurations
#axiomsClean real
#axiomsClean native
end
end VeriTile.Bench.Repairs.MatmulDequantInt4GridYChecked.Raw
