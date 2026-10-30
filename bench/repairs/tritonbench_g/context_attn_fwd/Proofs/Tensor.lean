import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.Entry
import VeriTile.Torch.FP32

namespace VeriTile.Bench.Repairs.ContextAttnFwd.CheckedTensor
open VeriTile.Triton FP.Structural NativeGraph
open VeriTile.Torch.FP32 (Tensor TensorLaws logicalValue)
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 4000000

def read (m : Physical.Memory) (p : Nat) : VeriTile.Torch.FP32.Word := ⟨Snapshot.load32 m p⟩

theorem byte_codec (m : Physical.Memory) (p : Nat) (word : Word)
    (bytes : ∀ b : Fin 4,m (p+b.val)=Physical.byte word b) : Snapshot.load32 m p=word := by
  unfold Snapshot.load32
  have h0 := bytes ⟨0,by omega⟩
  have h1 := bytes ⟨1,by omega⟩
  have h2 := bytes ⟨2,by omega⟩
  have h3 := bytes ⟨3,by omega⟩
  simp only [Nat.add_zero] at h0
  rw [h0,h1,h2,h3]
  apply BitVec.eq_of_toNat_eq
  have bound := word.isLt
  simp only [Physical.byte,BitVec.extractLsb'_toNat,Nat.shiftRight_eq_div_pow,BitVec.toNat_ofNat]
  norm_num at *
  omega

structure Output (v : Checked.Views) where
  tensor : Tensor
  metadata : tensor.metadata=v.o.metadata
  laws : TensorLaws tensor

def flatIndex (c : Host.Call) (idx : GridIndex c.grid) (i : TileIndex [c.layout.BM,c.layout.D]) : Nat :=
  ((c.start (Native.batchAt c idx)+VeriTile.Bench.TritonBenchG.ContextAttnFwd.Formula.queryRow c.layout (c.launchAt idx) i.1.val)*
    c.args.o.heads+VeriTile.Bench.TritonBenchG.ContextAttnFwd.Formula.head c.layout (c.launchAt idx))*c.args.o.dims+i.2.1.val

theorem flat_index (c : Host.Call) (hc : Geometry.Checked c extent) (idx : GridIndex c.grid)
    (i : TileIndex [c.layout.BM,c.layout.D]) (active : c.active idx i) :
    c.offset idx i=flatIndex c idx i ∧ flatIndex c idx i<c.args.o.rows*c.args.o.heads*c.args.o.dims := by
  have hb := Host.batch_bound c idx
  have head := Host.head_bound c idx
  have dimension := i.2.1.isLt
  have inputMetadata := hc.metadata _ hb
  have row : c.start (Native.batchAt c idx)+
      VeriTile.Bench.TritonBenchG.ContextAttnFwd.Formula.queryRow c.layout (c.launchAt idx) i.1.val<c.args.o.rows := by
    change _<c.length _ at active
    rw [hc.output_rows]
    simp only [Native.batchAt] at *
    omega
  have hh : VeriTile.Bench.TritonBenchG.ContextAttnFwd.Formula.head c.layout (c.launchAt idx)<c.args.o.heads := by
    rw [hc.output_heads];exact head
  have hd : i.2.1.val<c.args.o.dims := by rw [hc.output_dims];exact dimension
  constructor
  · exact Geometry.contiguous_address c.args.o hc.contiguous _ _ _ row hh hd
  · unfold flatIndex
    have first : (c.start (Native.batchAt c idx)+
        VeriTile.Bench.TritonBenchG.ContextAttnFwd.Formula.queryRow c.layout (c.launchAt idx) i.1.val)*c.args.o.heads+
        VeriTile.Bench.TritonBenchG.ContextAttnFwd.Formula.head c.layout (c.launchAt idx)<c.args.o.rows*c.args.o.heads := by
      nlinarith
    nlinarith

theorem direct (t : Tensor) (laws : TensorLaws t) (m : Physical.Memory)
    (contiguous : t.metadata.isContiguous=Bool.true) (negative : t.metadata.isNeg=Bool.false)
    (i : Nat) (hi : i<t.metadata.numel) :
    read m (t.metadata.dataPtr.toNat+4*i)=logicalValue (read m) t laws i := by
  open VeriTile.Torch.FP32 in
  have eq := direct_read_eq_logical (read m) t laws contiguous negative i hi
  open VeriTile.Torch.FP32 in
  have ptr := congrArg BitVec.toNat (actual_pointer t laws contiguous (by omega))
  open VeriTile.Torch.FP32 in
  have span := (dataView t laws contiguous).spanFits
  open VeriTile.Torch.FP32 in
  have bound : (dataView t laws contiguous).base<2^64 := by
    change (dataView t laws contiguous).base+4*t.metadata.numel ≤ 2^64 at span
    omega
  open VeriTile.Torch.FP32 in
  simp only [DataView.address,Nat.mul_zero,Nat.add_zero,Tensor.dataPtr,BitVec.toNat_ofNat,
    Nat.mod_eq_of_lt bound] at ptr
  open VeriTile.Torch.FP32 in
  simpa only [DataView.address,ptr] using eq

theorem logical_output (c : Host.Call) (v : Checked.Views) (B : Native.Bytes extent)
    (guard : Checked.tensorChecks c v) (hc : Geometry.Checked c extent)
    (storage : Checked.StorageBinding c v B) (out : Output v) (m : Physical.Memory)
    (ops : ScalarOps) (graph : State Expr) (idx : GridIndex c.grid)
    (i : TileIndex [c.layout.BM,c.layout.D]) (active : c.active idx i)
    (bytes : ∀ b : Fin 4,m (Physical.address c B (idx,i,b))=
      Physical.byte (Physical.outputWord c ops graph (idx,i,b)) b) :
    (logicalValue (read m) out.tensor out.laws (flatIndex c idx i)).bits=
      Physical.outputWord c ops graph (idx,i,⟨0,by omega⟩) := by
  rcases guard with ⟨_,_,_,shape,_,_,_,attrs,_,contiguous,_⟩
  have attributes := attrs v.o (by simp [Checked.all])
  have index := flat_index c hc idx i active
  have bound : flatIndex c idx i<out.tensor.metadata.numel := by
    rw [out.metadata]
    simpa [VeriTile.Torch.TensorMetadata.numel,shape,Nat.mul_assoc] using index.2
  rw [←direct out.tensor out.laws m
    (by simpa [out.metadata] using contiguous) (by simpa [out.metadata] using attributes.2.2.1) _ bound]
  change Snapshot.load32 m (out.tensor.metadata.dataPtr.toNat+4*flatIndex c idx i)=_
  rw [out.metadata,←storage.base c.regions.out v.o (by simp [Checked.roleViews]),←index.1]
  apply byte_codec
  intro b
  exact bytes b

/-- A normal strided Tensor's logical-to-storage address observation. For the
cache role this includes its observed batch gap; it is not an output law. -/
structure Input (c : Host.Call) (B : Native.Bytes extent) (rg : RegionName) where
  tensor : Tensor
  laws : TensorLaws tensor
  noNeg : tensor.metadata.isNeg=Bool.false
  offset : Nat → Nat
  address : ∀ i,i<tensor.metadata.numel →
    tensor.storage.address (tensor.physicalIndex i)=B.base rg+4*offset i

theorem logical_source (c : Host.Call) (B : Native.Bytes extent) (m : Physical.Memory)
    (roles : Snapshot.LogicalRoles c) (rg : RegionName)
    (role : rg∈[c.regions.query,c.regions.key,c.regions.value,c.regions.out])
    (input : Input c B rg) (i : Nat) (bound : i < input.tensor.metadata.numel) :
    ((Snapshot.initial c B m).mem rg (input.offset i)).read .real=
      Expr.leaf ((logicalValue (read m) input.tensor input.laws i).bits) := by
  rw [Snapshot.floating c B m rg _ (roles rg role)]
  simp only [logicalValue,input.noNeg,Bool.false_eq_true,if_false,input.address i bound,read]

#axiomsClean byte_codec
#axiomsClean logical_output
#axiomsClean logical_source
end
end VeriTile.Bench.Repairs.ContextAttnFwd.CheckedTensor
