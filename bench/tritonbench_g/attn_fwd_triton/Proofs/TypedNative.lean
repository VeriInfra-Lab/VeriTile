import bench.tritonbench_g.attn_fwd_triton.Proofs.TypedPhysical
import bench.tritonbench_g.attn_fwd_triton.Proofs.NativeLaunch
import bench.tritonbench_g.attn_fwd_triton.Proofs.NativeScale

/-! Bind ordinary returned-view metadata to the actual native pointer formulas.
Only active accesses require exact addresses; dead padded pointers may wrap. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.TypedExecution
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.LayerNormFwd
open Address (Allocation wrap64 signed64 byteAddress)
open Verified.PhysicalCells (Layout Role width)
open Verified.NativeOffsets
noncomputable section
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false

structure Allocated (p : Config) (base : Role → Nat) (alloc : Role → Allocation) : Prop where
  bases : ∀ r, (alloc r).base=base r
  widths : ∀ r, (alloc r).width=width r
  covered : ∀ z h n d r, roleActive p z h n d r → roleOffset p z h n d r<(alloc r).extent

def machineOffset (p : Config) (z h n d : Nat) : Role → Nat
  | .query => matrix p.queryBatch p.queryHead p.queryRow p.queryDim z h n d
  | .key => matrix p.keyBatch p.keyHead p.keyRow p.keyDim z h n d
  | .value => matrix p.valueBatch p.valueHead p.valueRow p.valueDim z h n d
  | .output => matrix p.outputBatch p.outputHead p.outputRow p.outputDim z h n d
  | .queryScale | .keyScale => wrap64 n

theorem machineOffset_mod (p : Config) (z h n d : Nat) (r : Role) :
    machineOffset p z h n d r=wrap64 (roleOffset p z h n d r) := by
  cases r <;> simp only [machineOffset,roleOffset,matrix_mod,linearAddress]

/-- Every role uses the same base, width and extent as the physical view. -/
theorem native_address (p : Config) (a : Layout) (base : Role → Nat) (v : Views a p base)
    (alloc : Role → Allocation) (ha : Allocated p base alloc)
    (z h n d : Nat) (r : Role) (hr : roleActive p z h n d r) :
    machineOffset p z h n d r=roleOffset p z h n d r ∧
      signed64 (machineOffset p z h n d r)=(roleOffset p z h n d r : Int) ∧
      bytePointer (alloc r).base (alloc r).width (machineOffset p z h n d r)=a.address (roleKey p z h n d r) := by
  have hx := active_address (alloc r) _ _ (ha.covered z h n d r hr) (machineOffset_mod p z h n d r)
  refine ⟨hx.1,hx.2.1,?_⟩
  rw [hx.2.2,v.address z h n d r hr]
  simp [byteAddress,ha.bases r,ha.widths r,Nat.mul_comm]

theorem native_helper (p : Config) (a : Layout) (base : Role → Nat) (v : Views a p base)
    (alloc : Role → Allocation) (ha : Allocated p base alloc)
    (z h lower iterations lane d : Nat) :
    (roleActive p z h (lower+iterations*64+lane) d .key →
      bytePointer (alloc .key).base (alloc .key).width
        (keyValue p.keyBatch p.keyHead p.keyRow p.keyDim z h lane d lower iterations)=
        a.address (roleKey p z h (lower+iterations*64+lane) d .key)) ∧
    (roleActive p z h (lower+iterations*64+lane) d .value →
      bytePointer (alloc .value).base (alloc .value).width
        (keyValue p.valueBatch p.valueHead p.valueRow p.valueDim z h lane d lower iterations)=
        a.address (roleKey p z h (lower+iterations*64+lane) d .value)) := by
  constructor
  all_goals intro hr
  all_goals rw [keyValue_mod]
  · have hx := (native_address p a base v alloc ha z h (lower+iterations*64+lane) d .key hr).2.2
    simpa only [machineOffset_mod,roleOffset,linearAddress] using hx
  · have hx := (native_address p a base v alloc ha z h (lower+iterations*64+lane) d .value hr).2.2
    simpa only [machineOffset_mod,roleOffset,linearAddress] using hx

theorem native_packed (p : Config) (a : Layout) (base : Role → Nat) (v : Views a p base)
    (alloc : Role → Allocation) (ha : Allocated p base alloc)
    (head block : Nat) :
    (head*((p.context+127)/128)+block<p.batches*p.heads*((p.context+127)/128) →
      bytePointer (alloc .queryScale).base (alloc .queryScale).width
        (packedScale head ((p.context+127)/128) block)=
        a.address (roleKey p 0 0 (head*((p.context+127)/128)+block) 0 .queryScale)) ∧
    (head*((p.context+63)/64)+block<p.batches*p.heads*((p.context+63)/64) →
      bytePointer (alloc .keyScale).base (alloc .keyScale).width
        (packedScale head ((p.context+63)/64) block)=
        a.address (roleKey p 0 0 (head*((p.context+63)/64)+block) 0 .keyScale)) := by
  constructor
  all_goals intro hr
  all_goals rw [packedScale_mod]
  · exact (native_address p a base v alloc ha 0 0 _ 0 .queryScale ⟨rfl,rfl,rfl,hr⟩).2.2
  · exact (native_address p a base v alloc ha 0 0 _ 0 .keyScale ⟨rfl,rfl,rfl,hr⟩).2.2

theorem native_key_scale_helper (p : Config) (a : Layout) (base : Role → Nat) (v : Views a p base)
    (alloc : Role → Allocation) (ha : Allocated p base alloc)
    (head lower iterations : Nat)
    (hr : head*((p.context+63)/64)+(lower+iterations*64)/64<p.batches*p.heads*((p.context+63)/64)) :
    bytePointer (alloc .keyScale).base (alloc .keyScale).width
      (advance (packedScale head ((p.context+63)/64) (lower/64)) (wrap64 1) iterations)=
      a.address (roleKey p 0 0 (head*((p.context+63)/64)+(lower+iterations*64)/64) 0 .keyScale) := by
  rw [scale_progression]
  exact (native_address p a base v alloc ha 0 0 _ 0 .keyScale ⟨rfl,rfl,rfl,hr⟩).2.2

def launchShape (p : Config) : NativeLaunch.Shape := ⟨p.batches,p.heads,p.context⟩

def ControlCertificate (p : Config) (BD : Nat) : Prop :=
  (signed64 p.context=(p.context : Int) ∧
    signed64 (p.context+127)=(p.context+127 : Nat) ∧ signed64 (p.context+63)=(p.context+63 : Nat)) ∧
  (∀ start row, start*128<p.context → row<128 →
    signed64 (start*128+row)=(start*128+row : Nat) ∧
    signed64 ((start+1)*128)=((start+1)*128 : Nat) ∧
    signed64 (min ((start+1)*128) p.context)=(min ((start+1)*128) p.context : Nat)) ∧
  (∀ cursor lane, cursor<p.context → lane<64 →
    signed64 (p.context-cursor)=(p.context : Int)-(cursor : Int) ∧
    signed64 (cursor+lane)=(cursor+lane : Nat) ∧ signed64 (cursor+64)=(cursor+64 : Nat)) ∧
  (∀ lane, lane<BD → signed64 lane=(lane : Int)) ∧
  (∀ head, head<p.batches*p.heads →
    signed64 (head*((p.context+127)/128))=(head*((p.context+127)/128) : Nat) ∧
    signed64 (head*((p.context+63)/64))=(head*((p.context+63)/64) : Nat))

theorem control_certificate (p : Config) (BD : Nat)
    (h : NativeLaunch.CudaLaunchable (launchShape p) BD) : ControlCertificate p BD :=
  ⟨NativeLaunch.cdiv_inputs _ _ h,
    fun start row hs hr => NativeLaunch.row_arithmetic _ _ start row h hs hr,
    fun cursor lane hc hl => NativeLaunch.key_arithmetic _ _ cursor lane h hc hl,
    fun lane hl => NativeLaunch.feature_lanes _ _ lane h hl,
    fun head hh => NativeLaunch.packed_offsets _ _ head h hh⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.TypedExecution
