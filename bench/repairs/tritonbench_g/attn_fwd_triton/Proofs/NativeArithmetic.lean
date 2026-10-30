import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.Address
/-! Pure fixed-width source pointer arithmetic shared by the independent Real
and opaque layers. No task execution theorem is imported. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.NativeOffsets
open VeriTile.Triton
open VeriTile.Bench.Repairs.LayerNormFwd.Address
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false
def linearAddress (sz sh sn sd z h n d : Nat) : Nat := z*sz+h*sh+n*sn+d*sd

theorem add_wrap (a b : Nat) : add64 (wrap64 a) (wrap64 b)=wrap64 (a+b) := by
  simp [add64,wrap64,Nat.add_mod]

theorem mul_wrap (a b : Nat) : mul64 (wrap64 a) (wrap64 b)=wrap64 (a*b) := by
  simp [mul64,wrap64,Nat.mul_mod]

def matrix (sz sh sn sd z h n d : Nat) : Nat :=
  add64 (add64 (add64 (mul64 (wrap64 z) (wrap64 sz)) (mul64 (wrap64 h) (wrap64 sh)))
    (mul64 (wrap64 n) (wrap64 sn))) (mul64 (wrap64 d) (wrap64 sd))

theorem matrix_mod (sz sh sn sd z h n d : Nat) :
    matrix sz sh sn sd z h n d=wrap64 (linearAddress sz sh sn sd z h n d) := by
  simp only [matrix,mul_wrap,add_wrap,linearAddress]

def advance (ptr step : Nat) : Nat → Nat
  | 0 => ptr
  | i+1 => add64 (advance ptr step i) step

theorem advance_mod (ptr step i : Nat) :
    advance (wrap64 ptr) (wrap64 step) i=wrap64 (ptr+i*step) := by
  induction i with
  | zero => simp [advance]
  | succ i ih =>
    rw [advance,ih,add_wrap]
    congr 1
    ring

/-- Original K/V pointer tile, explicit helper offset, then every repeated
BN*stride advance. No bound is imposed on BN*stride or the final advance. -/
def keyValue (sz sh sn sd z h lane d lower iterations : Nat) : Nat :=
  advance (add64 (matrix sz sh sn sd z h lane d) (mul64 (wrap64 sn) (wrap64 lower)))
    (mul64 (wrap64 64) (wrap64 sn)) iterations

theorem keyValue_mod (sz sh sn sd z h lane d lower iterations : Nat) :
    keyValue sz sh sn sd z h lane d lower iterations=
      wrap64 (linearAddress sz sh sn sd z h (lower+iterations*64+lane) d) := by
  rw [keyValue,matrix_mod,mul_wrap,mul_wrap,add_wrap,advance_mod]
  congr 1
  simp only [linearAddress]
  ring

def packedScale (head count block : Nat) : Nat :=
  add64 (mul64 (wrap64 head) (wrap64 count)) (wrap64 block)

theorem packedScale_mod (head count block : Nat) :
    packedScale head count block=wrap64 (head*count+block) := by
  simp only [packedScale,mul_wrap,add_wrap]

def bytePointer (base width offset : Nat) : Nat :=
  add64 (wrap64 base) (mul64 (wrap64 offset) (wrap64 width))

theorem bytePointer_mod (base width offset : Nat) :
    bytePointer base width offset=wrap64 (base+offset*width) := by
  simp only [bytePointer,mul_wrap,add_wrap]

theorem bytePointer_wrapped (base width offset : Nat) :
    bytePointer base width (wrap64 offset)=bytePointer base width offset := by
  simp [bytePointer,wrap64,Nat.mod_mod]

/-- Ordinary finite allocation metadata discharges active element and byte
exactness, including high unsigned pointer bases. -/
theorem active_address (a : Allocation) (offset machine : Nat)
    (ha : offset<a.extent) (hm : machine=wrap64 offset) :
    machine=offset ∧ signed64 machine=(offset : Int) ∧
      bytePointer a.base a.width machine=byteAddress a offset := by
  have he := byte_address a ha
  have ho : offset<2^63 := lt_of_lt_of_le ha a.signed_extent
  have hm' : machine=offset := hm.trans (wrap64_exact ho)
  refine ⟨hm',by rw [hm']; exact signed64_exact ho,?_⟩
  rw [hm',bytePointer_mod]
  exact Nat.mod_eq_of_lt he.2.2.1

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.NativeOffsets
