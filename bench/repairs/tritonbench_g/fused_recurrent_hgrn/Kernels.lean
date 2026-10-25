import VeriTile.Triton
import VeriTile.Triton.DSL.CallSignature

namespace VeriTile.Bench.Repairs.FusedRecurrentHgrn
open VeriTile.Triton
noncomputable section
set_option maxRecDepth 10000
set_option maxHeartbeats 5000000

def forwardSignature : CallSignature where
 pointers := [("x",.fp32),("g",.fp32),("o",.fp32),("h0",.fp32),("ht",.fp32)]
 scalars := []
 constexpr := ["T","D","BD","USE_INITIAL_STATE","STORE_FINAL_STATE"]
 readPorts := ["x","g","h0"]
 writePorts := ["o","ht"]
def backwardSignature : CallSignature where
 pointers := [("g",.fp32),("o",.fp32),("dx",.fp32),("dg",.fp32),("do",.fp32),("h0",.fp32),("dht",.fp32)]
 scalars := []
 constexpr := ["T","D","BD","USE_INITIAL_STATE","USE_FINAL_GRAD"]
 readPorts := ["g","o","h0","do","dht"]
 writePorts := ["dx","dg"]

/-- Raw body of `fused_recurrent_hgrn_repaired.py:fused_recurrent_hgrn_fwd_kernel`. The call signature carries launch metadata. -/
def forward (T D BD : Nat) (USE_INITIAL_STATE STORE_FINAL_STATE : Bool) : ComputeKernel :=
  let x : RegionName := "x"
  let g : RegionName := "g"
  let o : RegionName := "o"
  let h0 : RegionName := "h0"
  let ht : RegionName := "ht"
  with_triton_signature forwardSignature in
  triton {
    i_d, i_bh = tl.program_id(0), tl.program_id(1)
    o_d = i_d * BD + tl.arange(0, BD)
    mask = o_d < D

    p_x = x + i_bh * T * D + o_d
    p_g = g + i_bh * T * D + o_d
    p_o = o + i_bh * T * D + o_d

    b_h = tl.zeros([BD], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_bh * D + o_d
        b_h += tl.load(p_h0, mask=mask, other=0).to(tl.float32)
    for _ in range(0, T):
        b_x = tl.load(p_x, mask=mask, other=0).to(tl.float32)
        b_g = tl.load(p_g, mask=mask, other=0).to(tl.float32)
        b_h = b_g * b_h + b_x
        tl.store(p_o, b_h.to(p_o.dtype.element_ty), mask=mask)

        p_x += D
        p_g += D
        p_o += D

    if STORE_FINAL_STATE:
        p_ht = ht + i_bh * D + o_d
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask)
  }

/-- Raw body of `fused_recurrent_hgrn_stride_folded.py:fused_recurrent_hgrn_fwd_kernel`. The call signature carries launch metadata. -/
def forwardStrideFolded (T D BD : Nat) (USE_INITIAL_STATE STORE_FINAL_STATE : Bool) : ComputeKernel :=
  let x : RegionName := "x"
  let g : RegionName := "g"
  let o : RegionName := "o"
  let h0 : RegionName := "h0"
  let ht : RegionName := "ht"
  with_triton_signature forwardSignature in
  triton {
    i_d, i_bh = tl.program_id(0), tl.program_id(1)
    o_d = i_d * BD + tl.arange(0, BD)
    mask = o_d < D

    p_x = x + i_bh * (T * D) + o_d
    p_g = g + i_bh * (T * D) + o_d
    p_o = o + i_bh * (T * D) + o_d

    b_h = tl.zeros([BD], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_bh * D + o_d
        b_h += tl.load(p_h0, mask=mask, other=0).to(tl.float32)
    for _ in range(0, T):
        b_x = tl.load(p_x, mask=mask, other=0).to(tl.float32)
        b_g = tl.load(p_g, mask=mask, other=0).to(tl.float32)
        b_h = b_g * b_h + b_x
        tl.store(p_o, b_h.to(p_o.dtype.element_ty), mask=mask)

        p_x += D
        p_g += D
        p_o += D

    if STORE_FINAL_STATE:
        p_ht = ht + i_bh * D + o_d
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask)
  }

/-- Raw body of `fused_recurrent_hgrn_repaired.py:fused_recurrent_hgrn_bwd_kernel`. The call signature carries launch metadata. -/
def backward (T D BD : Nat) (USE_INITIAL_STATE USE_FINAL_GRAD : Bool) : ComputeKernel :=
  let g : RegionName := "g"
  let o : RegionName := "o"
  let dx : RegionName := "dx"
  let dg : RegionName := "dg"
  let «do» : RegionName := "do"
  let h0 : RegionName := "h0"
  let dht : RegionName := "dht"
  with_triton_signature backwardSignature in
  triton {
    i_d, i_bh = tl.program_id(0), tl.program_id(1)
    o_d = i_d * BD + tl.arange(0, BD)
    mask = o_d < D

    b_dh = tl.zeros([BD], dtype=tl.float32)
    if USE_FINAL_GRAD:
        b_dh = tl.load(dht + i_bh * D + o_d, mask=mask, other=0).to(tl.float32)
    for step in range(0, T):
        i = T - 1 - step
        p_g = g + (i_bh * T + i) * D + o_d
        p_dx = dx + (i_bh * T + i) * D + o_d
        p_dg = dg + (i_bh * T + i) * D + o_d
        p_do = do + (i_bh * T + i) * D + o_d
        b_g = tl.load(p_g, mask=mask, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask, other=0).to(tl.float32)
        if i > 0:
            b_o = tl.load(o + (i_bh * T + i - 1) * D + o_d, mask=mask, other=0).to(tl.float32)
        elif USE_INITIAL_STATE:
            b_o = tl.load(h0 + i_bh * D + o_d, mask=mask, other=0).to(tl.float32)
        else:
            b_o = tl.zeros([BD], dtype=tl.float32)

        b_dh = b_dh + b_do
        b_dx = b_dh
        b_dg = b_dh * b_o
        b_dh = b_dh * b_g
        tl.store(p_dx, b_dx.to(p_dx.dtype.element_ty), mask=mask)
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), mask=mask)
  }

end
end VeriTile.Bench.Repairs.FusedRecurrentHgrn
