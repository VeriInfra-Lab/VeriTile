import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Source

/-! Complete original Python bodies and existing row-index commutation
candidates. Caller types, constexpr values, ports and the libdevice import
binding live outside each quotation. -/
namespace VeriTile.Bench.Repairs.FastCeLoss.TypedSource
open VeriTile.Triton
set_option maxRecDepth 24000
set_option maxHeartbeats 4000000

def forwardSignature : CallSignature where
  pointers := [("logits_ptr", .fp32), ("loss_ptr", .fp32), ("logsumexp_ptr", .fp32), ("labels_ptr", .i64)]
  scalars := [("logits_row_stride", .integer)]
  constexpr := ["VOCAB_SIZE", "BLOCK_SIZE", "DO_SOFTCAPPING", "SOFTCAP", "DO_LOGIT_SCALING", "LOGIT_SCALE"]
  readPorts := ["logits_ptr", "labels_ptr"]
  writePorts := ["loss_ptr", "logsumexp_ptr"]
  unaryImports := [("triton_tanh", "triton.language.extra.libdevice.tanh")]

def forward (cap scale : Bool)
    (logits_ptr loss_ptr logsumexp_ptr : RegionName) (labels_ptr : Region .int)
    (VOCAB_SIZE logits_row_stride BLOCK_SIZE : Nat) (SOFTCAP LOGIT_SCALE : ℝ) : ComputeKernel :=
  let DO_SOFTCAPPING := cap
  let DO_LOGIT_SCALING := scale
  with_triton_call forwardSignature call_args [logits_row_stride := logits_row_stride] in triton {
    row_idx = tl.program_id(0).to(tl.int64)
    logits_ptr += row_idx * logits_row_stride.to(tl.int64)
    loss_ptr += row_idx
    logsumexp_ptr += row_idx
    labels_ptr += row_idx

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE

    label_idx = tl.load(labels_ptr).to(tl.int64)
    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))

    if DO_LOGIT_SCALING: logits = LOGIT_SCALE * logits
    if DO_SOFTCAPPING: logits = SOFTCAP * triton_tanh(logits / SOFTCAP)

    logits = tl.where(mask, logits.to(tl.float32), -float("inf"))
    c = tl.max(logits, 0)
    logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))

    if label_idx != -100:
        x = tl.load(logits_ptr + label_idx)
        if DO_LOGIT_SCALING: x = LOGIT_SCALE * x
        if DO_SOFTCAPPING: x = SOFTCAP * triton_tanh(x / SOFTCAP)
        loss = logsumexp - x.to(tl.float32)
    else:
        loss = 0.0
    tl.store(logsumexp_ptr, logsumexp)
    tl.store(loss_ptr, loss)
  }

def forwardIndexCommuted (cap scale : Bool)
    (logits_ptr loss_ptr logsumexp_ptr : RegionName) (labels_ptr : Region .int)
    (VOCAB_SIZE logits_row_stride BLOCK_SIZE : Nat) (SOFTCAP LOGIT_SCALE : ℝ) : ComputeKernel :=
  let DO_SOFTCAPPING := cap
  let DO_LOGIT_SCALING := scale
  with_triton_call forwardSignature call_args [logits_row_stride := logits_row_stride] in triton {
    row_idx = tl.program_id(0).to(tl.int64)
    logits_ptr += logits_row_stride.to(tl.int64) * row_idx
    loss_ptr += row_idx
    logsumexp_ptr += row_idx
    labels_ptr += row_idx

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE

    label_idx = tl.load(labels_ptr).to(tl.int64)
    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))

    if DO_LOGIT_SCALING: logits = LOGIT_SCALE * logits
    if DO_SOFTCAPPING: logits = SOFTCAP * triton_tanh(logits / SOFTCAP)

    logits = tl.where(mask, logits.to(tl.float32), -float("inf"))
    c = tl.max(logits, 0)
    logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))

    if label_idx != -100:
        x = tl.load(logits_ptr + label_idx)
        if DO_LOGIT_SCALING: x = LOGIT_SCALE * x
        if DO_SOFTCAPPING: x = SOFTCAP * triton_tanh(x / SOFTCAP)
        loss = logsumexp - x.to(tl.float32)
    else:
        loss = 0.0
    tl.store(logsumexp_ptr, logsumexp)
    tl.store(loss_ptr, loss)
  }

def chunkedSignature : CallSignature where
  pointers := [("logits_ptr", .fp32), ("loss_ptr", .fp32), ("logsumexp_ptr", .fp32), ("labels_ptr", .i64)]
  scalars := [("logits_row_stride", .integer)]
  constexpr := ["VOCAB_SIZE", "N_CHUNKS", "BLOCK_SIZE", "DO_SOFTCAPPING", "SOFTCAP", "DO_LOGIT_SCALING", "LOGIT_SCALE"]
  readPorts := ["logits_ptr", "labels_ptr"]
  writePorts := ["loss_ptr", "logsumexp_ptr"]
  unaryImports := [("triton_tanh", "triton.language.extra.libdevice.tanh")]

def chunked (cap scale : Bool)
    (logits_ptr loss_ptr logsumexp_ptr : RegionName) (labels_ptr : Region .int)
    (VOCAB_SIZE N_CHUNKS logits_row_stride BLOCK_SIZE : Nat) (SOFTCAP LOGIT_SCALE : ℝ) : ComputeKernel :=
  let DO_SOFTCAPPING := cap
  let DO_LOGIT_SCALING := scale
  with_triton_call chunkedSignature call_args [logits_row_stride := logits_row_stride] in triton {
    row_idx = tl.program_id(0).to(tl.int64)
    chunk_idx = tl.program_id(1).to(tl.int64)
    logits_ptr += row_idx * logits_row_stride.to(tl.int64)
    loss_ptr += row_idx
    logsumexp_ptr += row_idx * N_CHUNKS + chunk_idx
    labels_ptr += row_idx

    col_offsets = chunk_idx*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE

    label_idx = tl.load(labels_ptr).to(tl.int64)
    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))

    if DO_LOGIT_SCALING: logits = LOGIT_SCALE * logits
    if DO_SOFTCAPPING: logits = SOFTCAP * triton_tanh(logits / SOFTCAP)

    logits = tl.where(mask, logits.to(tl.float32), -float("inf"))
    c = tl.max(logits, 0)
    logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))

    if chunk_idx == 0:
        if label_idx != -100:
            x = tl.load(logits_ptr + label_idx).to(tl.float32)
            if DO_LOGIT_SCALING: x = LOGIT_SCALE * x
            if DO_SOFTCAPPING: x = SOFTCAP * triton_tanh(x / SOFTCAP)
            loss = -1.0 * x.to(tl.float32)
        else:
            loss = 0.0
        tl.store(loss_ptr, loss)
    tl.store(logsumexp_ptr, logsumexp)
  }

def chunkedIndexCommuted (cap scale : Bool)
    (logits_ptr loss_ptr logsumexp_ptr : RegionName) (labels_ptr : Region .int)
    (VOCAB_SIZE N_CHUNKS logits_row_stride BLOCK_SIZE : Nat) (SOFTCAP LOGIT_SCALE : ℝ) : ComputeKernel :=
  let DO_SOFTCAPPING := cap
  let DO_LOGIT_SCALING := scale
  with_triton_call chunkedSignature call_args [logits_row_stride := logits_row_stride] in triton {
    row_idx = tl.program_id(0).to(tl.int64)
    chunk_idx = tl.program_id(1).to(tl.int64)
    logits_ptr += logits_row_stride.to(tl.int64) * row_idx
    loss_ptr += row_idx
    logsumexp_ptr += row_idx * N_CHUNKS + chunk_idx
    labels_ptr += row_idx

    col_offsets = chunk_idx*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE

    label_idx = tl.load(labels_ptr).to(tl.int64)
    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))

    if DO_LOGIT_SCALING: logits = LOGIT_SCALE * logits
    if DO_SOFTCAPPING: logits = SOFTCAP * triton_tanh(logits / SOFTCAP)

    logits = tl.where(mask, logits.to(tl.float32), -float("inf"))
    c = tl.max(logits, 0)
    logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))

    if chunk_idx == 0:
        if label_idx != -100:
            x = tl.load(logits_ptr + label_idx).to(tl.float32)
            if DO_LOGIT_SCALING: x = LOGIT_SCALE * x
            if DO_SOFTCAPPING: x = SOFTCAP * triton_tanh(x / SOFTCAP)
            loss = -1.0 * x.to(tl.float32)
        else:
            loss = 0.0
        tl.store(loss_ptr, loss)
    tl.store(logsumexp_ptr, logsumexp)
  }

def backwardSignature : CallSignature where
  pointers := [("logits_ptr", .fp32), ("dloss_ptr", .fp32), ("logsumexp_ptr", .fp32), ("labels_ptr", .i64)]
  scalars := [("logits_row_stride", .integer), ("dloss_row_stride", .integer)]
  constexpr := ["VOCAB_SIZE", "BLOCK_SIZE", "DO_SOFTCAPPING", "SOFTCAP", "DO_LOGIT_SCALING", "LOGIT_SCALE"]
  readPorts := ["logits_ptr", "labels_ptr", "dloss_ptr", "logsumexp_ptr"]
  writePorts := ["logits_ptr"]
  unaryImports := [("triton_tanh", "triton.language.extra.libdevice.tanh")]

def backward (cap scale : Bool)
    (logits_ptr dloss_ptr logsumexp_ptr : RegionName) (labels_ptr : Region .int)
    (VOCAB_SIZE logits_row_stride dloss_row_stride BLOCK_SIZE : Nat) (SOFTCAP LOGIT_SCALE : ℝ) : ComputeKernel :=
  let DO_SOFTCAPPING := cap
  let DO_LOGIT_SCALING := scale
  with_triton_call backwardSignature call_args [logits_row_stride := logits_row_stride, dloss_row_stride := dloss_row_stride] in triton {
    row_idx = tl.program_id(0).to(tl.int64)
    block_idx = tl.program_id(1).to(tl.int64)

    logits_ptr += row_idx * logits_row_stride.to(tl.int64)
    dloss_ptr += row_idx * dloss_row_stride
    col_offsets = block_idx*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE
    label_idx = tl.load(labels_ptr + row_idx).to(tl.int64)

    if label_idx != -100:
        dloss = tl.load(dloss_ptr)
    else:
        dloss = 0.0

    x = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))

    if DO_LOGIT_SCALING:
        x = x * LOGIT_SCALE

    if DO_SOFTCAPPING:
        partial = triton_tanh(x / SOFTCAP)
        x = SOFTCAP * partial

    logsumexp = tl.load(logsumexp_ptr + row_idx)
    y = tl.exp(x.to(tl.float32) - logsumexp)
    y = tl.where(
        col_offsets == label_idx,
        y - 1.0,
        y,
    )

    if DO_LOGIT_SCALING:
        y = y * LOGIT_SCALE

    if DO_SOFTCAPPING:
        y = y * (1.0 - partial*partial)

    tl.store(logits_ptr + col_offsets, tl.where(label_idx == -100, 0.0, dloss * y), mask=mask)
  }

def backwardIndexCommuted (cap scale : Bool)
    (logits_ptr dloss_ptr logsumexp_ptr : RegionName) (labels_ptr : Region .int)
    (VOCAB_SIZE logits_row_stride dloss_row_stride BLOCK_SIZE : Nat) (SOFTCAP LOGIT_SCALE : ℝ) : ComputeKernel :=
  let DO_SOFTCAPPING := cap
  let DO_LOGIT_SCALING := scale
  with_triton_call backwardSignature call_args [logits_row_stride := logits_row_stride, dloss_row_stride := dloss_row_stride] in triton {
    row_idx = tl.program_id(0).to(tl.int64)
    block_idx = tl.program_id(1).to(tl.int64)

    logits_ptr += logits_row_stride.to(tl.int64) * row_idx
    dloss_ptr += row_idx * dloss_row_stride
    col_offsets = block_idx*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE
    label_idx = tl.load(labels_ptr + row_idx).to(tl.int64)

    if label_idx != -100:
        dloss = tl.load(dloss_ptr)
    else:
        dloss = 0.0

    x = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))

    if DO_LOGIT_SCALING:
        x = x * LOGIT_SCALE

    if DO_SOFTCAPPING:
        partial = triton_tanh(x / SOFTCAP)
        x = SOFTCAP * partial

    logsumexp = tl.load(logsumexp_ptr + row_idx)
    y = tl.exp(x.to(tl.float32) - logsumexp)
    y = tl.where(
        col_offsets == label_idx,
        y - 1.0,
        y,
    )

    if DO_LOGIT_SCALING:
        y = y * LOGIT_SCALE

    if DO_SOFTCAPPING:
        y = y * (1.0 - partial*partial)

    tl.store(logits_ptr + col_offsets, tl.where(label_idx == -100, 0.0, dloss * y), mask=mask)
  }

end VeriTile.Bench.Repairs.FastCeLoss.TypedSource
