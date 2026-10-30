import VeriTile.Triton.DSL
import VeriTile.Triton.Source

/-! Complete repaired FP32/int64 source and the existing row-index commutation
candidates. Each constexpr specialization is an ordinary quotation, preserving
source precision, casts, order, libdevice tanh, masks and stores. Explicit ports
retain the original input/output interface. Host programs remain in
Proofs/Source. -/
namespace VeriTile.Bench.Repairs.FastCeLoss.TypedSource
open VeriTile.Triton

/-- Repaired source; all four constexpr flag combinations. -/
def forward (cap scale : Bool) := match cap, scale with
  | Bool.false, Bool.false =>
    fun (logits_ptr loss_ptr logsumexp_ptr : RegionName) (labels_ptr : Region .int)
        (VOCAB_SIZE logits_row_stride BLOCK_SIZE : Nat) (SOFTCAP LOGIT_SCALE : ℝ) =>
      triton {
        tl.ports inputs=[logits_ptr, Region.cast labels_ptr], outputs=[loss_ptr, logsumexp_ptr]
        tl.region labels_ptr = tl.int64, logits_ptr = tl.float32, loss_ptr = tl.float32, logsumexp_ptr = tl.float32
        row_idx = tl.program_id(0).to(tl.int64)
        logits_ptr += row_idx * ($(logits_row_stride)).to(tl.int64)
        loss_ptr += row_idx
        logsumexp_ptr += row_idx
        labels_ptr += row_idx
        col_offsets = tl.arange(0, $(BLOCK_SIZE))
        mask = col_offsets < $(VOCAB_SIZE)
        label_idx = tl.load(labels_ptr).to(tl.int64)
        logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))
        logits = tl.where(mask, logits.to(tl.float32), -float("inf"))
        c = tl.max(logits, 0)
        logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))
        if label_idx != $((-100 : Int)) {
          x = tl.load(logits_ptr + label_idx)
          loss = logsumexp - x.to(tl.float32)
        } else {
          loss = 0.0
        }
        tl.store(logsumexp_ptr, logsumexp.to(tl.float32))
        tl.store(loss_ptr, loss.to(tl.float32))
      }
  | Bool.false, Bool.true =>
    fun (logits_ptr loss_ptr logsumexp_ptr : RegionName) (labels_ptr : Region .int)
        (VOCAB_SIZE logits_row_stride BLOCK_SIZE : Nat) (SOFTCAP LOGIT_SCALE : ℝ) =>
      triton {
        tl.ports inputs=[logits_ptr, Region.cast labels_ptr], outputs=[loss_ptr, logsumexp_ptr]
        tl.region labels_ptr = tl.int64, logits_ptr = tl.float32, loss_ptr = tl.float32, logsumexp_ptr = tl.float32
        row_idx = tl.program_id(0).to(tl.int64)
        logits_ptr += row_idx * ($(logits_row_stride)).to(tl.int64)
        loss_ptr += row_idx
        logsumexp_ptr += row_idx
        labels_ptr += row_idx
        col_offsets = tl.arange(0, $(BLOCK_SIZE))
        mask = col_offsets < $(VOCAB_SIZE)
        label_idx = tl.load(labels_ptr).to(tl.int64)
        logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))
        logits = $((LOGIT_SCALE : ℝ)) * logits
        logits = tl.where(mask, logits.to(tl.float32), -float("inf"))
        c = tl.max(logits, 0)
        logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))
        if label_idx != $((-100 : Int)) {
          x = tl.load(logits_ptr + label_idx)
          x = $((LOGIT_SCALE : ℝ)) * x
          loss = logsumexp - x.to(tl.float32)
        } else {
          loss = 0.0
        }
        tl.store(logsumexp_ptr, logsumexp.to(tl.float32))
        tl.store(loss_ptr, loss.to(tl.float32))
      }
  | Bool.true, Bool.false =>
    fun (logits_ptr loss_ptr logsumexp_ptr : RegionName) (labels_ptr : Region .int)
        (VOCAB_SIZE logits_row_stride BLOCK_SIZE : Nat) (SOFTCAP LOGIT_SCALE : ℝ) =>
      triton {
        tl.ports inputs=[logits_ptr, Region.cast labels_ptr], outputs=[loss_ptr, logsumexp_ptr]
        tl.region labels_ptr = tl.int64, logits_ptr = tl.float32, loss_ptr = tl.float32, logsumexp_ptr = tl.float32
        row_idx = tl.program_id(0).to(tl.int64)
        logits_ptr += row_idx * ($(logits_row_stride)).to(tl.int64)
        loss_ptr += row_idx
        logsumexp_ptr += row_idx
        labels_ptr += row_idx
        col_offsets = tl.arange(0, $(BLOCK_SIZE))
        mask = col_offsets < $(VOCAB_SIZE)
        label_idx = tl.load(labels_ptr).to(tl.int64)
        logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))
        logits = $((SOFTCAP : ℝ)) * triton_tanh(logits / $((SOFTCAP : ℝ)))
        logits = tl.where(mask, logits.to(tl.float32), -float("inf"))
        c = tl.max(logits, 0)
        logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))
        if label_idx != $((-100 : Int)) {
          x = tl.load(logits_ptr + label_idx)
          x = $((SOFTCAP : ℝ)) * triton_tanh(x / $((SOFTCAP : ℝ)))
          loss = logsumexp - x.to(tl.float32)
        } else {
          loss = 0.0
        }
        tl.store(logsumexp_ptr, logsumexp.to(tl.float32))
        tl.store(loss_ptr, loss.to(tl.float32))
      }
  | Bool.true, Bool.true =>
    fun (logits_ptr loss_ptr logsumexp_ptr : RegionName) (labels_ptr : Region .int)
        (VOCAB_SIZE logits_row_stride BLOCK_SIZE : Nat) (SOFTCAP LOGIT_SCALE : ℝ) =>
      triton {
        tl.ports inputs=[logits_ptr, Region.cast labels_ptr], outputs=[loss_ptr, logsumexp_ptr]
        tl.region labels_ptr = tl.int64, logits_ptr = tl.float32, loss_ptr = tl.float32, logsumexp_ptr = tl.float32
        row_idx = tl.program_id(0).to(tl.int64)
        logits_ptr += row_idx * ($(logits_row_stride)).to(tl.int64)
        loss_ptr += row_idx
        logsumexp_ptr += row_idx
        labels_ptr += row_idx
        col_offsets = tl.arange(0, $(BLOCK_SIZE))
        mask = col_offsets < $(VOCAB_SIZE)
        label_idx = tl.load(labels_ptr).to(tl.int64)
        logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))
        logits = $((LOGIT_SCALE : ℝ)) * logits
        logits = $((SOFTCAP : ℝ)) * triton_tanh(logits / $((SOFTCAP : ℝ)))
        logits = tl.where(mask, logits.to(tl.float32), -float("inf"))
        c = tl.max(logits, 0)
        logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))
        if label_idx != $((-100 : Int)) {
          x = tl.load(logits_ptr + label_idx)
          x = $((LOGIT_SCALE : ℝ)) * x
          x = $((SOFTCAP : ℝ)) * triton_tanh(x / $((SOFTCAP : ℝ)))
          loss = logsumexp - x.to(tl.float32)
        } else {
          loss = 0.0
        }
        tl.store(logsumexp_ptr, logsumexp.to(tl.float32))
        tl.store(loss_ptr, loss.to(tl.float32))
      }

/-- Repaired source; all four constexpr flag combinations. -/
def chunked (cap scale : Bool) := match cap, scale with
  | Bool.false, Bool.false =>
    fun (logits_ptr loss_ptr logsumexp_ptr : RegionName) (labels_ptr : Region .int)
        (VOCAB_SIZE N_CHUNKS logits_row_stride BLOCK_SIZE : Nat) (SOFTCAP LOGIT_SCALE : ℝ) =>
      triton {
        tl.ports inputs=[logits_ptr, Region.cast labels_ptr], outputs=[loss_ptr, logsumexp_ptr]
        tl.region labels_ptr = tl.int64, logits_ptr = tl.float32, loss_ptr = tl.float32, logsumexp_ptr = tl.float32
        row_idx = tl.program_id(0).to(tl.int64)
        chunk_idx = tl.program_id(1).to(tl.int64)
        logits_ptr += row_idx * ($(logits_row_stride)).to(tl.int64)
        loss_ptr += row_idx
        logsumexp_ptr += row_idx * $(N_CHUNKS) + chunk_idx
        labels_ptr += row_idx
        col_offsets = chunk_idx * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
        mask = col_offsets < $(VOCAB_SIZE)
        label_idx = tl.load(labels_ptr).to(tl.int64)
        logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))
        logits = tl.where(mask, logits.to(tl.float32), -float("inf"))
        c = tl.max(logits, 0)
        logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))
        if chunk_idx == $(0) {
          if label_idx != $((-100 : Int)) {
            x = tl.load(logits_ptr + label_idx).to(tl.float32)
            loss = -1.0 * x.to(tl.float32)
          } else {
            loss = 0.0
          }
          tl.store(loss_ptr, loss.to(tl.float32))
        }
        tl.store(logsumexp_ptr, logsumexp.to(tl.float32))
      }
  | Bool.false, Bool.true =>
    fun (logits_ptr loss_ptr logsumexp_ptr : RegionName) (labels_ptr : Region .int)
        (VOCAB_SIZE N_CHUNKS logits_row_stride BLOCK_SIZE : Nat) (SOFTCAP LOGIT_SCALE : ℝ) =>
      triton {
        tl.ports inputs=[logits_ptr, Region.cast labels_ptr], outputs=[loss_ptr, logsumexp_ptr]
        tl.region labels_ptr = tl.int64, logits_ptr = tl.float32, loss_ptr = tl.float32, logsumexp_ptr = tl.float32
        row_idx = tl.program_id(0).to(tl.int64)
        chunk_idx = tl.program_id(1).to(tl.int64)
        logits_ptr += row_idx * ($(logits_row_stride)).to(tl.int64)
        loss_ptr += row_idx
        logsumexp_ptr += row_idx * $(N_CHUNKS) + chunk_idx
        labels_ptr += row_idx
        col_offsets = chunk_idx * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
        mask = col_offsets < $(VOCAB_SIZE)
        label_idx = tl.load(labels_ptr).to(tl.int64)
        logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))
        logits = $((LOGIT_SCALE : ℝ)) * logits
        logits = tl.where(mask, logits.to(tl.float32), -float("inf"))
        c = tl.max(logits, 0)
        logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))
        if chunk_idx == $(0) {
          if label_idx != $((-100 : Int)) {
            x = tl.load(logits_ptr + label_idx).to(tl.float32)
            x = $((LOGIT_SCALE : ℝ)) * x
            loss = -1.0 * x.to(tl.float32)
          } else {
            loss = 0.0
          }
          tl.store(loss_ptr, loss.to(tl.float32))
        }
        tl.store(logsumexp_ptr, logsumexp.to(tl.float32))
      }
  | Bool.true, Bool.false =>
    fun (logits_ptr loss_ptr logsumexp_ptr : RegionName) (labels_ptr : Region .int)
        (VOCAB_SIZE N_CHUNKS logits_row_stride BLOCK_SIZE : Nat) (SOFTCAP LOGIT_SCALE : ℝ) =>
      triton {
        tl.ports inputs=[logits_ptr, Region.cast labels_ptr], outputs=[loss_ptr, logsumexp_ptr]
        tl.region labels_ptr = tl.int64, logits_ptr = tl.float32, loss_ptr = tl.float32, logsumexp_ptr = tl.float32
        row_idx = tl.program_id(0).to(tl.int64)
        chunk_idx = tl.program_id(1).to(tl.int64)
        logits_ptr += row_idx * ($(logits_row_stride)).to(tl.int64)
        loss_ptr += row_idx
        logsumexp_ptr += row_idx * $(N_CHUNKS) + chunk_idx
        labels_ptr += row_idx
        col_offsets = chunk_idx * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
        mask = col_offsets < $(VOCAB_SIZE)
        label_idx = tl.load(labels_ptr).to(tl.int64)
        logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))
        logits = $((SOFTCAP : ℝ)) * triton_tanh(logits / $((SOFTCAP : ℝ)))
        logits = tl.where(mask, logits.to(tl.float32), -float("inf"))
        c = tl.max(logits, 0)
        logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))
        if chunk_idx == $(0) {
          if label_idx != $((-100 : Int)) {
            x = tl.load(logits_ptr + label_idx).to(tl.float32)
            x = $((SOFTCAP : ℝ)) * triton_tanh(x / $((SOFTCAP : ℝ)))
            loss = -1.0 * x.to(tl.float32)
          } else {
            loss = 0.0
          }
          tl.store(loss_ptr, loss.to(tl.float32))
        }
        tl.store(logsumexp_ptr, logsumexp.to(tl.float32))
      }
  | Bool.true, Bool.true =>
    fun (logits_ptr loss_ptr logsumexp_ptr : RegionName) (labels_ptr : Region .int)
        (VOCAB_SIZE N_CHUNKS logits_row_stride BLOCK_SIZE : Nat) (SOFTCAP LOGIT_SCALE : ℝ) =>
      triton {
        tl.ports inputs=[logits_ptr, Region.cast labels_ptr], outputs=[loss_ptr, logsumexp_ptr]
        tl.region labels_ptr = tl.int64, logits_ptr = tl.float32, loss_ptr = tl.float32, logsumexp_ptr = tl.float32
        row_idx = tl.program_id(0).to(tl.int64)
        chunk_idx = tl.program_id(1).to(tl.int64)
        logits_ptr += row_idx * ($(logits_row_stride)).to(tl.int64)
        loss_ptr += row_idx
        logsumexp_ptr += row_idx * $(N_CHUNKS) + chunk_idx
        labels_ptr += row_idx
        col_offsets = chunk_idx * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
        mask = col_offsets < $(VOCAB_SIZE)
        label_idx = tl.load(labels_ptr).to(tl.int64)
        logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))
        logits = $((LOGIT_SCALE : ℝ)) * logits
        logits = $((SOFTCAP : ℝ)) * triton_tanh(logits / $((SOFTCAP : ℝ)))
        logits = tl.where(mask, logits.to(tl.float32), -float("inf"))
        c = tl.max(logits, 0)
        logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))
        if chunk_idx == $(0) {
          if label_idx != $((-100 : Int)) {
            x = tl.load(logits_ptr + label_idx).to(tl.float32)
            x = $((LOGIT_SCALE : ℝ)) * x
            x = $((SOFTCAP : ℝ)) * triton_tanh(x / $((SOFTCAP : ℝ)))
            loss = -1.0 * x.to(tl.float32)
          } else {
            loss = 0.0
          }
          tl.store(loss_ptr, loss.to(tl.float32))
        }
        tl.store(logsumexp_ptr, logsumexp.to(tl.float32))
      }

/-- Repaired source; all four constexpr flag combinations. -/
def backward (cap scale : Bool) := match cap, scale with
  | Bool.false, Bool.false =>
    fun (logits_ptr dloss_ptr logsumexp_ptr : RegionName) (labels_ptr : Region .int)
        (VOCAB_SIZE logits_row_stride dloss_row_stride BLOCK_SIZE : Nat) (SOFTCAP LOGIT_SCALE : ℝ) =>
      triton {
        tl.ports inputs=[logits_ptr, dloss_ptr, logsumexp_ptr, Region.cast labels_ptr], outputs=[logits_ptr]
        tl.region labels_ptr = tl.int64, logits_ptr = tl.float32, dloss_ptr = tl.float32, logsumexp_ptr = tl.float32
        row_idx = tl.program_id(0).to(tl.int64)
        block_idx = tl.program_id(1).to(tl.int64)
        logits_ptr += row_idx * ($(logits_row_stride)).to(tl.int64)
        dloss_ptr += row_idx * $(dloss_row_stride)
        col_offsets = block_idx * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
        mask = col_offsets < $(VOCAB_SIZE)
        label_idx = tl.load(labels_ptr + row_idx).to(tl.int64)
        if label_idx != $((-100 : Int)) {
          dloss = tl.load(dloss_ptr)
        }
        else {
          dloss = 0.0
        }
        x = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))
        logsumexp = tl.load(logsumexp_ptr + row_idx)
        y = tl.exp(x.to(tl.float32) - logsumexp)
        y = tl.where(col_offsets == label_idx, y - 1.0, y)
        tl.store(logits_ptr + col_offsets, tl.where(label_idx == $((-100 : Int)), 0.0, dloss * y).to(tl.float32), mask=mask)
      }
  | Bool.false, Bool.true =>
    fun (logits_ptr dloss_ptr logsumexp_ptr : RegionName) (labels_ptr : Region .int)
        (VOCAB_SIZE logits_row_stride dloss_row_stride BLOCK_SIZE : Nat) (SOFTCAP LOGIT_SCALE : ℝ) =>
      triton {
        tl.ports inputs=[logits_ptr, dloss_ptr, logsumexp_ptr, Region.cast labels_ptr], outputs=[logits_ptr]
        tl.region labels_ptr = tl.int64, logits_ptr = tl.float32, dloss_ptr = tl.float32, logsumexp_ptr = tl.float32
        row_idx = tl.program_id(0).to(tl.int64)
        block_idx = tl.program_id(1).to(tl.int64)
        logits_ptr += row_idx * ($(logits_row_stride)).to(tl.int64)
        dloss_ptr += row_idx * $(dloss_row_stride)
        col_offsets = block_idx * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
        mask = col_offsets < $(VOCAB_SIZE)
        label_idx = tl.load(labels_ptr + row_idx).to(tl.int64)
        if label_idx != $((-100 : Int)) {
          dloss = tl.load(dloss_ptr)
        }
        else {
          dloss = 0.0
        }
        x = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))
        x = x * $((LOGIT_SCALE : ℝ))
        logsumexp = tl.load(logsumexp_ptr + row_idx)
        y = tl.exp(x.to(tl.float32) - logsumexp)
        y = tl.where(col_offsets == label_idx, y - 1.0, y)
        y = y * $((LOGIT_SCALE : ℝ))
        tl.store(logits_ptr + col_offsets, tl.where(label_idx == $((-100 : Int)), 0.0, dloss * y).to(tl.float32), mask=mask)
      }
  | Bool.true, Bool.false =>
    fun (logits_ptr dloss_ptr logsumexp_ptr : RegionName) (labels_ptr : Region .int)
        (VOCAB_SIZE logits_row_stride dloss_row_stride BLOCK_SIZE : Nat) (SOFTCAP LOGIT_SCALE : ℝ) =>
      triton {
        tl.ports inputs=[logits_ptr, dloss_ptr, logsumexp_ptr, Region.cast labels_ptr], outputs=[logits_ptr]
        tl.region labels_ptr = tl.int64, logits_ptr = tl.float32, dloss_ptr = tl.float32, logsumexp_ptr = tl.float32
        row_idx = tl.program_id(0).to(tl.int64)
        block_idx = tl.program_id(1).to(tl.int64)
        logits_ptr += row_idx * ($(logits_row_stride)).to(tl.int64)
        dloss_ptr += row_idx * $(dloss_row_stride)
        col_offsets = block_idx * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
        mask = col_offsets < $(VOCAB_SIZE)
        label_idx = tl.load(labels_ptr + row_idx).to(tl.int64)
        if label_idx != $((-100 : Int)) {
          dloss = tl.load(dloss_ptr)
        }
        else {
          dloss = 0.0
        }
        x = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))
        «partial» = triton_tanh(x / $((SOFTCAP : ℝ)))
        x = $((SOFTCAP : ℝ)) * «partial»
        logsumexp = tl.load(logsumexp_ptr + row_idx)
        y = tl.exp(x.to(tl.float32) - logsumexp)
        y = tl.where(col_offsets == label_idx, y - 1.0, y)
        y = y * (1.0 - «partial» * «partial»)
        tl.store(logits_ptr + col_offsets, tl.where(label_idx == $((-100 : Int)), 0.0, dloss * y).to(tl.float32), mask=mask)
      }
  | Bool.true, Bool.true =>
    fun (logits_ptr dloss_ptr logsumexp_ptr : RegionName) (labels_ptr : Region .int)
        (VOCAB_SIZE logits_row_stride dloss_row_stride BLOCK_SIZE : Nat) (SOFTCAP LOGIT_SCALE : ℝ) =>
      triton {
        tl.ports inputs=[logits_ptr, dloss_ptr, logsumexp_ptr, Region.cast labels_ptr], outputs=[logits_ptr]
        tl.region labels_ptr = tl.int64, logits_ptr = tl.float32, dloss_ptr = tl.float32, logsumexp_ptr = tl.float32
        row_idx = tl.program_id(0).to(tl.int64)
        block_idx = tl.program_id(1).to(tl.int64)
        logits_ptr += row_idx * ($(logits_row_stride)).to(tl.int64)
        dloss_ptr += row_idx * $(dloss_row_stride)
        col_offsets = block_idx * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
        mask = col_offsets < $(VOCAB_SIZE)
        label_idx = tl.load(labels_ptr + row_idx).to(tl.int64)
        if label_idx != $((-100 : Int)) {
          dloss = tl.load(dloss_ptr)
        }
        else {
          dloss = 0.0
        }
        x = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))
        x = x * $((LOGIT_SCALE : ℝ))
        «partial» = triton_tanh(x / $((SOFTCAP : ℝ)))
        x = $((SOFTCAP : ℝ)) * «partial»
        logsumexp = tl.load(logsumexp_ptr + row_idx)
        y = tl.exp(x.to(tl.float32) - logsumexp)
        y = tl.where(col_offsets == label_idx, y - 1.0, y)
        y = y * $((LOGIT_SCALE : ℝ))
        y = y * (1.0 - «partial» * «partial»)
        tl.store(logits_ptr + col_offsets, tl.where(label_idx == $((-100 : Int)), 0.0, dloss * y).to(tl.float32), mask=mask)
      }

/-- Existing signed row-index commutation candidate; all four constexpr flag combinations. -/
def forwardIndexCommuted (cap scale : Bool) := match cap, scale with
  | Bool.false, Bool.false =>
    fun (logits_ptr loss_ptr logsumexp_ptr : RegionName) (labels_ptr : Region .int)
        (VOCAB_SIZE logits_row_stride BLOCK_SIZE : Nat) (SOFTCAP LOGIT_SCALE : ℝ) =>
      triton {
        tl.ports inputs=[logits_ptr, Region.cast labels_ptr], outputs=[loss_ptr, logsumexp_ptr]
        tl.region labels_ptr = tl.int64, logits_ptr = tl.float32, loss_ptr = tl.float32, logsumexp_ptr = tl.float32
        row_idx = tl.program_id(0).to(tl.int64)
        logits_ptr += ($(logits_row_stride)).to(tl.int64) * row_idx
        loss_ptr += row_idx
        logsumexp_ptr += row_idx
        labels_ptr += row_idx
        col_offsets = tl.arange(0, $(BLOCK_SIZE))
        mask = col_offsets < $(VOCAB_SIZE)
        label_idx = tl.load(labels_ptr).to(tl.int64)
        logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))
        logits = tl.where(mask, logits.to(tl.float32), -float("inf"))
        c = tl.max(logits, 0)
        logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))
        if label_idx != $((-100 : Int)) {
          x = tl.load(logits_ptr + label_idx)
          loss = logsumexp - x.to(tl.float32)
        } else {
          loss = 0.0
        }
        tl.store(logsumexp_ptr, logsumexp.to(tl.float32))
        tl.store(loss_ptr, loss.to(tl.float32))
      }
  | Bool.false, Bool.true =>
    fun (logits_ptr loss_ptr logsumexp_ptr : RegionName) (labels_ptr : Region .int)
        (VOCAB_SIZE logits_row_stride BLOCK_SIZE : Nat) (SOFTCAP LOGIT_SCALE : ℝ) =>
      triton {
        tl.ports inputs=[logits_ptr, Region.cast labels_ptr], outputs=[loss_ptr, logsumexp_ptr]
        tl.region labels_ptr = tl.int64, logits_ptr = tl.float32, loss_ptr = tl.float32, logsumexp_ptr = tl.float32
        row_idx = tl.program_id(0).to(tl.int64)
        logits_ptr += ($(logits_row_stride)).to(tl.int64) * row_idx
        loss_ptr += row_idx
        logsumexp_ptr += row_idx
        labels_ptr += row_idx
        col_offsets = tl.arange(0, $(BLOCK_SIZE))
        mask = col_offsets < $(VOCAB_SIZE)
        label_idx = tl.load(labels_ptr).to(tl.int64)
        logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))
        logits = $((LOGIT_SCALE : ℝ)) * logits
        logits = tl.where(mask, logits.to(tl.float32), -float("inf"))
        c = tl.max(logits, 0)
        logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))
        if label_idx != $((-100 : Int)) {
          x = tl.load(logits_ptr + label_idx)
          x = $((LOGIT_SCALE : ℝ)) * x
          loss = logsumexp - x.to(tl.float32)
        } else {
          loss = 0.0
        }
        tl.store(logsumexp_ptr, logsumexp.to(tl.float32))
        tl.store(loss_ptr, loss.to(tl.float32))
      }
  | Bool.true, Bool.false =>
    fun (logits_ptr loss_ptr logsumexp_ptr : RegionName) (labels_ptr : Region .int)
        (VOCAB_SIZE logits_row_stride BLOCK_SIZE : Nat) (SOFTCAP LOGIT_SCALE : ℝ) =>
      triton {
        tl.ports inputs=[logits_ptr, Region.cast labels_ptr], outputs=[loss_ptr, logsumexp_ptr]
        tl.region labels_ptr = tl.int64, logits_ptr = tl.float32, loss_ptr = tl.float32, logsumexp_ptr = tl.float32
        row_idx = tl.program_id(0).to(tl.int64)
        logits_ptr += ($(logits_row_stride)).to(tl.int64) * row_idx
        loss_ptr += row_idx
        logsumexp_ptr += row_idx
        labels_ptr += row_idx
        col_offsets = tl.arange(0, $(BLOCK_SIZE))
        mask = col_offsets < $(VOCAB_SIZE)
        label_idx = tl.load(labels_ptr).to(tl.int64)
        logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))
        logits = $((SOFTCAP : ℝ)) * triton_tanh(logits / $((SOFTCAP : ℝ)))
        logits = tl.where(mask, logits.to(tl.float32), -float("inf"))
        c = tl.max(logits, 0)
        logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))
        if label_idx != $((-100 : Int)) {
          x = tl.load(logits_ptr + label_idx)
          x = $((SOFTCAP : ℝ)) * triton_tanh(x / $((SOFTCAP : ℝ)))
          loss = logsumexp - x.to(tl.float32)
        } else {
          loss = 0.0
        }
        tl.store(logsumexp_ptr, logsumexp.to(tl.float32))
        tl.store(loss_ptr, loss.to(tl.float32))
      }
  | Bool.true, Bool.true =>
    fun (logits_ptr loss_ptr logsumexp_ptr : RegionName) (labels_ptr : Region .int)
        (VOCAB_SIZE logits_row_stride BLOCK_SIZE : Nat) (SOFTCAP LOGIT_SCALE : ℝ) =>
      triton {
        tl.ports inputs=[logits_ptr, Region.cast labels_ptr], outputs=[loss_ptr, logsumexp_ptr]
        tl.region labels_ptr = tl.int64, logits_ptr = tl.float32, loss_ptr = tl.float32, logsumexp_ptr = tl.float32
        row_idx = tl.program_id(0).to(tl.int64)
        logits_ptr += ($(logits_row_stride)).to(tl.int64) * row_idx
        loss_ptr += row_idx
        logsumexp_ptr += row_idx
        labels_ptr += row_idx
        col_offsets = tl.arange(0, $(BLOCK_SIZE))
        mask = col_offsets < $(VOCAB_SIZE)
        label_idx = tl.load(labels_ptr).to(tl.int64)
        logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))
        logits = $((LOGIT_SCALE : ℝ)) * logits
        logits = $((SOFTCAP : ℝ)) * triton_tanh(logits / $((SOFTCAP : ℝ)))
        logits = tl.where(mask, logits.to(tl.float32), -float("inf"))
        c = tl.max(logits, 0)
        logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))
        if label_idx != $((-100 : Int)) {
          x = tl.load(logits_ptr + label_idx)
          x = $((LOGIT_SCALE : ℝ)) * x
          x = $((SOFTCAP : ℝ)) * triton_tanh(x / $((SOFTCAP : ℝ)))
          loss = logsumexp - x.to(tl.float32)
        } else {
          loss = 0.0
        }
        tl.store(logsumexp_ptr, logsumexp.to(tl.float32))
        tl.store(loss_ptr, loss.to(tl.float32))
      }

/-- Existing signed row-index commutation candidate; all four constexpr flag combinations. -/
def chunkedIndexCommuted (cap scale : Bool) := match cap, scale with
  | Bool.false, Bool.false =>
    fun (logits_ptr loss_ptr logsumexp_ptr : RegionName) (labels_ptr : Region .int)
        (VOCAB_SIZE N_CHUNKS logits_row_stride BLOCK_SIZE : Nat) (SOFTCAP LOGIT_SCALE : ℝ) =>
      triton {
        tl.ports inputs=[logits_ptr, Region.cast labels_ptr], outputs=[loss_ptr, logsumexp_ptr]
        tl.region labels_ptr = tl.int64, logits_ptr = tl.float32, loss_ptr = tl.float32, logsumexp_ptr = tl.float32
        row_idx = tl.program_id(0).to(tl.int64)
        chunk_idx = tl.program_id(1).to(tl.int64)
        logits_ptr += ($(logits_row_stride)).to(tl.int64) * row_idx
        loss_ptr += row_idx
        logsumexp_ptr += row_idx * $(N_CHUNKS) + chunk_idx
        labels_ptr += row_idx
        col_offsets = chunk_idx * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
        mask = col_offsets < $(VOCAB_SIZE)
        label_idx = tl.load(labels_ptr).to(tl.int64)
        logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))
        logits = tl.where(mask, logits.to(tl.float32), -float("inf"))
        c = tl.max(logits, 0)
        logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))
        if chunk_idx == $(0) {
          if label_idx != $((-100 : Int)) {
            x = tl.load(logits_ptr + label_idx).to(tl.float32)
            loss = -1.0 * x.to(tl.float32)
          } else {
            loss = 0.0
          }
          tl.store(loss_ptr, loss.to(tl.float32))
        }
        tl.store(logsumexp_ptr, logsumexp.to(tl.float32))
      }
  | Bool.false, Bool.true =>
    fun (logits_ptr loss_ptr logsumexp_ptr : RegionName) (labels_ptr : Region .int)
        (VOCAB_SIZE N_CHUNKS logits_row_stride BLOCK_SIZE : Nat) (SOFTCAP LOGIT_SCALE : ℝ) =>
      triton {
        tl.ports inputs=[logits_ptr, Region.cast labels_ptr], outputs=[loss_ptr, logsumexp_ptr]
        tl.region labels_ptr = tl.int64, logits_ptr = tl.float32, loss_ptr = tl.float32, logsumexp_ptr = tl.float32
        row_idx = tl.program_id(0).to(tl.int64)
        chunk_idx = tl.program_id(1).to(tl.int64)
        logits_ptr += ($(logits_row_stride)).to(tl.int64) * row_idx
        loss_ptr += row_idx
        logsumexp_ptr += row_idx * $(N_CHUNKS) + chunk_idx
        labels_ptr += row_idx
        col_offsets = chunk_idx * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
        mask = col_offsets < $(VOCAB_SIZE)
        label_idx = tl.load(labels_ptr).to(tl.int64)
        logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))
        logits = $((LOGIT_SCALE : ℝ)) * logits
        logits = tl.where(mask, logits.to(tl.float32), -float("inf"))
        c = tl.max(logits, 0)
        logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))
        if chunk_idx == $(0) {
          if label_idx != $((-100 : Int)) {
            x = tl.load(logits_ptr + label_idx).to(tl.float32)
            x = $((LOGIT_SCALE : ℝ)) * x
            loss = -1.0 * x.to(tl.float32)
          } else {
            loss = 0.0
          }
          tl.store(loss_ptr, loss.to(tl.float32))
        }
        tl.store(logsumexp_ptr, logsumexp.to(tl.float32))
      }
  | Bool.true, Bool.false =>
    fun (logits_ptr loss_ptr logsumexp_ptr : RegionName) (labels_ptr : Region .int)
        (VOCAB_SIZE N_CHUNKS logits_row_stride BLOCK_SIZE : Nat) (SOFTCAP LOGIT_SCALE : ℝ) =>
      triton {
        tl.ports inputs=[logits_ptr, Region.cast labels_ptr], outputs=[loss_ptr, logsumexp_ptr]
        tl.region labels_ptr = tl.int64, logits_ptr = tl.float32, loss_ptr = tl.float32, logsumexp_ptr = tl.float32
        row_idx = tl.program_id(0).to(tl.int64)
        chunk_idx = tl.program_id(1).to(tl.int64)
        logits_ptr += ($(logits_row_stride)).to(tl.int64) * row_idx
        loss_ptr += row_idx
        logsumexp_ptr += row_idx * $(N_CHUNKS) + chunk_idx
        labels_ptr += row_idx
        col_offsets = chunk_idx * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
        mask = col_offsets < $(VOCAB_SIZE)
        label_idx = tl.load(labels_ptr).to(tl.int64)
        logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))
        logits = $((SOFTCAP : ℝ)) * triton_tanh(logits / $((SOFTCAP : ℝ)))
        logits = tl.where(mask, logits.to(tl.float32), -float("inf"))
        c = tl.max(logits, 0)
        logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))
        if chunk_idx == $(0) {
          if label_idx != $((-100 : Int)) {
            x = tl.load(logits_ptr + label_idx).to(tl.float32)
            x = $((SOFTCAP : ℝ)) * triton_tanh(x / $((SOFTCAP : ℝ)))
            loss = -1.0 * x.to(tl.float32)
          } else {
            loss = 0.0
          }
          tl.store(loss_ptr, loss.to(tl.float32))
        }
        tl.store(logsumexp_ptr, logsumexp.to(tl.float32))
      }
  | Bool.true, Bool.true =>
    fun (logits_ptr loss_ptr logsumexp_ptr : RegionName) (labels_ptr : Region .int)
        (VOCAB_SIZE N_CHUNKS logits_row_stride BLOCK_SIZE : Nat) (SOFTCAP LOGIT_SCALE : ℝ) =>
      triton {
        tl.ports inputs=[logits_ptr, Region.cast labels_ptr], outputs=[loss_ptr, logsumexp_ptr]
        tl.region labels_ptr = tl.int64, logits_ptr = tl.float32, loss_ptr = tl.float32, logsumexp_ptr = tl.float32
        row_idx = tl.program_id(0).to(tl.int64)
        chunk_idx = tl.program_id(1).to(tl.int64)
        logits_ptr += ($(logits_row_stride)).to(tl.int64) * row_idx
        loss_ptr += row_idx
        logsumexp_ptr += row_idx * $(N_CHUNKS) + chunk_idx
        labels_ptr += row_idx
        col_offsets = chunk_idx * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
        mask = col_offsets < $(VOCAB_SIZE)
        label_idx = tl.load(labels_ptr).to(tl.int64)
        logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))
        logits = $((LOGIT_SCALE : ℝ)) * logits
        logits = $((SOFTCAP : ℝ)) * triton_tanh(logits / $((SOFTCAP : ℝ)))
        logits = tl.where(mask, logits.to(tl.float32), -float("inf"))
        c = tl.max(logits, 0)
        logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))
        if chunk_idx == $(0) {
          if label_idx != $((-100 : Int)) {
            x = tl.load(logits_ptr + label_idx).to(tl.float32)
            x = $((LOGIT_SCALE : ℝ)) * x
            x = $((SOFTCAP : ℝ)) * triton_tanh(x / $((SOFTCAP : ℝ)))
            loss = -1.0 * x.to(tl.float32)
          } else {
            loss = 0.0
          }
          tl.store(loss_ptr, loss.to(tl.float32))
        }
        tl.store(logsumexp_ptr, logsumexp.to(tl.float32))
      }

/-- Existing signed row-index commutation candidate; all four constexpr flag combinations. -/
def backwardIndexCommuted (cap scale : Bool) := match cap, scale with
  | Bool.false, Bool.false =>
    fun (logits_ptr dloss_ptr logsumexp_ptr : RegionName) (labels_ptr : Region .int)
        (VOCAB_SIZE logits_row_stride dloss_row_stride BLOCK_SIZE : Nat) (SOFTCAP LOGIT_SCALE : ℝ) =>
      triton {
        tl.ports inputs=[logits_ptr, dloss_ptr, logsumexp_ptr, Region.cast labels_ptr], outputs=[logits_ptr]
        tl.region labels_ptr = tl.int64, logits_ptr = tl.float32, dloss_ptr = tl.float32, logsumexp_ptr = tl.float32
        row_idx = tl.program_id(0).to(tl.int64)
        block_idx = tl.program_id(1).to(tl.int64)
        logits_ptr += ($(logits_row_stride)).to(tl.int64) * row_idx
        dloss_ptr += row_idx * $(dloss_row_stride)
        col_offsets = block_idx * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
        mask = col_offsets < $(VOCAB_SIZE)
        label_idx = tl.load(labels_ptr + row_idx).to(tl.int64)
        if label_idx != $((-100 : Int)) {
          dloss = tl.load(dloss_ptr)
        }
        else {
          dloss = 0.0
        }
        x = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))
        logsumexp = tl.load(logsumexp_ptr + row_idx)
        y = tl.exp(x.to(tl.float32) - logsumexp)
        y = tl.where(col_offsets == label_idx, y - 1.0, y)
        tl.store(logits_ptr + col_offsets, tl.where(label_idx == $((-100 : Int)), 0.0, dloss * y).to(tl.float32), mask=mask)
      }
  | Bool.false, Bool.true =>
    fun (logits_ptr dloss_ptr logsumexp_ptr : RegionName) (labels_ptr : Region .int)
        (VOCAB_SIZE logits_row_stride dloss_row_stride BLOCK_SIZE : Nat) (SOFTCAP LOGIT_SCALE : ℝ) =>
      triton {
        tl.ports inputs=[logits_ptr, dloss_ptr, logsumexp_ptr, Region.cast labels_ptr], outputs=[logits_ptr]
        tl.region labels_ptr = tl.int64, logits_ptr = tl.float32, dloss_ptr = tl.float32, logsumexp_ptr = tl.float32
        row_idx = tl.program_id(0).to(tl.int64)
        block_idx = tl.program_id(1).to(tl.int64)
        logits_ptr += ($(logits_row_stride)).to(tl.int64) * row_idx
        dloss_ptr += row_idx * $(dloss_row_stride)
        col_offsets = block_idx * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
        mask = col_offsets < $(VOCAB_SIZE)
        label_idx = tl.load(labels_ptr + row_idx).to(tl.int64)
        if label_idx != $((-100 : Int)) {
          dloss = tl.load(dloss_ptr)
        }
        else {
          dloss = 0.0
        }
        x = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))
        x = x * $((LOGIT_SCALE : ℝ))
        logsumexp = tl.load(logsumexp_ptr + row_idx)
        y = tl.exp(x.to(tl.float32) - logsumexp)
        y = tl.where(col_offsets == label_idx, y - 1.0, y)
        y = y * $((LOGIT_SCALE : ℝ))
        tl.store(logits_ptr + col_offsets, tl.where(label_idx == $((-100 : Int)), 0.0, dloss * y).to(tl.float32), mask=mask)
      }
  | Bool.true, Bool.false =>
    fun (logits_ptr dloss_ptr logsumexp_ptr : RegionName) (labels_ptr : Region .int)
        (VOCAB_SIZE logits_row_stride dloss_row_stride BLOCK_SIZE : Nat) (SOFTCAP LOGIT_SCALE : ℝ) =>
      triton {
        tl.ports inputs=[logits_ptr, dloss_ptr, logsumexp_ptr, Region.cast labels_ptr], outputs=[logits_ptr]
        tl.region labels_ptr = tl.int64, logits_ptr = tl.float32, dloss_ptr = tl.float32, logsumexp_ptr = tl.float32
        row_idx = tl.program_id(0).to(tl.int64)
        block_idx = tl.program_id(1).to(tl.int64)
        logits_ptr += ($(logits_row_stride)).to(tl.int64) * row_idx
        dloss_ptr += row_idx * $(dloss_row_stride)
        col_offsets = block_idx * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
        mask = col_offsets < $(VOCAB_SIZE)
        label_idx = tl.load(labels_ptr + row_idx).to(tl.int64)
        if label_idx != $((-100 : Int)) {
          dloss = tl.load(dloss_ptr)
        }
        else {
          dloss = 0.0
        }
        x = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))
        «partial» = triton_tanh(x / $((SOFTCAP : ℝ)))
        x = $((SOFTCAP : ℝ)) * «partial»
        logsumexp = tl.load(logsumexp_ptr + row_idx)
        y = tl.exp(x.to(tl.float32) - logsumexp)
        y = tl.where(col_offsets == label_idx, y - 1.0, y)
        y = y * (1.0 - «partial» * «partial»)
        tl.store(logits_ptr + col_offsets, tl.where(label_idx == $((-100 : Int)), 0.0, dloss * y).to(tl.float32), mask=mask)
      }
  | Bool.true, Bool.true =>
    fun (logits_ptr dloss_ptr logsumexp_ptr : RegionName) (labels_ptr : Region .int)
        (VOCAB_SIZE logits_row_stride dloss_row_stride BLOCK_SIZE : Nat) (SOFTCAP LOGIT_SCALE : ℝ) =>
      triton {
        tl.ports inputs=[logits_ptr, dloss_ptr, logsumexp_ptr, Region.cast labels_ptr], outputs=[logits_ptr]
        tl.region labels_ptr = tl.int64, logits_ptr = tl.float32, dloss_ptr = tl.float32, logsumexp_ptr = tl.float32
        row_idx = tl.program_id(0).to(tl.int64)
        block_idx = tl.program_id(1).to(tl.int64)
        logits_ptr += ($(logits_row_stride)).to(tl.int64) * row_idx
        dloss_ptr += row_idx * $(dloss_row_stride)
        col_offsets = block_idx * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
        mask = col_offsets < $(VOCAB_SIZE)
        label_idx = tl.load(labels_ptr + row_idx).to(tl.int64)
        if label_idx != $((-100 : Int)) {
          dloss = tl.load(dloss_ptr)
        }
        else {
          dloss = 0.0
        }
        x = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))
        x = x * $((LOGIT_SCALE : ℝ))
        «partial» = triton_tanh(x / $((SOFTCAP : ℝ)))
        x = $((SOFTCAP : ℝ)) * «partial»
        logsumexp = tl.load(logsumexp_ptr + row_idx)
        y = tl.exp(x.to(tl.float32) - logsumexp)
        y = tl.where(col_offsets == label_idx, y - 1.0, y)
        y = y * $((LOGIT_SCALE : ℝ))
        y = y * (1.0 - «partial» * «partial»)
        tl.store(logits_ptr + col_offsets, tl.where(label_idx == $((-100 : Int)), 0.0, dloss * y).to(tl.float32), mask=mask)
      }

end VeriTile.Bench.Repairs.FastCeLoss.TypedSource
