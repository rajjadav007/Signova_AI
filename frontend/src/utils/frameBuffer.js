/**
 * Rolling Circular Frame Buffer & Sequence Validation Utility (FE-05)
 * Maintains a fixed-capacity FIFO queue of sampled video frames for temporal model input.
 */

export const DEFAULT_SEQUENCE_CAPACITY = 64;

export const FRAME_SEQUENCE_STATUS = {
  EMPTY: "EMPTY",
  COLLECTING: "COLLECTING",
  READY: "READY",
  PAUSED: "PAUSED",
  CAMERA_STOPPED: "CAMERA_STOPPED",
  ERROR: "ERROR",
};

export const FRAME_SEQUENCE_MESSAGES = {
  [FRAME_SEQUENCE_STATUS.EMPTY]: "Frame buffer empty. Waiting for camera frames.",
  [FRAME_SEQUENCE_STATUS.COLLECTING]: "Collecting temporal frame sequence...",
  [FRAME_SEQUENCE_STATUS.READY]: "Frame sequence ready for preprocessing.",
  [FRAME_SEQUENCE_STATUS.PAUSED]: "Frame collection paused.",
  [FRAME_SEQUENCE_STATUS.CAMERA_STOPPED]: "Camera stopped. Frame collection inactive.",
  [FRAME_SEQUENCE_STATUS.ERROR]: "Frame sequence validation failed.",
};

/**
 * Safely close/release an ImageBitmap or transferable resource attached to a frame.
 * @param {Object} frame
 */
export function releaseFrameResource(frame) {
  if (!frame) return;
  try {
    if (frame.imageBitmap && typeof frame.imageBitmap.close === "function") {
      frame.imageBitmap.close();
      frame.imageBitmap = null;
    }
    if (frame.data && typeof frame.data.close === "function") {
      frame.data.close();
      frame.data = null;
    }
  } catch {
    // Suppress close error on already detached bitmap
  }
}

/**
 * Circular FIFO frame buffer with bounded capacity and resource cleanup.
 */
export class CircularFrameBuffer {
  /**
   * @param {number} capacity - Maximum frames retained (default: 64)
   */
  constructor(capacity = DEFAULT_SEQUENCE_CAPACITY) {
    if (typeof capacity !== "number" || capacity <= 0) {
      throw new Error(`Invalid buffer capacity: ${capacity}. Must be positive number.`);
    }
    this.capacity = capacity;
    this.buffer = [];
    this.lastSequenceIndex = -1;
  }

  /**
   * Current frame count in buffer.
   * @returns {number}
   */
  get size() {
    return this.buffer.length;
  }

  /**
   * Whether buffer has reached maximum capacity.
   * @returns {boolean}
   */
  get isFull() {
    return this.buffer.length >= this.capacity;
  }

  /**
   * Add a validated frame to the rolling buffer.
   * Evicts the oldest frame if capacity is reached.
   *
   * @param {Object} frame - Frame metadata & payload
   * @returns {boolean} - True if successfully inserted, false if rejected
   */
  push(frame) {
    if (!this.isValidFrame(frame)) {
      return false;
    }

    // Check for duplicate sequence index
    if (frame.sequenceIndex <= this.lastSequenceIndex) {
      // Reject duplicate or out-of-order sequence index
      return false;
    }

    // Check for non-chronological timestamp against latest item
    if (this.buffer.length > 0) {
      const lastFrame = this.buffer[this.buffer.length - 1];
      if (frame.timestamp <= lastFrame.timestamp) {
        return false;
      }
    }

    // Evict oldest frame when capacity is exceeded
    if (this.buffer.length >= this.capacity) {
      const evicted = this.buffer.shift();
      releaseFrameResource(evicted);
    }

    this.buffer.push(frame);
    this.lastSequenceIndex = frame.sequenceIndex;
    return true;
  }

  /**
   * Retrieve shallow copy of all frames in chronological order.
   * @returns {Array<Object>}
   */
  getFrames() {
    return [...this.buffer];
  }

  /**
   * Retrieve the oldest frame in the buffer.
   * @returns {Object|null}
   */
  peekOldest() {
    return this.buffer.length > 0 ? this.buffer[0] : null;
  }

  /**
   * Retrieve the newest frame in the buffer.
   * @returns {Object|null}
   */
  peekNewest() {
    return this.buffer.length > 0 ? this.buffer[this.buffer.length - 1] : null;
  }

  /**
   * Clear all frames and release associated image resources.
   */
  clear() {
    for (const frame of this.buffer) {
      releaseFrameResource(frame);
    }
    this.buffer = [];
    this.lastSequenceIndex = -1;
  }

  /**
   * Basic structural validation for individual incoming frame.
   * @param {Object} frame
   * @returns {boolean}
   */
  isValidFrame(frame) {
    if (!frame || typeof frame !== "object") return false;
    if (typeof frame.sequenceIndex !== "number" || frame.sequenceIndex < 0) return false;
    if (typeof frame.timestamp !== "number" || isNaN(frame.timestamp) || frame.timestamp <= 0) return false;
    if (typeof frame.width !== "number" || frame.width <= 0) return false;
    if (typeof frame.height !== "number" || frame.height <= 0) return false;
    return true;
  }

  /**
   * Comprehensive validation of the sequence before marking READY.
   * Checks:
   * 1. Exactly 64 frames available.
   * 2. Every frame has valid positive timestamp.
   * 3. Every frame has valid dimensions.
   * 4. Frames in strict chronological order.
   * 5. No duplicate sequence indices.
   * 6. No corrupted or undefined frames.
   *
   * @returns {{ valid: boolean, error: string|null }}
   */
  validateSequence() {
    if (this.buffer.length !== this.capacity) {
      return {
        valid: false,
        error: `Incomplete sequence: expected ${this.capacity} frames, got ${this.buffer.length}.`,
      };
    }

    const seenIndices = new Set();
    let previousTimestamp = -Infinity;

    for (let i = 0; i < this.buffer.length; i++) {
      const frame = this.buffer[i];

      if (!this.isValidFrame(frame)) {
        return {
          valid: false,
          error: `Corrupted frame found at position ${i}.`,
        };
      }

      if (seenIndices.has(frame.sequenceIndex)) {
        return {
          valid: false,
          error: `Duplicate sequence index ${frame.sequenceIndex} at position ${i}.`,
        };
      }
      seenIndices.add(frame.sequenceIndex);

      if (frame.timestamp <= previousTimestamp) {
        return {
          valid: false,
          error: `Non-chronological timestamp at position ${i}: ${frame.timestamp} <= ${previousTimestamp}.`,
        };
      }
      previousTimestamp = frame.timestamp;
    }

    return { valid: true, error: null };
  }
}
