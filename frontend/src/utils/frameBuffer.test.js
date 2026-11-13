import { describe, it, expect, vi } from "vitest";
import {
  CircularFrameBuffer,
  FRAME_SEQUENCE_STATUS,
  FRAME_SEQUENCE_MESSAGES,
  releaseFrameResource,
} from "./frameBuffer";

function createMockFrame(index, timestamp, width = 640, height = 480, extra = {}) {
  return {
    sequenceIndex: index,
    timestamp,
    width,
    height,
    imageBitmap: {
      close: vi.fn(),
      ...extra.bitmapProps,
    },
    ...extra,
  };
}

describe("CircularFrameBuffer (FE-05)", () => {
  it("initializes an empty buffer with capacity 64", () => {
    const buffer = new CircularFrameBuffer(64);
    expect(buffer.size).toBe(0);
    expect(buffer.isFull).toBe(false);
    expect(buffer.getFrames()).toEqual([]);
    expect(buffer.peekOldest()).toBeNull();
    expect(buffer.peekNewest()).toBeNull();
  });

  it("adds one valid frame", () => {
    const buffer = new CircularFrameBuffer(64);
    const frame = createMockFrame(0, 100);
    const success = buffer.push(frame);

    expect(success).toBe(true);
    expect(buffer.size).toBe(1);
    expect(buffer.isFull).toBe(false);
    expect(buffer.peekOldest()).toBe(frame);
    expect(buffer.peekNewest()).toBe(frame);
  });

  it("adds multiple frames while preserving chronological order", () => {
    const buffer = new CircularFrameBuffer(64);
    for (let i = 0; i < 10; i++) {
      const added = buffer.push(createMockFrame(i, 100 + i * 200));
      expect(added).toBe(true);
    }

    expect(buffer.size).toBe(10);
    const frames = buffer.getFrames();
    expect(frames[0].sequenceIndex).toBe(0);
    expect(frames[9].sequenceIndex).toBe(9);
    expect(frames[0].timestamp).toBe(100);
    expect(frames[9].timestamp).toBe(1900);
  });

  it("reaches maximum capacity of 64 frames", () => {
    const buffer = new CircularFrameBuffer(64);
    for (let i = 0; i < 64; i++) {
      buffer.push(createMockFrame(i, 100 + i * 200));
    }

    expect(buffer.size).toBe(64);
    expect(buffer.isFull).toBe(true);
    const validation = buffer.validateSequence();
    expect(validation.valid).toBe(true);
    expect(validation.error).toBeNull();
  });

  it("evicts oldest frame and calls close() on evicted ImageBitmap when capacity 64 is exceeded", () => {
    const buffer = new CircularFrameBuffer(64);
    const initialFrames = [];

    for (let i = 0; i < 64; i++) {
      const frame = createMockFrame(i, 100 + i * 200);
      initialFrames.push(frame);
      buffer.push(frame);
    }

    expect(buffer.size).toBe(64);
    const evictedBitmapClose = initialFrames[0].imageBitmap.close;
    expect(evictedBitmapClose).not.toHaveBeenCalled();

    // Add 65th frame (sequenceIndex 64)
    const frame65 = createMockFrame(64, 100 + 64 * 200);
    const added = buffer.push(frame65);

    expect(added).toBe(true);
    expect(buffer.size).toBe(64);
    expect(buffer.isFull).toBe(true);

    // Oldest frame (index 0) was evicted and its ImageBitmap closed and nulled
    expect(evictedBitmapClose).toHaveBeenCalledTimes(1);
    expect(initialFrames[0].imageBitmap).toBeNull();
    expect(buffer.peekOldest().sequenceIndex).toBe(1);
    expect(buffer.peekNewest().sequenceIndex).toBe(64);
  });

  it("rejects duplicate sequence indices", () => {
    const buffer = new CircularFrameBuffer(64);
    buffer.push(createMockFrame(0, 100));

    // Same index
    const dup = createMockFrame(0, 200);
    const success = buffer.push(dup);

    expect(success).toBe(false);
    expect(buffer.size).toBe(1);
  });

  it("rejects non-chronological frames (timestamp <= previous frame)", () => {
    const buffer = new CircularFrameBuffer(64);
    buffer.push(createMockFrame(0, 500));

    // Lower timestamp
    const older = createMockFrame(1, 400);
    const success = buffer.push(older);

    expect(success).toBe(false);
    expect(buffer.size).toBe(1);
  });

  it("rejects invalid frames (missing dimensions, invalid timestamp, null)", () => {
    const buffer = new CircularFrameBuffer(64);

    expect(buffer.push(null)).toBe(false);
    expect(buffer.push(undefined)).toBe(false);
    expect(buffer.push({ sequenceIndex: 0, timestamp: 100, width: 0, height: 480 })).toBe(false);
    expect(buffer.push({ sequenceIndex: 0, timestamp: 100, width: 640, height: -1 })).toBe(false);
    expect(buffer.push({ sequenceIndex: 0, timestamp: NaN, width: 640, height: 480 })).toBe(false);
    expect(buffer.push({ sequenceIndex: -1, timestamp: 100, width: 640, height: 480 })).toBe(false);
    expect(buffer.size).toBe(0);
  });

  it("clears all frames and releases all ImageBitmap resources", () => {
    const buffer = new CircularFrameBuffer(64);
    const frames = [];
    for (let i = 0; i < 5; i++) {
      const f = createMockFrame(i, 100 + i * 200);
      frames.push(f);
      buffer.push(f);
    }

    const closeSpies = frames.map((f) => f.imageBitmap.close);

    expect(buffer.size).toBe(5);
    buffer.clear();

    expect(buffer.size).toBe(0);
    expect(buffer.isFull).toBe(false);
    closeSpies.forEach((spy) => {
      expect(spy).toHaveBeenCalledTimes(1);
    });
    frames.forEach((f) => {
      expect(f.imageBitmap).toBeNull();
    });
  });

  it("validates sequence readiness correctly", () => {
    const buffer = new CircularFrameBuffer(64);

    // Incomplete buffer (< 64)
    for (let i = 0; i < 30; i++) {
      buffer.push(createMockFrame(i, 100 + i * 200));
    }
    let res = buffer.validateSequence();
    expect(res.valid).toBe(false);
    expect(res.error).toContain("Incomplete sequence: expected 64 frames, got 30");

    // Fill to 64
    for (let i = 30; i < 64; i++) {
      buffer.push(createMockFrame(i, 100 + i * 200));
    }
    res = buffer.validateSequence();
    expect(res.valid).toBe(true);
    expect(res.error).toBeNull();
  });

  it("safely handles releaseFrameResource with null, empty, or throwing objects", () => {
    expect(() => releaseFrameResource(null)).not.toThrow();
    expect(() => releaseFrameResource({})).not.toThrow();
    const thrower = {
      imageBitmap: {
        close: () => {
          throw new Error("Already closed");
        },
      },
    };
    expect(() => releaseFrameResource(thrower)).not.toThrow();
  });
});
