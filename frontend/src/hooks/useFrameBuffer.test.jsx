import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useFrameBuffer } from "./useFrameBuffer";
import { FRAME_SEQUENCE_STATUS, FRAME_SEQUENCE_MESSAGES } from "../utils/frameBuffer";

describe("useFrameBuffer Hook (FE-05)", () => {
  let mockTime = 1000;

  beforeEach(() => {
    mockTime = 1000;
    vi.spyOn(performance, "now").mockImplementation(() => {
      mockTime += 200;
      return mockTime;
    });
  });

  it("initializes with empty buffer when camera is active", () => {
    const { result } = renderHook(() => useFrameBuffer({ isCameraActive: true, capacity: 64 }));

    expect(result.current.frameCount).toBe(0);
    expect(result.current.capacity).toBe(64);
    expect(result.current.status).toBe(FRAME_SEQUENCE_STATUS.EMPTY);
    expect(result.current.statusMessage).toBe(FRAME_SEQUENCE_MESSAGES[FRAME_SEQUENCE_STATUS.EMPTY]);
    expect(result.current.isReady).toBe(false);
    expect(result.current.error).toBeNull();
  });

  it("initializes with CAMERA_STOPPED when camera is not active", () => {
    const { result } = renderHook(() => useFrameBuffer({ isCameraActive: false, capacity: 64 }));

    expect(result.current.frameCount).toBe(0);
    expect(result.current.status).toBe(FRAME_SEQUENCE_STATUS.CAMERA_STOPPED);
    expect(result.current.isReady).toBe(false);
  });

  it("ingests frames, increments frameCount, and updates status to COLLECTING", async () => {
    const { result } = renderHook(() => useFrameBuffer({ isCameraActive: true, capacity: 64 }));

    await act(async () => {
      await result.current.addFrame({ width: 640, height: 480 });
    });

    expect(result.current.frameCount).toBe(1);
    expect(result.current.status).toBe(FRAME_SEQUENCE_STATUS.COLLECTING);
    expect(result.current.isReady).toBe(false);
  });

  it("transitions to READY with 64 valid frames and shows readiness message", async () => {
    const { result } = renderHook(() => useFrameBuffer({ isCameraActive: true, capacity: 64 }));

    await act(async () => {
      for (let i = 0; i < 64; i++) {
        await result.current.addFrame({ width: 640, height: 480 });
      }
    });

    expect(result.current.frameCount).toBe(64);
    expect(result.current.status).toBe(FRAME_SEQUENCE_STATUS.READY);
    expect(result.current.isReady).toBe(true);
    expect(result.current.statusMessage).toBe("Frame sequence ready for preprocessing.");
    expect(result.current.getFrames().length).toBe(64);
  });

  it("maintains rolling buffer at 64 frames when extra frames are added", async () => {
    const { result } = renderHook(() => useFrameBuffer({ isCameraActive: true, capacity: 64 }));

    await act(async () => {
      for (let i = 0; i < 70; i++) {
        await result.current.addFrame({ width: 640, height: 480 });
      }
    });

    expect(result.current.frameCount).toBe(64);
    expect(result.current.status).toBe(FRAME_SEQUENCE_STATUS.READY);
    expect(result.current.isReady).toBe(true);
    const frames = result.current.getFrames();
    expect(frames.length).toBe(64);
    // Sequence indices should be 6 to 69
    expect(frames[0].sequenceIndex).toBe(6);
    expect(frames[63].sequenceIndex).toBe(69);
  });

  it("resets buffer and sets CAMERA_STOPPED when camera stops", async () => {
    let isActive = true;
    const { result, rerender } = renderHook(() =>
      useFrameBuffer({ isCameraActive: isActive, capacity: 64 })
    );

    await act(async () => {
      for (let i = 0; i < 10; i++) {
        await result.current.addFrame({ width: 640, height: 480 });
      }
    });
    expect(result.current.frameCount).toBe(10);

    // Stop camera
    isActive = false;
    rerender();

    expect(result.current.frameCount).toBe(0);
    expect(result.current.status).toBe(FRAME_SEQUENCE_STATUS.CAMERA_STOPPED);
    expect(result.current.isReady).toBe(false);
  });

  it("resets to EMPTY when camera restarts after being stopped", async () => {
    let isActive = false;
    const { result, rerender } = renderHook(() =>
      useFrameBuffer({ isCameraActive: isActive, capacity: 64 })
    );

    expect(result.current.status).toBe(FRAME_SEQUENCE_STATUS.CAMERA_STOPPED);

    // Start camera
    isActive = true;
    rerender();

    expect(result.current.frameCount).toBe(0);
    expect(result.current.status).toBe(FRAME_SEQUENCE_STATUS.EMPTY);
  });

  it("rejects invalid frames and returns false without breaking sequence", async () => {
    const { result } = renderHook(() => useFrameBuffer({ isCameraActive: true, capacity: 64 }));

    let success;
    await act(async () => {
      success = await result.current.addFrame(null);
    });
    expect(success).toBe(false);
    expect(result.current.frameCount).toBe(0);

    await act(async () => {
      success = await result.current.addFrame({ width: 0, height: 480 });
    });
    expect(success).toBe(false);
    expect(result.current.frameCount).toBe(0);
  });

  it("releases resources when unmounted", async () => {
    const { result, unmount } = renderHook(() =>
      useFrameBuffer({ isCameraActive: true, capacity: 64 })
    );

    const closeMock = vi.fn();
    await act(async () => {
      await result.current.addFrame({
        width: 640,
        height: 480,
        imageBitmap: { close: closeMock },
      });
    });

    unmount();
    expect(closeMock).toHaveBeenCalledTimes(1);
  });
});
