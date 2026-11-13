import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import React from "react";
import { useFrameSampler } from "./useFrameSampler";

describe("useFrameSampler Hook (FE-04 & FE-05)", () => {
  let mockVideo;

  beforeEach(() => {
    vi.useFakeTimers();

    // Create a mock HTMLVideoElement
    mockVideo = {
      readyState: 4, // HAVE_ENOUGH_DATA
      paused: false,
      ended: false,
      videoWidth: 640,
      videoHeight: 480,
    };

    // Mock HTMLCanvasElement.prototype.getContext
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue({
      drawImage: vi.fn(),
    });
  });

  afterEach(() => {
    vi.clearAllTimers();
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("checks video readiness and rejects video if readyState < 2", async () => {
    mockVideo.readyState = 1; // HAVE_METADATA only
    const videoRef = { current: mockVideo };
    const onFrame = vi.fn();

    const { result } = renderHook(() =>
      useFrameSampler(videoRef, { enabled: true, intervalMs: 200, onFrame })
    );

    await act(async () => {
      const frame = await result.current.sampleSingleFrame();
      expect(frame).toBeNull();
    });

    expect(onFrame).not.toHaveBeenCalled();
    expect(result.current.stats.sampledCount).toBe(0);
  });

  it("rejects video if video dimensions are invalid (0x0)", async () => {
    mockVideo.videoWidth = 0;
    mockVideo.videoHeight = 0;
    const videoRef = { current: mockVideo };
    const onFrame = vi.fn();

    const { result } = renderHook(() =>
      useFrameSampler(videoRef, { enabled: true, intervalMs: 200, onFrame })
    );

    await act(async () => {
      const frame = await result.current.sampleSingleFrame();
      expect(frame).toBeNull();
    });

    expect(onFrame).not.toHaveBeenCalled();
  });

  it("rejects sampling if video is paused or ended", async () => {
    mockVideo.paused = true;
    const videoRef = { current: mockVideo };
    const onFrame = vi.fn();

    const { result } = renderHook(() =>
      useFrameSampler(videoRef, { enabled: true, intervalMs: 200, onFrame })
    );

    await act(async () => {
      const frame = await result.current.sampleSingleFrame();
      expect(frame).toBeNull();
    });
    expect(onFrame).not.toHaveBeenCalled();

    mockVideo.paused = false;
    mockVideo.ended = true;

    await act(async () => {
      const frame = await result.current.sampleSingleFrame();
      expect(frame).toBeNull();
    });
    expect(onFrame).not.toHaveBeenCalled();
  });

  it("samples frames at configured interval (200ms default)", async () => {
    const videoRef = { current: mockVideo };
    const onFrame = vi.fn();

    const { result } = renderHook(() =>
      useFrameSampler(videoRef, { enabled: true, intervalMs: 200, onFrame })
    );

    // Fast-forward 200ms
    await act(async () => {
      await vi.advanceTimersByTimeAsync(200);
    });

    expect(onFrame).toHaveBeenCalledTimes(1);
    expect(result.current.stats.sampledCount).toBe(1);

    // Fast-forward another 200ms
    await act(async () => {
      await vi.advanceTimersByTimeAsync(200);
    });

    expect(onFrame).toHaveBeenCalledTimes(2);
    expect(result.current.stats.sampledCount).toBe(2);

    // Fast-forward another 200ms
    await act(async () => {
      await vi.advanceTimersByTimeAsync(200);
    });

    expect(onFrame).toHaveBeenCalledTimes(3);
    expect(result.current.stats.sampledCount).toBe(3);
  });

  it("prevents overlapping asynchronous processing via processing lock", async () => {
    const videoRef = { current: mockVideo };
    let resolveSlowTask;
    const slowOnFrame = vi.fn(
      () =>
        new Promise((resolve) => {
          resolveSlowTask = resolve;
        })
    );

    const { result } = renderHook(() =>
      useFrameSampler(videoRef, { enabled: true, intervalMs: 200, onFrame: slowOnFrame })
    );

    // Trigger first frame (starts async work)
    let firstCallPromise;
    act(() => {
      firstCallPromise = result.current.sampleSingleFrame();
    });

    // Try concurrent second frame before first finishes
    await act(async () => {
      const secondCall = await result.current.sampleSingleFrame();
      expect(secondCall).toBeNull();
    });

    // Resolve first task
    await act(async () => {
      resolveSlowTask();
      await firstCallPromise;
    });

    expect(slowOnFrame).toHaveBeenCalledTimes(1);
  });

  it("stops sampling and clears intervals when enabled is false", async () => {
    const videoRef = { current: mockVideo };
    const onFrame = vi.fn();
    let enabled = true;

    const { rerender } = renderHook(() =>
      useFrameSampler(videoRef, { enabled, intervalMs: 200, onFrame })
    );

    await act(async () => {
      await vi.advanceTimersByTimeAsync(200);
    });
    expect(onFrame).toHaveBeenCalledTimes(1);

    // Disable sampler
    enabled = false;
    rerender();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(600);
    });
    // No more samples taken
    expect(onFrame).toHaveBeenCalledTimes(1);
  });

  it("restarts sampling when enabled transitions from false back to true", async () => {
    const videoRef = { current: mockVideo };
    const onFrame = vi.fn();
    let enabled = false;

    const { rerender } = renderHook(() =>
      useFrameSampler(videoRef, { enabled, intervalMs: 200, onFrame })
    );

    await act(async () => {
      await vi.advanceTimersByTimeAsync(400);
    });
    expect(onFrame).not.toHaveBeenCalled();

    // Enable
    enabled = true;
    rerender();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(200);
    });
    expect(onFrame).toHaveBeenCalledTimes(1);
  });

  it("clears interval and releases canvas resources on unmount", async () => {
    const videoRef = { current: mockVideo };
    const onFrame = vi.fn();

    const { unmount } = renderHook(() =>
      useFrameSampler(videoRef, { enabled: true, intervalMs: 200, onFrame })
    );

    await act(async () => {
      await vi.advanceTimersByTimeAsync(200);
    });
    expect(onFrame).toHaveBeenCalledTimes(1);

    unmount();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(600);
    });
    expect(onFrame).toHaveBeenCalledTimes(1);
  });
});
