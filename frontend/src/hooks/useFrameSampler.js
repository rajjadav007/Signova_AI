import { useEffect, useRef, useState, useCallback } from "react";

/**
 * useFrameSampler
 * Controlled off-screen frame capture mechanism for live video streams.
 *
 * @param {React.RefObject<HTMLVideoElement>} videoRef - Reference to live HTMLVideoElement
 * @param {Object} options
 * @param {boolean} options.enabled - Whether sampling is active
 * @param {number} options.intervalMs - Sampling interval in milliseconds (default 200ms / 5fps)
 * @param {number} options.targetWidth - Optional canvas scale width (default native video width)
 * @param {number} options.targetHeight - Optional canvas scale height (default native video height)
 * @param {Function} options.onFrame - Optional hook called when a frame is sampled (does NOT dispatch network requests)
 */
export function useFrameSampler(
  videoRef,
  {
    enabled = true,
    intervalMs = 200,
    targetWidth = null,
    targetHeight = null,
    onFrame = null,
  } = {}
) {
  const canvasRef = useRef(null);
  const timerRef = useRef(null);
  const isProcessingRef = useRef(false);
  const isMountedRef = useRef(true);

  const [stats, setStats] = useState({
    sampledCount: 0,
    lastSampleTime: null,
    fps: Math.round(1000 / intervalMs),
    resolution: null,
    isSampling: false,
  });

  // Lazy instantiate off-screen canvas element
  const getCanvas = useCallback(() => {
    if (!canvasRef.current && typeof document !== "undefined") {
      canvasRef.current = document.createElement("canvas");
    }
    return canvasRef.current;
  }, []);

  const sampleSingleFrame = useCallback(async () => {
    const video = videoRef.current;
    if (!video) return null;

    // Check if video is ready with valid dimensions and current frame data
    if (
      video.readyState < 2 || // HAVE_CURRENT_DATA
      video.paused ||
      video.ended ||
      !video.videoWidth ||
      !video.videoHeight
    ) {
      return null;
    }

    // Async lock to prevent overlapping frame tasks
    if (isProcessingRef.current) {
      return null;
    }

    isProcessingRef.current = true;

    try {
      const canvas = getCanvas();
      if (!canvas) return null;

      const width = targetWidth || video.videoWidth;
      const height = targetHeight || video.videoHeight;

      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }

      const ctx = canvas.getContext("2d", { willReadFrequently: true });
      if (!ctx) return null;

      // Draw the current video frame into the off-screen canvas
      ctx.drawImage(video, 0, 0, width, height);

      const timestamp = performance.now();
      const frameData = {
        canvas,
        width,
        height,
        timestamp,
      };

      // Execute optional local handler (e.g., preparation for FE-06)
      if (typeof onFrame === "function") {
        await onFrame(frameData);
      }

      if (isMountedRef.current) {
        setStats((prev) => ({
          ...prev,
          sampledCount: prev.sampledCount + 1,
          lastSampleTime: timestamp,
          resolution: `${width}x${height}`,
          isSampling: true,
        }));
      }

      return frameData;
    } catch {
      return null;
    } finally {
      isProcessingRef.current = false;
    }
  }, [videoRef, getCanvas, targetWidth, targetHeight, onFrame]);

  // Periodic controlled capture loop
  useEffect(() => {
    isMountedRef.current = true;

    if (!enabled) {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
      setStats((prev) => ({ ...prev, isSampling: false }));
      return;
    }

    // Set up controlled interval
    timerRef.current = setInterval(() => {
      sampleSingleFrame();
    }, intervalMs);

    return () => {
      isMountedRef.current = false;
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
      isProcessingRef.current = false;
      // Clean up canvas
      if (canvasRef.current) {
        canvasRef.current.width = 0;
        canvasRef.current.height = 0;
        canvasRef.current = null;
      }
    };
  }, [enabled, intervalMs, sampleSingleFrame]);

  return {
    stats,
    sampleSingleFrame,
  };
}
