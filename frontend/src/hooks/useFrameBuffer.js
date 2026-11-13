import { useState, useRef, useEffect, useCallback } from "react";
import {
  CircularFrameBuffer,
  DEFAULT_SEQUENCE_CAPACITY,
  FRAME_SEQUENCE_STATUS,
  FRAME_SEQUENCE_MESSAGES,
} from "../utils/frameBuffer";

/**
 * useFrameBuffer
 * Reusable hook managing the 64-frame rolling sequence buffer for Signova AI (FE-05).
 * Stores frames in memory (via CircularFrameBuffer) and exposes lightweight reactive state.
 *
 * @param {Object} options
 * @param {number} [options.capacity=64] - Sequence buffer capacity
 * @param {boolean} [options.isCameraActive=true] - Current camera streaming status
 * @param {boolean} [options.enabled=true] - Whether frame insertion is allowed
 */
export function useFrameBuffer({
  capacity = DEFAULT_SEQUENCE_CAPACITY,
  isCameraActive = true,
  enabled = true,
} = {}) {
  const bufferRef = useRef(new CircularFrameBuffer(capacity));
  const sequenceIndexRef = useRef(0);
  const isMountedRef = useRef(true);

  // Sync buffer capacity if changed
  useEffect(() => {
    if (bufferRef.current.capacity !== capacity) {
      bufferRef.current.clear();
      bufferRef.current = new CircularFrameBuffer(capacity);
    }
  }, [capacity]);

  const [state, setState] = useState({
    frameCount: 0,
    capacity,
    status: isCameraActive ? FRAME_SEQUENCE_STATUS.EMPTY : FRAME_SEQUENCE_STATUS.CAMERA_STOPPED,
    statusMessage: isCameraActive
      ? FRAME_SEQUENCE_MESSAGES[FRAME_SEQUENCE_STATUS.EMPTY]
      : FRAME_SEQUENCE_MESSAGES[FRAME_SEQUENCE_STATUS.CAMERA_STOPPED],
    isReady: false,
    error: null,
  });

  // Reset and clear all stored frames
  const reset = useCallback(
    (reasonStatus = FRAME_SEQUENCE_STATUS.EMPTY) => {
      bufferRef.current.clear();
      sequenceIndexRef.current = 0;
      if (isMountedRef.current) {
        setState({
          frameCount: 0,
          capacity,
          status: reasonStatus,
          statusMessage: FRAME_SEQUENCE_MESSAGES[reasonStatus] || "",
          isReady: false,
          error: null,
        });
      }
    },
    [capacity]
  );

  // Process and ingest a single sampled frame
  const addFrame = useCallback(
    async (rawFrame) => {
      if (!isCameraActive) {
        return false;
      }
      if (!enabled) {
        return false;
      }
      if (!rawFrame) {
        return false;
      }

      const seqIdx = sequenceIndexRef.current;
      const timestamp = rawFrame.timestamp || performance.now();
      const width = rawFrame.width || (rawFrame.canvas ? rawFrame.canvas.width : 0);
      const height = rawFrame.height || (rawFrame.canvas ? rawFrame.canvas.height : 0);

      // Validate basic geometry
      if (!width || !height || width <= 0 || height <= 0) {
        return false;
      }

      // Convert or acquire ImageBitmap representation if supported
      let imageBitmap = null;
      if (rawFrame.imageBitmap) {
        imageBitmap = rawFrame.imageBitmap;
      } else if (rawFrame.canvas && typeof createImageBitmap === "function") {
        try {
          imageBitmap = await createImageBitmap(rawFrame.canvas);
        } catch {
          imageBitmap = null;
        }
      }

      const frame = {
        sequenceIndex: seqIdx,
        timestamp,
        width,
        height,
        imageBitmap,
        data: rawFrame.data || null,
      };

      const pushed = bufferRef.current.push(frame);
      if (!pushed) {
        return false;
      }

      sequenceIndexRef.current = seqIdx + 1;

      // Check sequence readiness
      const isFull = bufferRef.current.isFull;
      let newStatus = FRAME_SEQUENCE_STATUS.COLLECTING;
      let newMsg = FRAME_SEQUENCE_MESSAGES[FRAME_SEQUENCE_STATUS.COLLECTING];
      let isReady = false;
      let error = null;

      if (isFull) {
        const validation = bufferRef.current.validateSequence();
        if (validation.valid) {
          newStatus = FRAME_SEQUENCE_STATUS.READY;
          newMsg = FRAME_SEQUENCE_MESSAGES[FRAME_SEQUENCE_STATUS.READY];
          isReady = true;
        } else {
          newStatus = FRAME_SEQUENCE_STATUS.ERROR;
          newMsg = validation.error || FRAME_SEQUENCE_MESSAGES[FRAME_SEQUENCE_STATUS.ERROR];
          error = validation.error;
        }
      }

      if (isMountedRef.current) {
        setState({
          frameCount: bufferRef.current.size,
          capacity,
          status: newStatus,
          statusMessage: newMsg,
          isReady,
          error,
        });
      }

      return true;
    },
    [isCameraActive, enabled, capacity]
  );

  // Retrieve current frames snapshot (ordered latest 64)
  const getFrames = useCallback(() => {
    return bufferRef.current.getFrames();
  }, []);

  // Handle camera lifecycle changes
  useEffect(() => {
    if (!isCameraActive) {
      reset(FRAME_SEQUENCE_STATUS.CAMERA_STOPPED);
    } else {
      reset(FRAME_SEQUENCE_STATUS.EMPTY);
    }
  }, [isCameraActive, reset]);

  // Cleanup on unmount
  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
      bufferRef.current.clear();
      sequenceIndexRef.current = 0;
    };
  }, []);

  return {
    frameCount: state.frameCount,
    capacity: state.capacity,
    status: state.status,
    statusMessage: state.statusMessage,
    isReady: state.isReady,
    error: state.error,
    addFrame,
    reset,
    getFrames,
  };
}
