import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import React from "react";
import {
  useAutoRecognition,
  MIN_GESTURE_DURATION_MS,
  MAX_GESTURE_DURATION_MS,
  RESULT_DISPLAY_DURATION_MS,
  RESET_SEQUENCE_DURATION_MS,
} from "./useAutoRecognition";
import * as ApiModule from "../services/api";
import * as HandPresenceModule from "../utils/handPresence";

describe("useAutoRecognition Hook (FE-06)", () => {
  let mockStream;
  let mockRecorderInstance;

  beforeEach(() => {
    vi.useFakeTimers();

    mockStream = {
      getTracks: vi.fn().mockReturnValue([{ stop: vi.fn(), kind: "video" }]),
    };

    mockRecorderInstance = {
      state: "inactive",
      start: vi.fn().mockImplementation(function () {
        this.state = "recording";
      }),
      stop: vi.fn().mockImplementation(function () {
        this.state = "inactive";
        if (this.onstop) this.onstop();
      }),
      ondataavailable: null,
      onstop: null,
    };

    window.MediaRecorder = vi.fn().mockImplementation(function () {
      return mockRecorderInstance;
    });
    window.MediaRecorder.isTypeSupported = vi.fn().mockReturnValue(true);

    window.SpeechSynthesisUtterance = vi.fn().mockImplementation(function (text) {
      this.text = text;
    });
    window.speechSynthesis = {
      speak: vi.fn(),
      cancel: vi.fn(),
    };
  });

  afterEach(() => {
    vi.clearAllTimers();
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  // ---------------------------------------------------------------------------
  // Hand Detection Tests
  // ---------------------------------------------------------------------------
  describe("Hand Detection", () => {
    it("1. initializes in NO_HAND state when no hand is visible", () => {
      const { result } = renderHook(() =>
        useAutoRecognition({
          isCameraActive: true,
          stream: mockStream,
        })
      );

      expect(result.current.state).toBe("NO_HAND");
      expect(result.current.stateLabel).toBe("Waiting for hand");
      expect(result.current.frameCount).toBe(0);
      expect(result.current.result).toBeNull();
      expect(result.current.error).toBeNull();
    });

    it("2. enters RECORDING_GESTURE when hand enters focus zone for 3 consecutive frames", async () => {
      vi.spyOn(HandPresenceModule, "detectHandPresenceInCanvas").mockReturnValue({
        detected: true,
        confidence: 0.9,
        hands: [{ side: "right", x: 0.5, y: 0.5, width: 0.2, height: 0.2 }],
        skinRatio: 0.05,
      });

      const { result } = renderHook(() =>
        useAutoRecognition({
          isCameraActive: true,
          stream: mockStream,
        })
      );

      const mockFrame = { canvas: { width: 640, height: 480 }, timestamp: 1000 };

      // 1 frame: not yet confirmed
      await act(async () => {
        await result.current.processFrame(mockFrame);
      });
      expect(result.current.state).toBe("NO_HAND");

      // 2 frames: not yet confirmed
      await act(async () => {
        await result.current.processFrame(mockFrame);
      });
      expect(result.current.state).toBe("NO_HAND");

      // 3 frames: confirmed hand presence (HAND_ENTERED)
      await act(async () => {
        await result.current.processFrame(mockFrame);
      });
      expect(result.current.state).toBe("RECORDING_GESTURE");
      expect(result.current.stateLabel).toBe("Recording sign");
      expect(mockRecorderInstance.start).toHaveBeenCalled();
    });

    it("3 & 4. detects left hand and right hand correctly", async () => {
      vi.spyOn(HandPresenceModule, "detectHandPresenceInCanvas").mockReturnValue({
        detected: true,
        confidence: 0.88,
        hands: [
          { side: "left", x: 0.2, y: 0.4, width: 0.2, height: 0.2 },
          { side: "right", x: 0.6, y: 0.4, width: 0.2, height: 0.2 },
        ],
        skinRatio: 0.08,
      });

      const { result } = renderHook(() =>
        useAutoRecognition({
          isCameraActive: true,
          stream: mockStream,
        })
      );

      const mockFrame = { canvas: { width: 640, height: 480 }, timestamp: 1000 };

      await act(async () => {
        for (let i = 0; i < 3; i++) {
          await result.current.processFrame(mockFrame);
        }
      });

      expect(result.current.detectedHands.length).toBe(2);
      expect(result.current.detectedHands.map((h) => h.side)).toEqual(["left", "right"]);
    });

    it("6. temporary detection loss (1-2 frames) does not split or stop recording", async () => {
      let isHandVisible = true;
      vi.spyOn(HandPresenceModule, "detectHandPresenceInCanvas").mockImplementation(() => ({
        detected: isHandVisible,
        confidence: isHandVisible ? 0.9 : 0.0,
        hands: isHandVisible ? [{ side: "right" }] : [],
        skinRatio: isHandVisible ? 0.05 : 0.0,
      }));

      const { result } = renderHook(() =>
        useAutoRecognition({
          isCameraActive: true,
          stream: mockStream,
        })
      );

      const mockFrame = { canvas: { width: 640, height: 480 }, timestamp: 1000 };

      // Hand enters (3 frames)
      await act(async () => {
        for (let i = 0; i < 3; i++) {
          await result.current.processFrame(mockFrame);
        }
      });
      expect(result.current.state).toBe("RECORDING_GESTURE");

      // Temporarily lost for 2 frames (< 5 frames lostThreshold)
      isHandVisible = false;
      await act(async () => {
        await result.current.processFrame(mockFrame);
        await result.current.processFrame(mockFrame);
      });

      // Still in RECORDING_GESTURE!
      expect(result.current.state).toBe("RECORDING_GESTURE");
      expect(mockRecorderInstance.stop).not.toHaveBeenCalled();

      // Hand reappears
      isHandVisible = true;
      await act(async () => {
        await result.current.processFrame(mockFrame);
      });
      expect(result.current.state).toBe("RECORDING_GESTURE");
    });
  });

  // ---------------------------------------------------------------------------
  // Gesture Recording Tests
  // ---------------------------------------------------------------------------
  describe("Gesture Recording & End Detection", () => {
    it("9 & 10. recording starts automatically and stops after stable hand loss (5 frames)", async () => {
      let isHandVisible = true;
      vi.spyOn(HandPresenceModule, "detectHandPresenceInCanvas").mockImplementation(() => ({
        detected: isHandVisible,
        confidence: isHandVisible ? 0.9 : 0.0,
        hands: isHandVisible ? [{ side: "right" }] : [],
        skinRatio: isHandVisible ? 0.05 : 0.0,
      }));

      const { result } = renderHook(() =>
        useAutoRecognition({
          isCameraActive: true,
          stream: mockStream,
        })
      );

      const mockFrame = { canvas: { width: 640, height: 480 }, timestamp: 1000 };

      // Hand enters
      await act(async () => {
        for (let i = 0; i < 3; i++) {
          await result.current.processFrame(mockFrame);
        }
      });
      expect(result.current.state).toBe("RECORDING_GESTURE");
      expect(mockRecorderInstance.start).toHaveBeenCalledTimes(1);

      // Hand leaves: 5 frames absent confirms hand loss (gesture end)
      isHandVisible = false;
      await act(async () => {
        for (let i = 0; i < 5; i++) {
          await result.current.processFrame(mockFrame);
        }
      });

      expect(mockRecorderInstance.stop).toHaveBeenCalledTimes(1);
    });

    it("12 & 13. rejects recording if duration is shorter than minimum threshold (< 800ms)", async () => {
      vi.spyOn(HandPresenceModule, "detectHandPresenceInCanvas").mockReturnValue({
        detected: true,
        confidence: 0.9,
        hands: [{ side: "right" }],
        skinRatio: 0.05,
      });

      vi.spyOn(ApiModule, "recognizeIsolatedVideo").mockResolvedValue({
        success: true,
        predicted_sign: "hello",
      });

      const { result } = renderHook(() =>
        useAutoRecognition({
          isCameraActive: true,
          stream: mockStream,
        })
      );

      const mockFrame = { canvas: { width: 640, height: 480 }, timestamp: 1000 };

      await act(async () => {
        for (let i = 0; i < 3; i++) {
          await result.current.processFrame(mockFrame);
        }
      });
      expect(result.current.state).toBe("RECORDING_GESTURE");

      // Advance only 300ms (too short: < 800ms)
      act(() => {
        vi.advanceTimersByTime(300);
      });

      // Stop recorder
      await act(async () => {
        mockRecorderInstance.stop();
      });

      // Backend must NOT be called for short gesture
      expect(ApiModule.recognizeIsolatedVideo).not.toHaveBeenCalled();
      expect(result.current.state).toBe("RESET_SEQUENCE");

      // Advances to NO_HAND
      act(() => {
        vi.advanceTimersByTime(RESET_SEQUENCE_DURATION_MS);
      });
      expect(result.current.state).toBe("NO_HAND");
    });

    it("14. validates WebM recording of adequate duration (>= 800ms) and moves to recognition", async () => {
      vi.spyOn(HandPresenceModule, "detectHandPresenceInCanvas").mockReturnValue({
        detected: true,
        confidence: 0.9,
        hands: [{ side: "right" }],
        skinRatio: 0.05,
      });

      const mockApiResponse = {
        success: true,
        predicted_sign: "water",
        display_name: "Water",
        confidence: 0.945,
        status: "recognized",
        is_valid: true,
        processing_time_ms: 38.0,
        model_used: "IsolatedBiGRU",
        top_predictions: [{ sign: "water", confidence: 0.945 }],
      };
      vi.spyOn(ApiModule, "recognizeIsolatedVideo").mockResolvedValue(mockApiResponse);

      const { result } = renderHook(() =>
        useAutoRecognition({
          isCameraActive: true,
          stream: mockStream,
        })
      );

      const mockFrame = { canvas: { width: 640, height: 480 }, timestamp: 1000 };

      await act(async () => {
        for (let i = 0; i < 3; i++) {
          await result.current.processFrame(mockFrame);
        }
      });

      // Simulate recorded data available
      mockRecorderInstance.ondataavailable({ data: new Blob(["webm-data"], { type: "video/webm" }) });

      // Advance 1500ms (> 800ms valid duration)
      act(() => {
        vi.advanceTimersByTime(1500);
      });

      await act(async () => {
        mockRecorderInstance.stop();
      });

      // Dispatches recognition
      expect(ApiModule.recognizeIsolatedVideo).toHaveBeenCalledTimes(1);
      expect(result.current.state).toBe("SUCCESS");
      expect(result.current.result.predicted_sign).toBe("water");
    });
  });

  // ---------------------------------------------------------------------------
  // Backend Recognition & Single Prediction Display Tests
  // ---------------------------------------------------------------------------
  describe("Backend Recognition", () => {
    it("16 & 17. submits only one request per completed gesture and prevents duplicates", async () => {
      vi.spyOn(ApiModule, "recognizeIsolatedVideo").mockImplementation(
        () => new Promise((resolve) => setTimeout(() => resolve({ success: true, predicted_sign: "hello", confidence: 0.95, status: "recognized" }), 500))
      );

      const { result } = renderHook(() =>
        useAutoRecognition({
          isCameraActive: true,
          stream: mockStream,
        })
      );

      const blob = new Blob(["mock video"], { type: "video/webm" });

      // First dispatch
      let p1;
      act(() => {
        p1 = result.current.dispatchRecognition(blob);
      });

      // Second simultaneous dispatch attempt
      let p2;
      act(() => {
        p2 = result.current.dispatchRecognition(blob);
      });

      await act(async () => {
        vi.advanceTimersByTime(600);
        await Promise.all([p1, p2]);
      });

      // Exactly 1 API call
      expect(ApiModule.recognizeIsolatedVideo).toHaveBeenCalledTimes(1);
    });

    it("20. handles uncertain prediction status when confidence is below 0.80", async () => {
      const mockUncertainResponse = {
        success: true,
        predicted_sign: "sorry",
        display_name: "Uncertain (Sorry)",
        confidence: 0.65,
        status: "uncertain",
        is_valid: false,
        processing_time_ms: 40.0,
        model_used: "IsolatedBiGRU",
        reason: "Confidence 0.65 below rejection threshold 0.80",
        top_predictions: [{ sign: "sorry", confidence: 0.65 }],
      };
      vi.spyOn(ApiModule, "recognizeIsolatedVideo").mockResolvedValue(mockUncertainResponse);

      const { result } = renderHook(() =>
        useAutoRecognition({
          isCameraActive: true,
          stream: mockStream,
        })
      );

      await act(async () => {
        await result.current.dispatchRecognition(new Blob(["mock"], { type: "video/webm" }));
      });

      expect(result.current.state).toBe("UNCERTAIN");
      expect(result.current.stateLabel).toBe("Sign unclear, try again");
      expect(result.current.result.reason).toContain("rejection threshold");
    });

    it("21. handles backend error gracefully and clears any previous result", async () => {
      vi.spyOn(ApiModule, "recognizeIsolatedVideo").mockRejectedValue(
        new Error("Recognition service unavailable")
      );

      const { result } = renderHook(() =>
        useAutoRecognition({
          isCameraActive: true,
          stream: mockStream,
        })
      );

      await act(async () => {
        await result.current.dispatchRecognition(new Blob(["mock"], { type: "video/webm" }));
      });

      expect(result.current.state).toBe("ERROR");
      expect(result.current.stateLabel).toBe("Recognition unavailable");
      expect(result.current.error).toBe("Recognition service unavailable");
      expect(result.current.result).toBeNull();
    });

    it("21b. rejects empty or zero-byte blob recording without deadlock and resets to NO_HAND", async () => {
      const { result } = renderHook(() =>
        useAutoRecognition({
          isCameraActive: true,
          stream: mockStream,
        })
      );

      const mockFrame = { canvas: { width: 640, height: 480 }, timestamp: 1000 };

      // Hand enters
      vi.spyOn(HandPresenceModule, "detectHandPresenceInCanvas").mockReturnValue({
        detected: true,
        confidence: 0.9,
        hands: [{ side: "right" }],
        skinRatio: 0.05,
      });

      await act(async () => {
        for (let i = 0; i < 3; i++) {
          await result.current.processFrame(mockFrame);
        }
      });
      expect(result.current.state).toBe("RECORDING_GESTURE");

      // Advance 1200ms (> MIN_GESTURE_DURATION_MS) but no data chunks recorded (zero-byte blob)
      act(() => {
        vi.advanceTimersByTime(1200);
      });

      await act(async () => {
        mockRecorderInstance.stop();
      });

      // Discards recording and enters RESET_SEQUENCE
      expect(result.current.state).toBe("RESET_SEQUENCE");

      // Advances past RESET_SEQUENCE_DURATION_MS to NO_HAND (preventing deadlock)
      act(() => {
        vi.advanceTimersByTime(RESET_SEQUENCE_DURATION_MS);
      });
      expect(result.current.state).toBe("NO_HAND");
    });
  });

  // ---------------------------------------------------------------------------
  // Sequence Reset and Sequential Recognition Tests
  // ---------------------------------------------------------------------------
  describe("Sequence Reset & Sequential Recognition", () => {
    it("22, 23 & 25. resets sequence and frame buffer between signs without stopping camera", async () => {
      const mockResetBuffer = vi.fn();
      const mockApiResponse = {
        success: true,
        predicted_sign: "hello",
        display_name: "Hello",
        confidence: 0.95,
        status: "recognized",
        is_valid: true,
      };
      vi.spyOn(ApiModule, "recognizeIsolatedVideo").mockResolvedValue(mockApiResponse);

      const { result } = renderHook(() =>
        useAutoRecognition({
          isCameraActive: true,
          stream: mockStream,
          onResetSequence: mockResetBuffer,
        })
      );

      await act(async () => {
        await result.current.dispatchRecognition(new Blob(["mock"], { type: "video/webm" }));
      });
      expect(result.current.state).toBe("SUCCESS");

      // Advance through 2.5s display window
      act(() => {
        vi.advanceTimersByTime(RESULT_DISPLAY_DURATION_MS);
      });

      expect(result.current.state).toBe("RESET_SEQUENCE");
      expect(mockResetBuffer).toHaveBeenCalled();
      expect(result.current.frameCount).toBe(0);

      // Camera stream was NOT stopped
      expect(mockStream.getTracks()[0].stop).not.toHaveBeenCalled();

      // Advance through reset sequence transition (400ms)
      act(() => {
        vi.advanceTimersByTime(RESET_SEQUENCE_DURATION_MS);
      });
      expect(result.current.state).toBe("WAITING_FOR_NEXT_SIGN");
      expect(result.current.stateLabel).toBe("Ready for next sign");
    });

    it("sequential recognition: tests 3 different signs consecutively without page refresh", async () => {
      let isHandVisible = false;
      vi.spyOn(HandPresenceModule, "detectHandPresenceInCanvas").mockImplementation(() => ({
        detected: isHandVisible,
        confidence: isHandVisible ? 0.9 : 0.0,
        hands: isHandVisible ? [{ side: "right" }] : [],
        skinRatio: isHandVisible ? 0.05 : 0.0,
      }));

      const signsToRecognize = [
        { predicted_sign: "hello", display_name: "Hello", confidence: 0.96 },
        { predicted_sign: "thank you", display_name: "Thank You", confidence: 0.93 },
        { predicted_sign: "water", display_name: "Water", confidence: 0.91 },
      ];

      let signIdx = 0;
      vi.spyOn(ApiModule, "recognizeIsolatedVideo").mockImplementation(async () => {
        const item = signsToRecognize[signIdx];
        signIdx++;
        return {
          success: true,
          predicted_sign: item.predicted_sign,
          display_name: item.display_name,
          confidence: item.confidence,
          status: "recognized",
          is_valid: true,
        };
      });

      const { result } = renderHook(() =>
        useAutoRecognition({
          isCameraActive: true,
          stream: mockStream,
        })
      );

      const mockFrame = { canvas: { width: 640, height: 480 }, timestamp: 1000 };

      // --- SIGN 1: HELLO ---
      isHandVisible = true;
      await act(async () => {
        for (let i = 0; i < 3; i++) await result.current.processFrame(mockFrame);
      });
      expect(result.current.state).toBe("RECORDING_GESTURE");

      mockRecorderInstance.ondataavailable({ data: new Blob(["s1"], { type: "video/webm" }) });
      act(() => {
        vi.advanceTimersByTime(1200);
      });

      // Hand leaves
      isHandVisible = false;
      await act(async () => {
        for (let i = 0; i < 5; i++) await result.current.processFrame(mockFrame);
      });
      expect(result.current.state).toBe("SUCCESS");
      expect(result.current.result.predicted_sign).toBe("hello");

      // Advance past display + reset
      act(() => {
        vi.advanceTimersByTime(RESULT_DISPLAY_DURATION_MS + RESET_SEQUENCE_DURATION_MS);
      });
      expect(result.current.state).toBe("WAITING_FOR_NEXT_SIGN");

      // Hand release confirmed (hand is down)
      await act(async () => {
        await result.current.processFrame(mockFrame);
      });
      expect(result.current.state).toBe("NO_HAND");

      // --- SIGN 2: THANK YOU ---
      isHandVisible = true;
      await act(async () => {
        for (let i = 0; i < 3; i++) await result.current.processFrame(mockFrame);
      });
      expect(result.current.state).toBe("RECORDING_GESTURE");

      mockRecorderInstance.ondataavailable({ data: new Blob(["s2"], { type: "video/webm" }) });
      act(() => {
        vi.advanceTimersByTime(1500);
      });

      // Hand leaves
      isHandVisible = false;
      await act(async () => {
        for (let i = 0; i < 5; i++) await result.current.processFrame(mockFrame);
      });
      expect(result.current.state).toBe("SUCCESS");
      expect(result.current.result.predicted_sign).toBe("thank you");

      // Advance past display + reset
      act(() => {
        vi.advanceTimersByTime(RESULT_DISPLAY_DURATION_MS + RESET_SEQUENCE_DURATION_MS);
      });
      expect(result.current.state).toBe("WAITING_FOR_NEXT_SIGN");

      // Hand release confirmed
      await act(async () => {
        await result.current.processFrame(mockFrame);
      });
      expect(result.current.state).toBe("NO_HAND");

      // --- SIGN 3: WATER ---
      isHandVisible = true;
      await act(async () => {
        for (let i = 0; i < 3; i++) await result.current.processFrame(mockFrame);
      });
      expect(result.current.state).toBe("RECORDING_GESTURE");

      mockRecorderInstance.ondataavailable({ data: new Blob(["s3"], { type: "video/webm" }) });
      act(() => {
        vi.advanceTimersByTime(1400);
      });

      // Hand leaves
      isHandVisible = false;
      await act(async () => {
        for (let i = 0; i < 5; i++) await result.current.processFrame(mockFrame);
      });
      expect(result.current.state).toBe("SUCCESS");
      expect(result.current.result.predicted_sign).toBe("water");

      // All 3 signs recognized sequentially
      expect(ApiModule.recognizeIsolatedVideo).toHaveBeenCalledTimes(3);
    });
  });
});
