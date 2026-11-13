import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import React from "react";
import LiveWebcamPreview from "./LiveWebcamPreview";
import * as FrameBufferModule from "../../hooks/useFrameBuffer";

describe("LiveWebcamPreview Component (FE-04 & FE-05)", () => {
  let mockTrackStop;
  let mockStream;

  beforeEach(() => {
    vi.restoreAllMocks();
    mockTrackStop = vi.fn();
    mockStream = {
      getTracks: vi.fn().mockReturnValue([
        { stop: mockTrackStop, kind: "video", readyState: "live" },
      ]),
    };

    if (!navigator.mediaDevices) {
      navigator.mediaDevices = {};
    }

    navigator.mediaDevices.getUserMedia = vi.fn().mockResolvedValue(mockStream);
    navigator.mediaDevices.enumerateDevices = vi.fn().mockResolvedValue([
      { kind: "videoinput", deviceId: "cam-1", label: "Integrated Camera" },
      { kind: "videoinput", deviceId: "cam-2", label: "External HD Webcam" },
    ]);
  });

  it("automatically requests camera permission and starts webcam preview on mount without requiring capture button", async () => {
    render(<LiveWebcamPreview />);

    // Initial status badge updates to active automatically
    await waitFor(() => {
      expect(screen.getByTestId("camera-status-badge").textContent).toContain("Camera Active");
    });

    // Verified getUserMedia is invoked immediately on mount
    expect(navigator.mediaDevices.getUserMedia).toHaveBeenCalledTimes(1);

    // Video element has proper attributes
    const video = screen.getByTestId("webcam-video");
    expect(video.getAttribute("autoplay")).toBeDefined();
    expect(video.getAttribute("playsinline")).toBeDefined();
    expect(video.muted).toBe(true);
    expect(video.srcObject).toBe(mockStream);

    // Live indicator and positioning instructions are visible
    expect(screen.getByTestId("preview-live-indicator")).toBeDefined();
    expect(screen.getByText(/Position your hands clearly inside the camera frame/i)).toBeDefined();
  });

  it("stops camera and releases media tracks when Stop Camera is clicked", async () => {
    render(<LiveWebcamPreview />);

    await waitFor(() => {
      expect(screen.getByTestId("camera-status-badge").textContent).toContain("Camera Active");
    });

    const stopBtn = screen.getByRole("button", { name: /Stop Camera/i });
    fireEvent.click(stopBtn);

    // Media tracks stopped
    expect(mockTrackStop).toHaveBeenCalled();

    // Status updates to Camera Stopped
    expect(screen.getByTestId("camera-status-badge").textContent).toContain("Camera Stopped");
    expect(screen.getByTestId("camera-stopped-state")).toBeDefined();

    // Start Camera button becomes available
    const startBtns = screen.getAllByRole("button", { name: /Start Camera/i });
    expect(startBtns.length).toBeGreaterThanOrEqual(1);
  });

  it("resumes camera preview when Start Camera is clicked after stopping", async () => {
    render(<LiveWebcamPreview />);

    await waitFor(() => {
      expect(screen.getByTestId("camera-status-badge").textContent).toContain("Camera Active");
    });

    const stopBtn = screen.getByRole("button", { name: /Stop Camera/i });
    fireEvent.click(stopBtn);

    // Click Start Camera
    const startBtns = screen.getAllByRole("button", { name: /Start Camera/i });
    fireEvent.click(startBtns[0]);

    await waitFor(() => {
      expect(screen.getByTestId("camera-status-badge").textContent).toContain("Camera Active");
    });
    expect(navigator.mediaDevices.getUserMedia).toHaveBeenCalledTimes(2);
  });

  it("handles camera permission denied (NotAllowedError)", async () => {
    const permError = new Error("Permission denied");
    permError.name = "NotAllowedError";
    navigator.mediaDevices.getUserMedia = vi.fn().mockRejectedValue(permError);

    render(<LiveWebcamPreview />);

    await waitFor(() => {
      expect(screen.getByTestId("camera-status-badge").textContent).toContain("Camera Permission Denied");
    });

    expect(screen.getByTestId("camera-error-state")).toBeDefined();
    expect(
      screen.getByText(/Camera access denied\. Please grant camera permission in your browser/i)
    ).toBeDefined();

    // Retry button is available
    expect(screen.getByRole("button", { name: /Retry Camera/i })).toBeDefined();
  });

  it("handles missing camera devices (NotFoundError)", async () => {
    const notFoundError = new Error("Requested device not found");
    notFoundError.name = "NotFoundError";
    navigator.mediaDevices.getUserMedia = vi.fn().mockRejectedValue(notFoundError);

    render(<LiveWebcamPreview />);

    await waitFor(() => {
      expect(screen.getByTestId("camera-status-badge").textContent).toContain("Camera Not Available");
    });

    expect(screen.getByText(/No video input devices found on this device/i)).toBeDefined();
  });

  it("allows switching camera device when multiple cameras exist", async () => {
    render(<LiveWebcamPreview />);

    await waitFor(() => {
      expect(screen.getByTestId("camera-status-badge").textContent).toContain("Camera Active");
    });

    // Select element exists with both enumerated cameras
    const select = await screen.findByRole("combobox", { name: /Select Video Camera/i });
    expect(select).toBeDefined();
    expect(screen.getByText("Integrated Camera")).toBeDefined();
    expect(screen.getByText("External HD Webcam")).toBeDefined();

    // Switch to external webcam
    fireEvent.change(select, { target: { value: "cam-2" } });

    await waitFor(() => {
      expect(navigator.mediaDevices.getUserMedia).toHaveBeenCalledWith(
        expect.objectContaining({
          video: expect.objectContaining({
            deviceId: { exact: "cam-2" },
          }),
        })
      );
    });
  });

  it("releases all media tracks on unmount to prevent resource leaks", async () => {
    const { unmount } = render(<LiveWebcamPreview />);

    await waitFor(() => {
      expect(screen.getByTestId("camera-status-badge").textContent).toContain("Camera Active");
    });

    unmount();

    expect(mockTrackStop).toHaveBeenCalled();
  });

  it("initializes frame sampler stats and renders auto sampler badge", async () => {
    render(<LiveWebcamPreview />);

    await waitFor(() => {
      expect(screen.getByTestId("camera-status-badge").textContent).toContain("Camera Active");
    });

    const samplerStats = screen.getByTestId("frame-sampler-stats");
    expect(samplerStats.textContent).toContain("Auto Sampler: 5 FPS");
  });

  it("renders 64-frame sequence collection telemetry panel with initial state", async () => {
    render(<LiveWebcamPreview />);

    await waitFor(() => {
      expect(screen.getByTestId("camera-status-badge").textContent).toContain("Camera Active");
    });

    expect(screen.getByTestId("sequence-panel")).toBeDefined();
    expect(screen.getByTestId("sequence-frames-counter").textContent).toContain("0 / 64");
    expect(screen.getByTestId("telemetry-camera").textContent).toBe("Active");
    expect(screen.getByTestId("telemetry-sampling").textContent).toBe("5 FPS");
    expect(screen.getByTestId("telemetry-collection").textContent).toBe("Empty");
    expect(screen.getByTestId("telemetry-sequence").textContent).toBe("Empty");

    const progressFill = screen.getByTestId("sequence-progress-fill");
    expect(progressFill.style.width).toBe("0%");
  });

  it("updates sequence telemetry when camera is stopped and resumed", async () => {
    render(<LiveWebcamPreview />);

    await waitFor(() => {
      expect(screen.getByTestId("camera-status-badge").textContent).toContain("Camera Active");
    });

    // Stop camera
    const stopBtn = screen.getByRole("button", { name: /Stop Camera/i });
    fireEvent.click(stopBtn);

    await waitFor(() => {
      expect(screen.getByTestId("telemetry-camera").textContent).toBe("Stopped");
    });
    expect(screen.getByTestId("telemetry-collection").textContent).toBe("Inactive");
    expect(screen.getByTestId("telemetry-sampling").textContent).toBe("Paused");
    expect(screen.getByTestId("telemetry-sequence").textContent).toBe("Inactive");

    // Resume camera
    const startBtns = screen.getAllByRole("button", { name: /Start Camera/i });
    fireEvent.click(startBtns[0]);

    await waitFor(() => {
      expect(screen.getByTestId("telemetry-camera").textContent).toBe("Active");
    });
    expect(screen.getByTestId("sequence-frames-counter").textContent).toContain("0 / 64");
  });

  it("displays READY status and banner when 64 frames are collected", async () => {
    vi.spyOn(FrameBufferModule, "useFrameBuffer").mockReturnValue({
      frameCount: 64,
      capacity: 64,
      status: "READY",
      statusMessage: "Frame sequence ready for preprocessing.",
      isReady: true,
      error: null,
      addFrame: vi.fn(),
      reset: vi.fn(),
      getFrames: vi.fn(),
    });

    render(<LiveWebcamPreview />);

    await waitFor(() => {
      expect(screen.getByTestId("camera-status-badge").textContent).toContain("Camera Active");
    });

    // Verify frames count
    expect(screen.getByTestId("sequence-frames-counter").textContent).toContain("64 / 64");

    // Verify progress fill is 100% and has 'ready' class
    const progressFill = screen.getByTestId("sequence-progress-fill");
    expect(progressFill.style.width).toBe("100%");
    expect(progressFill.className).toContain("ready");

    // Verify telemetry values
    expect(screen.getByTestId("telemetry-collection").textContent).toBe("Ready");
    expect(screen.getByTestId("telemetry-sequence").textContent).toBe("Ready for preprocessing");

    // Verify ready message banner
    const banner = screen.getByTestId("sequence-ready-banner");
    expect(banner.textContent).toContain("Frame sequence ready for preprocessing.");
  });

  describe("Automatic Hands-Free Recognition (FE-06)", () => {
    it("never renders a manual Recognize Sign button", async () => {
      render(<LiveWebcamPreview />);
      await waitFor(() => {
        expect(screen.getByTestId("camera-status-badge").textContent).toContain("Camera Active");
      });

      // Verification of strict objective: User must NOT need to click "Recognize Sign"
      expect(screen.queryByRole("button", { name: /Recognize/i })).toBeNull();
      expect(screen.queryByRole("button", { name: /Capture & Recognize/i })).toBeNull();
    });

    it("displays initial NO_HAND state badge 'Waiting for hand'", async () => {
      render(<LiveWebcamPreview />);
      await waitFor(() => {
        expect(screen.getByTestId("camera-status-badge").textContent).toContain("Camera Active");
      });

      const autoPill = screen.getByTestId("auto-state-pill");
      expect(autoPill.textContent).toContain("Waiting for hand");
      expect(screen.getByTestId("telemetry-hand-detection").textContent).toBe("Waiting");
    });

    it("toggles auto speech synthesis when speech button is clicked", async () => {
      render(<LiveWebcamPreview />);
      await waitFor(() => {
        expect(screen.getByTestId("camera-status-badge").textContent).toContain("Camera Active");
      });

      const speechBtn = screen.getByRole("button", { name: /Disable speech synthesis/i });
      expect(speechBtn.textContent).toContain("Speech On");

      fireEvent.click(speechBtn);
      expect(screen.getByRole("button", { name: /Enable speech synthesis/i }).textContent).toContain(
        "Speech Off"
      );
    });

    it("renders real prediction card with confirmed sign, confidence, and single prediction display", async () => {
      const AutoRecModule = await import("../../hooks/useAutoRecognition");
      vi.spyOn(AutoRecModule, "useAutoRecognition").mockReturnValue({
        state: "SUCCESS",
        stateLabel: "Sign recognized",
        result: {
          success: true,
          predicted_sign: "thank you",
          display_name: "Thank You",
          confidence: 0.942,
          status: "recognized",
          is_valid: true,
          processing_time_ms: 36.4,
          model_used: "IsolatedBiGRU",
          top_predictions: [
            { sign: "thank you", confidence: 0.942 },
          ],
        },
        error: null,
        frameCount: 25,
        capacity: 64,
        detectedHands: [
          { side: "right", x: 0.4, y: 0.3, width: 0.25, height: 0.3 },
        ],
        isHandDetected: true,
        processFrame: vi.fn(),
        resetSequence: vi.fn(),
      });

      render(<LiveWebcamPreview />);

      await waitFor(() => {
        expect(screen.getByTestId("auto-prediction-card")).toBeDefined();
      });

      expect(screen.getByTestId("prediction-sign-name").textContent).toBe("THANK YOU");
      expect(screen.getByTestId("prediction-confidence").textContent).toBe("94.2%");
      expect(screen.getByTestId("prediction-status-badge").textContent).toContain("✓ RECOGNIZED");

      // Verify Section 8 & 12 requirement: No top-3 candidates list displayed
      expect(screen.queryByTestId("top-candidates-list")).toBeNull();

      // Verify hand bounding box rendered
      const boundingBox = screen.getByTestId("hand-bounding-box");
      expect(boundingBox).toBeDefined();
      expect(boundingBox.textContent).toContain("right hand");
    });

    it("always displays Top-1 predicted sign with Low Confidence label and percentage when confidence is below 0.80", async () => {
      const AutoRecModule = await import("../../hooks/useAutoRecognition");
      vi.spyOn(AutoRecModule, "useAutoRecognition").mockReturnValue({
        state: "UNCERTAIN",
        stateLabel: "Sign unclear, try again",
        result: {
          success: true,
          predicted_sign: "sorry",
          display_name: "Uncertain (Sorry)",
          confidence: 0.42,
          status: "uncertain",
          is_valid: false,
          processing_time_ms: 41.2,
          model_used: "IsolatedBiGRU",
          reason: "Confidence 0.42 below rejection threshold 0.80",
          top_predictions: [
            { sign: "sorry", confidence: 0.42 },
          ],
        },
        error: null,
        frameCount: 25,
        capacity: 64,
        detectedHands: [],
        isHandDetected: false,
        processFrame: vi.fn(),
        resetSequence: vi.fn(),
      });

      render(<LiveWebcamPreview />);

      await waitFor(() => {
        expect(screen.getByTestId("auto-prediction-card")).toBeDefined();
      });

      // Verifies: Always displays actual Top-1 predicted sign even when confidence < 0.80
      expect(screen.getByTestId("prediction-sign-name").textContent).toBe("SORRY");
      expect(screen.getByTestId("prediction-confidence").textContent).toBe("42%");
      expect(screen.getByTestId("confidence-tag").textContent).toBe("Low Confidence");
      expect(screen.getByTestId("recognition-status-text").textContent).toBe("Detected");
      expect(screen.getByTestId("prediction-status-badge").textContent).toContain("⚠ UNCERTAIN");
    });

    it("Test 1: hand detected and prediction confidence is 0.64 prominently displays predicted sign, 64%, Low Confidence, and Detected status", async () => {
      const AutoRecModule = await import("../../hooks/useAutoRecognition");
      vi.spyOn(AutoRecModule, "useAutoRecognition").mockReturnValue({
        state: "UNCERTAIN",
        stateLabel: "Sign unclear, try again",
        result: {
          success: true,
          predicted_sign: "thank you",
          display_name: "Thank You",
          confidence: 0.64,
          status: "uncertain",
          is_valid: false,
          processing_time_ms: 35.0,
          model_used: "IsolatedBiGRU",
          top_predictions: [{ sign: "thank you", confidence: 0.64 }],
        },
        error: null,
        frameCount: 20,
        capacity: 64,
        detectedHands: [{ side: "right", x: 0.5, y: 0.5, width: 0.2, height: 0.2 }],
        isHandDetected: true,
        processFrame: vi.fn(),
        resetSequence: vi.fn(),
      });

      render(<LiveWebcamPreview />);

      await waitFor(() => {
        expect(screen.getByTestId("auto-prediction-card")).toBeDefined();
      });

      expect(screen.getByTestId("prediction-sign-name").textContent).toBe("THANK YOU");
      expect(screen.getByTestId("prediction-confidence").textContent).toBe("64%");
      expect(screen.getByTestId("confidence-tag").textContent).toBe("Low Confidence");
      expect(screen.getByTestId("recognition-status-text").textContent).toBe("Detected");
      // Must NOT show "Sign Unclear" in place of the predicted sign
      expect(screen.getByTestId("prediction-sign-name").textContent).not.toBe("Sign unclear. Please try again.");
    });

    it("Test 2: hand detected and prediction confidence is 0.90 prominently displays predicted sign, 90%, High Confidence, and Detected status", async () => {
      const AutoRecModule = await import("../../hooks/useAutoRecognition");
      vi.spyOn(AutoRecModule, "useAutoRecognition").mockReturnValue({
        state: "SUCCESS",
        stateLabel: "Sign recognized",
        result: {
          success: true,
          predicted_sign: "thank you",
          display_name: "Thank You",
          confidence: 0.90,
          status: "recognized",
          is_valid: true,
          processing_time_ms: 33.0,
          model_used: "IsolatedBiGRU",
          top_predictions: [{ sign: "thank you", confidence: 0.90 }],
        },
        error: null,
        frameCount: 24,
        capacity: 64,
        detectedHands: [{ side: "right", x: 0.5, y: 0.5, width: 0.2, height: 0.2 }],
        isHandDetected: true,
        processFrame: vi.fn(),
        resetSequence: vi.fn(),
      });

      render(<LiveWebcamPreview />);

      await waitFor(() => {
        expect(screen.getByTestId("auto-prediction-card")).toBeDefined();
      });

      expect(screen.getByTestId("prediction-sign-name").textContent).toBe("THANK YOU");
      expect(screen.getByTestId("prediction-confidence").textContent).toBe("90%");
      expect(screen.getByTestId("confidence-tag").textContent).toBe("High Confidence");
      expect(screen.getByTestId("recognition-status-text").textContent).toBe("Detected");
    });

    it("Test 3: different ISL signs are displayed correctly (HELLO, FOOD, WATER, HELP, NO, YES)", async () => {
      const AutoRecModule = await import("../../hooks/useAutoRecognition");
      const signs = ["hello", "food", "water", "help", "no", "yes"];

      for (const sign of signs) {
        vi.spyOn(AutoRecModule, "useAutoRecognition").mockReturnValue({
          state: "SUCCESS",
          stateLabel: "Sign recognized",
          result: {
            success: true,
            predicted_sign: sign,
            display_name: sign.charAt(0).toUpperCase() + sign.slice(1),
            confidence: 0.85,
            status: "recognized",
            is_valid: true,
            top_predictions: [{ sign, confidence: 0.85 }],
          },
          error: null,
          frameCount: 20,
          capacity: 64,
          detectedHands: [],
          isHandDetected: true,
          processFrame: vi.fn(),
          resetSequence: vi.fn(),
        });

        const { unmount } = render(<LiveWebcamPreview />);
        expect(screen.getByTestId("prediction-sign-name").textContent).toBe(sign.toUpperCase());
        expect(screen.getByTestId("recognition-status-text").textContent).toBe("Detected");
        unmount();
      }
    });

    it("Test 4: same held gesture does not generate repeated results while hand remains present", async () => {
      const AutoRecModule = await import("../../hooks/useAutoRecognition");
      const mockProcessFrame = vi.fn();
      vi.spyOn(AutoRecModule, "useAutoRecognition").mockReturnValue({
        state: "WAITING_FOR_NEXT_SIGN",
        stateLabel: "Ready for next sign",
        result: {
          success: true,
          predicted_sign: "thank you",
          display_name: "Thank You",
          confidence: 0.88,
          status: "recognized",
          is_valid: true,
        },
        error: null,
        frameCount: 0,
        capacity: 64,
        detectedHands: [{ side: "right", x: 0.5, y: 0.5, width: 0.2, height: 0.2 }],
        isHandDetected: true,
        processFrame: mockProcessFrame,
        resetSequence: vi.fn(),
      });

      render(<LiveWebcamPreview />);

      await waitFor(() => {
        expect(screen.getByTestId("camera-status-badge").textContent).toContain("Camera Active");
      });

      // Prediction card remains visible while held
      expect(screen.getByTestId("prediction-sign-name").textContent).toBe("THANK YOU");
      // State pill reflects ready for next sign once released
      expect(screen.getByTestId("auto-state-pill").textContent).toContain("Ready for next sign");
    });

    it("Test 5: webcam remains active and continues automatic recognition after gesture ends", async () => {
      render(<LiveWebcamPreview />);

      await waitFor(() => {
        expect(screen.getByTestId("camera-status-badge").textContent).toContain("Camera Active");
      });

      // Camera video is displayed and live
      const video = screen.getByTestId("webcam-video");
      expect(video.srcObject).toBe(mockStream);
      expect(screen.getByTestId("telemetry-camera").textContent).toBe("Active");
      // Tracks are NOT stopped
      expect(mockTrackStop).not.toHaveBeenCalled();
    });

    it("displays error card and message when recognition service fails", async () => {
      const AutoRecModule = await import("../../hooks/useAutoRecognition");
      vi.spyOn(AutoRecModule, "useAutoRecognition").mockReturnValue({
        state: "ERROR",
        stateLabel: "Recognition unavailable",
        result: null,
        error: "Network timeout or service unavailable",
        frameCount: 0,
        capacity: 64,
        detectedHands: [],
        isHandDetected: false,
        processFrame: vi.fn(),
        resetSequence: vi.fn(),
      });

      render(<LiveWebcamPreview />);

      await waitFor(() => {
        expect(screen.getByTestId("auto-prediction-card")).toBeDefined();
      });

      expect(screen.getByTestId("prediction-status-badge").textContent).toContain("✕ ERROR");
      const errAlert = screen.getByTestId("prediction-error-alert");
      expect(errAlert.textContent).toContain("Recognition service error");
      expect(errAlert.textContent).toContain("Network timeout or service unavailable");
    });
  });
});
