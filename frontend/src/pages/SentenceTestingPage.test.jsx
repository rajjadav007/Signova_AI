import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import React from "react";
import SentenceTestingPage from "./SentenceTestingPage";
import * as api from "../services/api";

const MOCK_DEMO_CLASSES = {
  total: 5,
  classes: [
    { demo_class_id: 0, original_sentence_id: 5, sentence: "comb your hair", isl_gloss: "COMB YOU HAIR" },
    { demo_class_id: 1, original_sentence_id: 23, sentence: "help me", isl_gloss: "HELP ME" },
    { demo_class_id: 2, original_sentence_id: 35, sentence: "i am fine. thank you sir", isl_gloss: "I FINE. THANK YOU SIR" },
    { demo_class_id: 3, original_sentence_id: 53, sentence: "i need water", isl_gloss: "I NEED WATER" },
    { demo_class_id: 4, original_sentence_id: 91, sentence: "who are you", isl_gloss: "WHO YOU" },
  ],
};

const MOCK_SUCCESS_RECOGNITION = {
  success: true,
  is_recognized: true,
  status: "Recognized",
  predicted_sentence: "comb your hair",
  predicted_class_id: 0,
  original_sentence_id: 5,
  top_similarity: 0.6883,
  top_distance: 0.3117,
  runner_up_distance: 0.3696,
  margin_to_second: 0.0579,
  matching_latency_ms: 5.4,
  processing_time_ms: 1820.5,
  rankings: [
    { class_id: 0, original_sentence_id: 5, sentence: "comb your hair", isl_gloss: "COMB YOU HAIR", dtw_distance: 0.3117, similarity: 0.6883 },
    { class_id: 3, original_sentence_id: 53, sentence: "i need water", isl_gloss: "I NEED WATER", dtw_distance: 0.3696, similarity: 0.6304 },
    { class_id: 2, original_sentence_id: 35, sentence: "i am fine. thank you sir", isl_gloss: "I FINE. THANK YOU SIR", dtw_distance: 0.4177, similarity: 0.5823 },
    { class_id: 4, original_sentence_id: 91, sentence: "who are you", isl_gloss: "WHO YOU", dtw_distance: 0.4399, similarity: 0.5601 },
    { class_id: 1, original_sentence_id: 23, sentence: "help me", isl_gloss: "HELP ME", dtw_distance: 0.4746, similarity: 0.5254 },
  ],
  model_used: "DTWKNNClassifier-5Class",
  note: "Similarity score is uncalibrated DTW alignment similarity [0.0..1.0], not a class probability.",
};

const MOCK_REJECTED_RECOGNITION = {
  success: true,
  is_recognized: false,
  status: "Sentence not recognized.",
  predicted_sentence: null,
  predicted_class_id: null,
  original_sentence_id: null,
  top_similarity: 0.598,
  top_distance: 0.402,
  runner_up_distance: 0.405,
  margin_to_second: 0.003,
  matching_latency_ms: 7.2,
  processing_time_ms: 1910.0,
  rankings: [
    { class_id: 0, original_sentence_id: 5, sentence: "comb your hair", isl_gloss: "COMB YOU HAIR", dtw_distance: 0.402, similarity: 0.598 },
    { class_id: 3, original_sentence_id: 53, sentence: "i need water", isl_gloss: "I NEED WATER", dtw_distance: 0.405, similarity: 0.595 },
  ],
  model_used: "DTWKNNClassifier-5Class",
  note: "Similarity score is uncalibrated DTW alignment similarity [0.0..1.0], not a class probability.",
};

describe("Signova AI: SentenceTestingPage (Hackathon Demo)", () => {
  beforeEach(() => {
    vi.restoreAllMocks();

    const mockCreate = vi.fn().mockReturnValue("blob:http://localhost:3000/mock-video");
    const mockRevoke = vi.fn();

    if (typeof window !== "undefined") {
      window.URL.createObjectURL = mockCreate;
      window.URL.revokeObjectURL = mockRevoke;
    }
    if (typeof URL !== "undefined") {
      URL.createObjectURL = mockCreate;
      URL.revokeObjectURL = mockRevoke;
    }

    vi.spyOn(api, "fetchFiveClassClasses").mockResolvedValue(MOCK_DEMO_CLASSES);
    vi.spyOn(api, "fetchDemoSentenceClasses").mockResolvedValue(MOCK_DEMO_CLASSES);
  });

  it("renders page title, project name, supported formats, and initial dropzone", async () => {
    render(<SentenceTestingPage />);

    expect(screen.getByText(/Signova AI/i)).toBeDefined();
    expect(screen.getByText(/ISL Sentence Recognition/i)).toBeDefined();
    expect(screen.getByText(/Click to browse or drop ISL video here/i)).toBeDefined();
    expect(screen.getByText(/Supported formats: MP4, WebM, AVI, MOV, MKV/i)).toBeDefined();

    const recognizeBtn = screen.getByTestId("btn-recognize-sentence");
    expect(recognizeBtn.disabled).toBe(true);
  });

  it("handles valid video file upload and displays video preview player", async () => {
    render(<SentenceTestingPage />);

    const file = new File(["dummy-video-data"], "test_signing.mp4", { type: "video/mp4" });
    const input = screen.getByTestId("sentence-file-input");

    fireEvent.change(input, { target: { files: [file] } });

    expect(await screen.findByTestId("video-preview-player")).toBeDefined();
    expect(screen.getByText("test_signing.mp4")).toBeDefined();

    const recognizeBtn = screen.getByTestId("btn-recognize-sentence");
    expect(recognizeBtn.disabled).toBe(false);
  });

  it("displays error message when uploading an unsupported file format", async () => {
    render(<SentenceTestingPage />);

    const invalidFile = new File(["data"], "test.pdf", { type: "application/pdf" });
    const input = screen.getByTestId("sentence-file-input");

    fireEvent.change(input, { target: { files: [invalidFile] } });

    expect(await screen.findByRole("alert")).toBeDefined();
    expect(screen.getByText(/Unsupported video format "\.pdf"/i)).toBeDefined();
  });

  it("executes recognition, displays real prediction output without technical metrics", async () => {
    vi.spyOn(api, "recognizeFiveClassSentenceVideo").mockResolvedValue(MOCK_SUCCESS_RECOGNITION);

    render(<SentenceTestingPage />);

    const file = new File(["video-bytes"], "comb_sample.mp4", { type: "video/mp4" });
    const input = screen.getByTestId("sentence-file-input");
    fireEvent.change(input, { target: { files: [file] } });
    await screen.findByTestId("video-preview-player");

    const recognizeBtn = screen.getByTestId("btn-recognize-sentence");
    fireEvent.click(recognizeBtn);

    const heroSentence = await screen.findByTestId("predicted-sentence-text");
    expect(heroSentence.textContent.toLowerCase()).toContain("comb your hair");
    expect(screen.getByText(/RECOGNIZED SENTENCE:/i)).toBeDefined();

    // Technical metrics, accuracy, and probability panels removed
    expect(screen.queryByTestId("recognition-status-badge")).toBeNull();
    expect(screen.queryByTestId("probabilities-card")).toBeNull();
    expect(screen.queryByText(/Full 5-Class Alignment Rankings/i)).toBeNull();
  });

  it("handles unknown input rejection correctly displaying 'No sentence detected. Please try again.'", async () => {
    vi.spyOn(api, "recognizeFiveClassSentenceVideo").mockResolvedValue(MOCK_REJECTED_RECOGNITION);

    render(<SentenceTestingPage />);

    const file = new File(["video-bytes"], "unknown_gesture.mp4", { type: "video/mp4" });
    const input = screen.getByTestId("sentence-file-input");
    fireEvent.change(input, { target: { files: [file] } });
    await screen.findByTestId("video-preview-player");

    const recognizeBtn = screen.getByTestId("btn-recognize-sentence");
    fireEvent.click(recognizeBtn);

    const heroSentence = await screen.findByTestId("predicted-sentence-text");
    expect(heroSentence.textContent).toContain("No sentence detected. Please try again.");
    expect(screen.queryByTestId("recognition-status-badge")).toBeNull();
  });

  it("displays clean result card layout without accuracy or technical panels", async () => {
    vi.spyOn(api, "recognizeFiveClassSentenceVideo").mockResolvedValue(MOCK_SUCCESS_RECOGNITION);

    render(<SentenceTestingPage />);

    // Step 1: Upload and recognize
    const file = new File(["data"], "test_comb.mp4", { type: "video/mp4" });
    const input = screen.getByTestId("sentence-file-input");
    fireEvent.change(input, { target: { files: [file] } });
    await screen.findByTestId("video-preview-player");

    fireEvent.click(screen.getByTestId("btn-recognize-sentence"));

    const heroSentence = await screen.findByTestId("predicted-sentence-text");
    expect(heroSentence.textContent.toLowerCase()).toContain("comb your hair");

    // Verification and accuracy UI removed
    expect(screen.queryByTestId("verification-badge")).toBeNull();
    expect(screen.queryByTestId("stat-accuracy")).toBeNull();
    expect(screen.queryByTestId("history-table")).toBeNull();
  });

  it("renders 5 supported sentence cards and model status indicators in header", async () => {
    render(<SentenceTestingPage />);

    // Header status badges
    expect(screen.getByText(/Model: Five-Class DTW-KNN/i)).toBeDefined();
    expect(screen.getByText(/Supported sentences: 5/i)).toBeDefined();
    expect(screen.getByText(/Recognition status: Ready/i)).toBeDefined();

    // 5 supported sentence cards
    expect(screen.getByText(/"comb your hair"/i)).toBeDefined();
    expect(screen.getByText(/"help me"/i)).toBeDefined();
    expect(screen.getByText(/"i am fine\. thank you sir"/i)).toBeDefined();
    expect(screen.getByText(/"i need water"/i)).toBeDefined();
    expect(screen.getByText(/"who are you"/i)).toBeDefined();

    // Explanatory sentence banner
    expect(
      screen.getByText(/complete continuous Indian Sign Language sentence patterns/i)
    ).toBeDefined();
  });

  it("allows removing and clearing uploaded video", async () => {
    render(<SentenceTestingPage />);

    const file = new File(["video-bytes"], "preview_video.mp4", { type: "video/mp4" });
    const input = screen.getByTestId("sentence-file-input");
    fireEvent.change(input, { target: { files: [file] } });

    expect(await screen.findByTestId("video-preview-player")).toBeDefined();
    expect(screen.getByText("preview_video.mp4")).toBeDefined();

    // Click remove button
    const removeBtn = screen.getByTitle("Remove video");
    fireEvent.click(removeBtn);

    // Should return to dropzone
    expect(screen.getByText(/Click to browse or drop ISL video here/i)).toBeDefined();
    expect(screen.queryByTestId("video-preview-player")).toBeNull();
  });

  describe("Signova AI: BiGRU Attention Model & Live Webcam Integration", () => {
    const MOCK_BIGRU_SUCCESS = {
      success: true,
      is_recognized: true,
      status: "SUCCESS",
      predicted_sentence: "i need water",
      predicted_class_id: 3,
      original_sentence_id: 53,
      confidence: 0.9412,
      class_probabilities: {
        "comb your hair": 0.012,
        "help me": 0.015,
        "i am fine. thank you sir.": 0.021,
        "i need water": 0.9412,
        "who are you": 0.0108,
      },
      frame_count: 28,
      processing_time_ms: 1120.4,
      model_inference_latency_ms: 0.94,
      model_used: "FiveClassBiGRUAttentionModel",
    };

    const MOCK_BIGRU_REJECTED = {
      success: true,
      is_recognized: false,
      status: "INSUFFICIENT_FRAMES",
      predicted_sentence: "Sentence not recognized.",
      predicted_class_id: null,
      original_sentence_id: null,
      confidence: 0.0,
      class_probabilities: {},
      frame_count: 4,
      processing_time_ms: 320.0,
      model_inference_latency_ms: null,
      model_used: "FiveClassBiGRUAttentionModel",
      reason: "Only 4 frames captured. Minimum 8 frames required for a sentence.",
    };

    beforeEach(() => {
      // Mock MediaStream and getUserMedia
      const mockTrack = { stop: vi.fn() };
      const mockStream = {
        getTracks: vi.fn().mockReturnValue([mockTrack]),
      };

      Object.defineProperty(global.navigator, "mediaDevices", {
        value: {
          getUserMedia: vi.fn().mockResolvedValue(mockStream),
        },
        writable: true,
        configurable: true,
      });

      // Mock MediaRecorder
      class MockMediaRecorder {
        constructor(stream, options) {
          this.stream = stream;
          this.options = options;
          this.state = "inactive";
          this.ondataavailable = null;
          this.onstop = null;
        }
        start(timeslice) {
          this.state = "recording";
        }
        stop() {
          this.state = "inactive";
          if (this.ondataavailable) {
            this.ondataavailable({ data: new Blob(["dummy-frame-bytes"], { type: "video/webm" }) });
          }
          if (this.onstop) {
            this.onstop();
          }
        }
        static isTypeSupported() {
          return true;
        }
      }
      global.MediaRecorder = MockMediaRecorder;
    });

    it("toggles between BiGRU Attention and DTW-KNN models via architecture selector", () => {
      render(<SentenceTestingPage />);

      const activeBadge = screen.getByTestId("active-model-badge");
      expect(activeBadge.textContent).toContain("BiGRU Attention");

      // Switch to DTW-KNN
      const dtwBtn = screen.getByTestId("model-btn-dtw");
      fireEvent.click(dtwBtn);
      expect(activeBadge.textContent).toContain("DTW-KNN");

      // Switch back to BiGRU Attention
      const bigruBtn = screen.getByTestId("model-btn-bigru");
      fireEvent.click(bigruBtn);
      expect(activeBadge.textContent).toContain("BiGRU Attention");
    });

    it("switches input mode between file upload and live webcam", () => {
      render(<SentenceTestingPage />);

      // Initially on file upload mode
      expect(screen.getByTestId("video-dropzone")).toBeDefined();

      // Switch to Live Webcam Capture
      const webcamTab = screen.getByTestId("tab-webcam");
      fireEvent.click(webcamTab);

      expect(screen.getByTestId("webcam-inactive-box")).toBeDefined();
      expect(screen.getByText(/Live Webcam Recognition/i)).toBeDefined();

      // Switch back to File Upload
      const uploadTab = screen.getByTestId("tab-upload");
      fireEvent.click(uploadTab);
      expect(screen.getByTestId("video-dropzone")).toBeDefined();
    });

    it("executes complete live webcam signing flow with BiGRU probabilities and retry", async () => {
      vi.spyOn(api, "recognizeFiveClassSentenceVideo").mockResolvedValue(MOCK_BIGRU_SUCCESS);

      render(<SentenceTestingPage />);

      // 1. Switch to webcam mode
      fireEvent.click(screen.getByTestId("tab-webcam"));
      expect(screen.getByTestId("btn-start-camera")).toBeDefined();

      // 2. Start camera
      fireEvent.click(screen.getByTestId("btn-start-camera"));
      await waitFor(() => {
        expect(screen.getByTestId("webcam-video-feed")).toBeDefined();
      });
      expect(screen.getByText(/75 Landmarks \(Pose \+ Hands\) Ready/i)).toBeDefined();

      // 3. Start signing
      const startSigningBtn = screen.getByTestId("btn-start-signing");
      fireEvent.click(startSigningBtn);
      expect(screen.getByTestId("recording-indicator")).toBeDefined();

      // 4. Stop signing & recognize
      const stopSigningBtn = screen.getByTestId("btn-stop-signing");
      fireEvent.click(stopSigningBtn);

      // 5. Verify prediction results
      const heroSentence = await screen.findByTestId("predicted-sentence-text");
      expect(heroSentence.textContent.toLowerCase()).toContain("i need water");
      expect(screen.getByText(/RECOGNIZED SENTENCE:/i)).toBeDefined();

      // Technical metrics and probability panels are removed
      expect(screen.queryByTestId("probabilities-card")).toBeNull();
      expect(screen.queryByText("Conf: 94.1%")).toBeNull();

      // 6. Test retry recording button
      const retryBtn = screen.getByTestId("btn-retry-webcam");
      fireEvent.click(retryBtn);

      // Prediction is cleared, ready to sign again
      expect(screen.queryByTestId("predicted-sentence-text")).toBeNull();
      expect(screen.getByTestId("btn-start-signing")).toBeDefined();
    });

    it("handles insufficient frames or rejection in webcam mode safely", async () => {
      vi.spyOn(api, "recognizeFiveClassSentenceVideo").mockResolvedValue(MOCK_BIGRU_REJECTED);

      render(<SentenceTestingPage />);

      fireEvent.click(screen.getByTestId("tab-webcam"));
      fireEvent.click(screen.getByTestId("btn-start-camera"));
      await waitFor(() => expect(screen.getByTestId("webcam-video-feed")).toBeDefined());

      fireEvent.click(screen.getByTestId("btn-start-signing"));
      fireEvent.click(screen.getByTestId("btn-stop-signing"));

      const heroSentence = await screen.findByTestId("predicted-sentence-text");
      expect(heroSentence.textContent).toContain("No sentence detected. Please try again.");
    });

    it("verifies BiGRU file upload calls recognizeFiveClassSentenceVideo with 'bigru' and never 101-class model", async () => {
      const fiveClassSpy = vi.spyOn(api, "recognizeFiveClassSentenceVideo").mockResolvedValue(MOCK_BIGRU_SUCCESS);
      const sentence101Spy = vi.spyOn(api, "recognizeSentenceVideo");
      const demoSpy = vi.spyOn(api, "recognizeDemoSentenceVideo");

      render(<SentenceTestingPage />);

      const file = new File(["video-bytes"], "bigru_sample.mp4", { type: "video/mp4" });
      const input = screen.getByTestId("sentence-file-input");
      fireEvent.change(input, { target: { files: [file] } });
      await screen.findByTestId("video-preview-player");

      // Verify BiGRU is selected by default
      expect(screen.getByTestId("active-model-badge").textContent).toContain("BiGRU Attention");

      fireEvent.click(screen.getByTestId("btn-recognize-sentence"));

      await waitFor(() => {
        expect(fiveClassSpy).toHaveBeenCalledTimes(1);
        expect(fiveClassSpy).toHaveBeenCalledWith(file, "bigru");
        expect(sentence101Spy).not.toHaveBeenCalled();
        expect(demoSpy).not.toHaveBeenCalled();
      });
    });

    it("verifies DTW-KNN file upload calls recognizeFiveClassSentenceVideo with 'dtw' and never 101-class model", async () => {
      const fiveClassSpy = vi.spyOn(api, "recognizeFiveClassSentenceVideo").mockResolvedValue(MOCK_SUCCESS_RECOGNITION);
      const sentence101Spy = vi.spyOn(api, "recognizeSentenceVideo");
      const demoSpy = vi.spyOn(api, "recognizeDemoSentenceVideo");

      render(<SentenceTestingPage />);

      // Switch to DTW-KNN
      fireEvent.click(screen.getByTestId("model-btn-dtw"));
      expect(screen.getByTestId("active-model-badge").textContent).toContain("DTW-KNN");

      const file = new File(["video-bytes"], "dtw_sample.mp4", { type: "video/mp4" });
      const input = screen.getByTestId("sentence-file-input");
      fireEvent.change(input, { target: { files: [file] } });
      await screen.findByTestId("video-preview-player");

      fireEvent.click(screen.getByTestId("btn-recognize-sentence"));

      await waitFor(() => {
        expect(fiveClassSpy).toHaveBeenCalledTimes(1);
        expect(fiveClassSpy).toHaveBeenCalledWith(file, "dtw");
        expect(sentence101Spy).not.toHaveBeenCalled();
        expect(demoSpy).not.toHaveBeenCalled();
      });
    });

    it("displays backend errors clearly without silently falling back to another model", async () => {
      const fiveClassSpy = vi
        .spyOn(api, "recognizeFiveClassSentenceVideo")
        .mockRejectedValue(new Error("FastAPI 500: Landmark extraction pipeline failed"));
      const demoSpy = vi.spyOn(api, "recognizeDemoSentenceVideo");

      render(<SentenceTestingPage />);

      const file = new File(["video-bytes"], "corrupt_video.mp4", { type: "video/mp4" });
      const input = screen.getByTestId("sentence-file-input");
      fireEvent.change(input, { target: { files: [file] } });
      await screen.findByTestId("video-preview-player");

      fireEvent.click(screen.getByTestId("btn-recognize-sentence"));

      const alert = await screen.findByRole("alert");
      expect(alert.textContent).toContain("Landmark extraction pipeline failed");
      expect(demoSpy).not.toHaveBeenCalled();
      expect(screen.queryByTestId("predicted-sentence-text")).toBeNull();
    });
  });
});
