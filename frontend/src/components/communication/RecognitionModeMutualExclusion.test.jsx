import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor, within } from "@testing-library/react";
import React from "react";
import LiveRecognitionPage from "../../pages/LiveRecognitionPage";
import * as api from "../../services/api";

// Mock mediaDevices
if (!navigator.mediaDevices) {
  navigator.mediaDevices = {};
}
const mockVideoTrack = { stop: vi.fn(), kind: "video", readyState: "live" };
const mockAudioTrack = { stop: vi.fn(), kind: "audio", readyState: "live" };

navigator.mediaDevices.getUserMedia = vi.fn().mockImplementation((constraints) => {
  if (constraints && constraints.audio) {
    return Promise.resolve({
      getTracks: () => [mockAudioTrack],
    });
  }
  return Promise.resolve({
    getTracks: () => [mockVideoTrack],
  });
});
navigator.mediaDevices.enumerateDevices = vi.fn().mockResolvedValue([]);

// Mock SpeechSynthesis
window.speechSynthesis = {
  speak: vi.fn(),
  cancel: vi.fn(),
  getVoices: vi.fn().mockReturnValue([]),
  onvoiceschanged: null,
};

// Mock Audio
window.Audio = vi.fn().mockImplementation(() => ({
  play: vi.fn().mockResolvedValue(undefined),
  pause: vi.fn(),
  currentTime: 0,
  onended: null,
  onerror: null,
}));

describe("Mutual Exclusion Between Word and Continuous Sentence Recognition Modes", () => {
  beforeEach(() => {
    vi.clearAllMocks();

    vi.spyOn(api, "translateText").mockResolvedValue({
      success: true,
      translated_text: "પાણી",
      source_language_code: "en-IN",
      target_language_code: "gu-IN",
    });

    vi.spyOn(api, "synthesizeSpeech").mockResolvedValue({
      success: true,
      audio_base64: "MOCK_AUDIO",
      audio_format: "wav",
      target_language_code: "gu-IN",
      speaker: "priya",
    });

    vi.spyOn(api, "recognizeIsolatedVideo").mockResolvedValue({
      predicted_sign: "Water",
      confidence: 0.95,
      is_confident: true,
      status: "recognized",
    });

    vi.spyOn(api, "recognizeSentenceLandmarks").mockResolvedValue({
      success: true,
      status: "recognized",
      sentence: "are you free today",
      confidence: 0.942,
      gloss_tokens: ["YOU", "FREE", "TODAY"],
      top_predictions: [
        { sentence: "are you free today", confidence: 0.942 },
      ],
    });

    vi.spyOn(api, "recognizeFiveClassSentenceLandmarks").mockResolvedValue({
      success: true,
      status: "recognized",
      predicted_sentence: "are you free today",
      confidence: 0.942,
      tokens: ["YOU", "FREE", "TODAY"],
      top_predictions: [
        { sentence: "are you free today", confidence: 0.942 },
      ],
    });
  });

  it("1. Initializes in Word Recognition mode with correct status banner and isolated controls", () => {
    render(<LiveRecognitionPage />);

    // Verify Active Mode Status Banner
    const statusBanner = screen.getByTestId("active-mode-status");
    expect(statusBanner.textContent).toBe("Word Recognition: Individual sign detection active.");

    // Verify Mode buttons
    const isolatedBtn = screen.getByTestId("mode-isolated-btn");
    const sentenceBtn = screen.getByTestId("mode-sentence-btn");
    expect(isolatedBtn.getAttribute("aria-selected")).toBe("true");
    expect(sentenceBtn.getAttribute("aria-selected")).toBe("false");

    // Continuous sentence controls should NOT be visible
    expect(screen.queryByTestId("sentence-actions-bar")).toBeNull();
    expect(screen.queryByTestId("sentence-mode-badge")).toBeNull();
    expect(screen.queryByTestId("sentence-communication-result-card")).toBeNull();
  });

  it("2. Switching to Continuous Sentence Recognition mode displays sentence status and hides word UI", async () => {
    render(<LiveRecognitionPage />);

    const sentenceBtn = screen.getByTestId("mode-sentence-btn");
    fireEvent.click(sentenceBtn);

    // Verify Active Mode Status Banner updated
    const statusBanner = screen.getByTestId("active-mode-status");
    expect(statusBanner.textContent).toBe(
      "Continuous Sentence Recognition: Full sentence detection active."
    );

    // Verify button aria-selected states
    expect(sentenceBtn.getAttribute("aria-selected")).toBe("true");
    expect(screen.getByTestId("mode-isolated-btn").getAttribute("aria-selected")).toBe("false");

    // Continuous sentence controls MUST be visible
    expect(screen.getByTestId("sentence-actions-bar")).toBeDefined();
    expect(screen.getByTestId("sentence-mode-badge")).toBeDefined();

    // Isolated sign panels must NOT be visible
    expect(screen.queryByTestId("uncertain-suggestion-card")).toBeNull();
    expect(screen.queryByTestId("sign-communication-result-card")).toBeNull();
  });

  it("3. Continuous Sentence Recognition performs sentence inference and displays only sentence result", async () => {
    render(<LiveRecognitionPage />);

    // Switch to continuous sentence recognition
    fireEvent.click(screen.getByTestId("mode-sentence-btn"));

    // Trigger sentence record
    const recordBtn = screen.getByTestId("record-sentence-btn");
    fireEvent.click(recordBtn);

    // Wait for sentence result
    await waitFor(() => {
      expect(screen.getByTestId("sentence-communication-result-card")).toBeDefined();
      expect(screen.getByTestId("recognized-sentence-text").textContent).toBe("are you free today");
    });

    // Word recognition results MUST NOT be displayed
    expect(screen.queryByTestId("sign-communication-result-card")).toBeNull();
    expect(screen.queryByTestId("uncertain-suggestion-card")).toBeNull();
  });

  it("4. Switching back from Continuous Sentence to Word Recognition clears sentence result and resets UI", async () => {
    render(<LiveRecognitionPage />);

    // Switch to continuous mode and run inference
    fireEvent.click(screen.getByTestId("mode-sentence-btn"));
    fireEvent.click(screen.getByTestId("record-sentence-btn"));

    await waitFor(() => {
      expect(screen.getByTestId("sentence-communication-result-card")).toBeDefined();
    });

    // Switch back to word mode
    fireEvent.click(screen.getByTestId("mode-isolated-btn"));

    // Verify Status Banner
    expect(screen.getByTestId("active-mode-status").textContent).toBe(
      "Word Recognition: Individual sign detection active."
    );

    // Sentence card and actions bar MUST be gone immediately
    expect(screen.queryByTestId("sentence-communication-result-card")).toBeNull();
    expect(screen.queryByTestId("sentence-actions-bar")).toBeNull();
    expect(screen.queryByTestId("sentence-mode-badge")).toBeNull();
  });

  it("5. Asynchronous responses from previous mode are ignored after mode switch", async () => {
    let delayedResolve;
    api.recognizeSentenceLandmarks.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          delayedResolve = resolve;
        })
    );

    render(<LiveRecognitionPage />);

    // Start in continuous mode
    fireEvent.click(screen.getByTestId("mode-sentence-btn"));
    fireEvent.click(screen.getByTestId("record-sentence-btn"));

    // Switch to word mode while request is in flight
    fireEvent.click(screen.getByTestId("mode-isolated-btn"));

    // Late resolve of previous mode request
    if (delayedResolve) {
      delayedResolve({
        success: true,
        status: "recognized",
        sentence: "late response sentence",
        confidence: 0.99,
        gloss_tokens: ["LATE"],
        top_predictions: [{ sentence: "late response sentence", confidence: 0.99 }],
      });
    }

    // Verify word mode remains active and late sentence prediction did not render
    await waitFor(() => {
      expect(screen.getByTestId("active-mode-status").textContent).toBe(
        "Word Recognition: Individual sign detection active."
      );
    });

    expect(screen.queryByTestId("sentence-communication-result-card")).toBeNull();
    expect(screen.queryByText("late response sentence")).toBeNull();
  });
});
