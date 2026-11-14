import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import React from "react";
import LiveRecognitionPage from "./LiveRecognitionPage";
import PanelYourSign from "../components/communication/PanelYourSign";
import * as api from "../services/api";

// Mock mediaDevices
if (!navigator.mediaDevices) {
  navigator.mediaDevices = {};
}
const mockVideoTrack = { stop: vi.fn(), kind: "video", readyState: "live" };
navigator.mediaDevices.getUserMedia = vi.fn().mockResolvedValue({
  getTracks: () => [mockVideoTrack],
});
navigator.mediaDevices.enumerateDevices = vi.fn().mockResolvedValue([]);

// Mock Audio
window.Audio = vi.fn().mockImplementation(() => ({
  play: vi.fn().mockResolvedValue(undefined),
  pause: vi.fn(),
  currentTime: 0,
  onended: null,
  onerror: null,
}));

// Mock SpeechSynthesis
window.speechSynthesis = {
  speak: vi.fn(),
  cancel: vi.fn(),
  getVoices: vi.fn().mockReturnValue([]),
  onvoiceschanged: null,
};

describe("Signova AI: 25-Class Continuous ISL Sentence Recognition Frontend", () => {
  beforeEach(() => {
    vi.restoreAllMocks();

    vi.spyOn(api, "translateText").mockResolvedValue({
      success: true,
      translated_text: "તમે મુક્ત છો",
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

    vi.spyOn(api, "recognizeTwentyFiveClassSentenceLandmarks").mockResolvedValue({
      success: true,
      is_recognized: true,
      is_confident: true,
      status: "SUCCESS",
      predicted_sentence: "are you free today",
      predicted_class_id: 5,
      confidence: 0.885,
      model_identifier: "bigru_25class",
      model_used: "TwentyFiveClassBiGRUAttentionModel",
      top5_candidates: [
        { class_id: 5, sentence: "are you free today", probability: 0.885 },
        { class_id: 12, sentence: "hi how are you", probability: 0.042 },
        { class_id: 24, sentence: "what are you doing", probability: 0.028 },
        { class_id: 20, sentence: "i promise", probability: 0.021 },
        { class_id: 0, sentence: "comb your hair", probability: 0.011 },
      ],
      class_probabilities: {
        "are you free today": 0.885,
        "hi how are you": 0.042,
        "what are you doing": 0.028,
        "i promise": 0.021,
        "comb your hair": 0.011,
      },
      frame_count: 24,
      processing_time_ms: 1.1,
    });
  });

  it("1. Renders 2 selectable recognition modes in LiveRecognitionPage (Word & Unified Continuous 25-Class)", () => {
    render(<LiveRecognitionPage />);

    const isolatedBtn = screen.getByTestId("mode-isolated-btn");
    const sentenceBtn = screen.getByTestId("mode-sentence-btn");

    expect(isolatedBtn).toBeDefined();
    expect(sentenceBtn).toBeDefined();

    expect(isolatedBtn.textContent).toContain("Word Recognition");
    expect(sentenceBtn.textContent).toContain("Continuous Sentence Recognition");
    expect(sentenceBtn.textContent).toContain("25 Sentences");
  });

  it("2. Selecting Continuous Sentence mode activates unified 25-class recognition", async () => {
    render(<LiveRecognitionPage />);

    const sentenceBtn = screen.getByTestId("mode-sentence-btn");
    fireEvent.click(sentenceBtn);

    expect(sentenceBtn.getAttribute("aria-selected")).toBe("true");
    expect(screen.getByTestId("mode-isolated-btn").getAttribute("aria-selected")).toBe("false");

    expect(screen.getByTestId("heading-active-model")).toBeDefined();
    expect(screen.getByTestId("heading-active-model").textContent).toContain("25 Sentences");
    expect(screen.getByTestId("sentence-actions-bar")).toBeDefined();
  });

  it("3. PanelYourSign in 25-class mode performs inference and renders clean top prediction", async () => {
    render(
      <PanelYourSign
        otherLanguage="gu-IN"
        otherLanguageName="Gujarati"
        recognitionMode="sentence_25"
      />
    );

    expect(screen.getByTestId("sentence-mode-badge-25").textContent).toContain("25 ISL Sentences");

    const recordBtn = screen.getByTestId("record-sentence-btn");
    fireEvent.click(recordBtn);

    await waitFor(() => {
      expect(api.recognizeTwentyFiveClassSentenceLandmarks).toHaveBeenCalledTimes(1);
    });

    await waitFor(() => {
      expect(screen.getByTestId("recognized-sentence-text").textContent).toBe("are you free today");
    });

    // Verify confidence, margin, probability bars, and candidates are removed
    expect(screen.queryByTestId("sentence-confidence-score")).toBeNull();
    expect(screen.queryByTestId("sentence-top-alternatives")).toBeNull();
    expect(screen.queryByTestId("five-class-probabilities")).toBeNull();
  });

  it("4. Low-confidence predictions display actual top prediction without uncertainty banner or score", async () => {
    vi.spyOn(api, "recognizeTwentyFiveClassSentenceLandmarks").mockResolvedValueOnce({
      success: true,
      is_recognized: true,
      is_confident: false,
      status: "LOW_CONFIDENCE",
      predicted_sentence: "i am feeling bored",
      predicted_class_id: 15,
      confidence: 0.185,
      model_identifier: "bigru_25class",
      model_used: "TwentyFiveClassBiGRUAttentionModel",
      uncertainty_warning: "Uncertain prediction: Model confidence (18.5%) is below threshold (25.0%). Showing top candidate.",
      top5_candidates: [
        { class_id: 15, sentence: "i am feeling bored", probability: 0.185 },
        { class_id: 16, sentence: "i am feeling cold", probability: 0.142 },
        { class_id: 14, sentence: "i am crying", probability: 0.118 },
        { class_id: 8, sentence: "go and sleep", probability: 0.091 },
        { class_id: 17, sentence: "i am hungry", probability: 0.082 },
      ],
      class_probabilities: {},
      frame_count: 20,
      processing_time_ms: 1.0,
    });

    render(
      <PanelYourSign
        otherLanguage="gu-IN"
        otherLanguageName="Gujarati"
        recognitionMode="sentence_25"
      />
    );

    const recordBtn = screen.getByTestId("record-sentence-btn");
    fireEvent.click(recordBtn);

    await waitFor(() => {
      expect(screen.getByTestId("recognized-sentence-text").textContent).toBe("i am feeling bored");
    });

    // Verify uncertainty banner and confidence score are removed
    expect(screen.queryByTestId("sentence-confidence-score")).toBeNull();
    expect(screen.queryByTestId("sentence-low-confidence-banner")).toBeNull();
  });

  it("5. Switching modes clears buffers and previous results", async () => {
    const { rerender } = render(
      <PanelYourSign
        otherLanguage="gu-IN"
        otherLanguageName="Gujarati"
        recognitionMode="sentence_25"
      />
    );

    const recordBtn = screen.getByTestId("record-sentence-btn");
    fireEvent.click(recordBtn);

    await waitFor(() => {
      expect(screen.getByTestId("recognized-sentence-text")).toBeDefined();
    });

    // Switch to isolated word mode
    rerender(
      <PanelYourSign
        otherLanguage="gu-IN"
        otherLanguageName="Gujarati"
        recognitionMode="isolated"
      />
    );

    // Sentence result card should be cleared
    expect(screen.queryByTestId("sentence-communication-result-card")).toBeNull();
  });
});
