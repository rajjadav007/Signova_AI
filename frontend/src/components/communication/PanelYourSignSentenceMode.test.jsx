import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor, within } from "@testing-library/react";
import React from "react";
import PanelYourSign from "./PanelYourSign";
import LiveRecognitionPage from "../../pages/LiveRecognitionPage";
import * as api from "../../services/api";

// Mock mediaDevices
if (!navigator.mediaDevices) {
  navigator.mediaDevices = {};
}
const mockVideoTrack = { stop: vi.fn(), kind: "video", readyState: "live" };
navigator.mediaDevices.getUserMedia = vi.fn().mockResolvedValue({
  getTracks: () => [mockVideoTrack],
});
navigator.mediaDevices.enumerateDevices = vi.fn().mockResolvedValue([]);

// Mock HTMLAudioElement
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

describe("Continuous ISL Sentence Recognition Mode & Pipeline Integration", () => {
  beforeEach(() => {
    vi.restoreAllMocks();

    window.Audio = vi.fn().mockImplementation(() => ({
      play: vi.fn().mockResolvedValue(undefined),
      pause: vi.fn(),
      currentTime: 0,
      onended: null,
      onerror: null,
    }));

    window.speechSynthesis = {
      speak: vi.fn(),
      cancel: vi.fn(),
      getVoices: vi.fn().mockReturnValue([]),
      onvoiceschanged: null,
    };

    vi.spyOn(api, "translateText").mockResolvedValue({
      success: true,
      translated_text: "તમે આજે મુક્ત છો",
      source_language_code: "en-IN",
      target_language_code: "gu-IN",
    });

    vi.spyOn(api, "synthesizeSpeech").mockResolvedValue({
      success: true,
      audio_base64: "MOCK_AUDIO_BASE64_SENTENCE_WAV",
      audio_format: "wav",
      target_language_code: "gu-IN",
      speaker: "priya",
    });

    vi.spyOn(api, "recognizeSentenceLandmarks").mockResolvedValue({
      success: true,
      status: "recognized",
      sentence: "are you free today",
      confidence: 0.942,
      gloss_tokens: ["YOU", "FREE", "TODAY"],
      top_predictions: [
        { sentence: "are you free today", confidence: 0.942, gloss: "YOU FREE TODAY" },
        { sentence: "what are you doing", confidence: 0.035, gloss: "YOU DO WHAT" },
        { sentence: "can i help you", confidence: 0.012, gloss: "I HELP YOU" },
      ],
      processing_time_ms: 120,
    });

    vi.spyOn(api, "recognizeSentenceVideo").mockResolvedValue({
      success: true,
      status: "recognized",
      sentence: "are you hiding something",
      confidence: 0.915,
      gloss_tokens: ["YOU", "HIDE", "SOMETHING"],
      top_predictions: [
        { sentence: "are you hiding something", confidence: 0.915, gloss: "YOU HIDE SOMETHING" },
        { sentence: "are you free today", confidence: 0.045, gloss: "YOU FREE TODAY" },
        { sentence: "what happened", confidence: 0.02, gloss: "HAPPEN WHAT" },
      ],
      processing_time_ms: 240,
    });
  });

  // ---------------------------------------------------------------------------
  // 1. Mode Switcher on LiveRecognitionPage
  // ---------------------------------------------------------------------------
  it("renders the mode switcher bar with Isolated Signs and Continuous Sentences buttons", () => {
    render(<LiveRecognitionPage />);

    expect(screen.getByTestId("recognition-mode-switcher")).toBeDefined();
    const isolatedBtn = screen.getByTestId("mode-isolated-btn");
    const sentenceBtn = screen.getByTestId("mode-sentence-btn");

    expect(isolatedBtn.textContent).toContain("Isolated Signs");
    expect(isolatedBtn.textContent).toContain("10 Signs");
    expect(isolatedBtn.getAttribute("aria-selected")).toBe("true");

    expect(sentenceBtn.textContent).toContain("Continuous Sentences");
    expect(sentenceBtn.textContent).toContain("101 Sentences");
    expect(sentenceBtn.getAttribute("aria-selected")).toBe("false");
  });

  it("toggles mode to Continuous Sentences and updates PanelYourSign UI", async () => {
    render(<LiveRecognitionPage />);

    const sentenceBtn = screen.getByTestId("mode-sentence-btn");
    fireEvent.click(sentenceBtn);

    expect(sentenceBtn.getAttribute("aria-selected")).toBe("true");
    expect(screen.getByTestId("mode-isolated-btn").getAttribute("aria-selected")).toBe("false");

    // Continuous sentence badge and controls should now be visible in Panel A
    expect(screen.getByTestId("sentence-mode-badge").textContent).toContain(
      "Continuous Sentence Recognition (101 ISL Sentences)"
    );
    expect(screen.getByTestId("record-sentence-btn")).toBeDefined();
    expect(screen.getByTestId("sentence-upload-btn")).toBeDefined();
  });

  // ---------------------------------------------------------------------------
  // 2. PanelYourSign in Sentence Mode - Record Sentence from Webcam Buffer
  // ---------------------------------------------------------------------------
  it("records sentence from webcam buffer, triggers inference, and immediately translates & speaks", async () => {
    render(
      <PanelYourSign
        otherLanguage="gu-IN"
        otherLanguageName="Gujarati"
        recognitionMode="sentence"
      />
    );

    // Verify sentence mode badge and action buttons are present
    expect(screen.getByTestId("sentence-mode-badge")).toBeDefined();
    const recordBtn = screen.getByTestId("record-sentence-btn");
    expect(recordBtn.textContent).toContain("Record Sentence");

    // Click Record Sentence
    fireEvent.click(recordBtn);

    // Verifies backend API call to recognizeSentenceLandmarks
    await waitFor(() => {
      expect(api.recognizeSentenceLandmarks).toHaveBeenCalledTimes(1);
    });

    const callArgs = api.recognizeSentenceLandmarks.mock.calls[0];
    expect(Array.isArray(callArgs[0])).toBe(true);
    expect(callArgs[1].fps).toBe(25);
    expect(callArgs[1].top_k).toBe(3);

    // Verifies prominent English sentence is displayed
    await waitFor(() => {
      expect(screen.getByTestId("recognized-sentence-text").textContent).toBe(
        "are you free today"
      );
      expect(screen.getByTestId("sentence-confidence-score").textContent).toContain("94.2%");
    });

    // Verifies gloss tokens are displayed as chips
    const glossContainer = screen.getByTestId("sentence-gloss-tokens");
    expect(within(glossContainer).getByText("YOU")).toBeDefined();
    expect(within(glossContainer).getByText("FREE")).toBeDefined();
    expect(within(glossContainer).getByText("TODAY")).toBeDefined();

    // Verifies top 3 alternative candidate sentences are displayed
    const altContainer = screen.getByTestId("sentence-top-alternatives");
    expect(within(altContainer).getByText("what are you doing")).toBeDefined();
    expect(within(altContainer).getByText("can i help you")).toBeDefined();

    // Verifies speech and translation pipeline integration:
    // Recognized sentence immediately triggered Sarvam translation to Gujarati!
    await waitFor(() => {
      expect(api.translateText).toHaveBeenCalledWith("are you free today", "gu-IN", "en-IN");
    });

    // Spoken translation displayed
    await waitFor(() => {
      expect(screen.getByTestId("translated-sign-text").textContent).toBe("તમે આજે મુક્ત છો");
    });

    // Bulbul TTS triggered and audio played
    await waitFor(() => {
      expect(api.synthesizeSpeech).toHaveBeenCalledWith("તમે આજે મુક્ત છો", "gu-IN", "priya");
      expect(screen.getByTestId("audio-status-indicator")).toBeDefined();
    });
  });

  // ---------------------------------------------------------------------------
  // 3. PanelYourSign in Sentence Mode - Video Upload
  // ---------------------------------------------------------------------------
  it("handles video file upload, runs inference, and displays recognized sentence & alternatives", async () => {
    render(
      <PanelYourSign
        otherLanguage="gu-IN"
        otherLanguageName="Gujarati"
        recognitionMode="sentence"
      />
    );

    const fileInput = screen.getByTestId("sentence-file-input");
    const testVideoFile = new File(["dummy_mp4_bytes"], "sentence_test.mp4", {
      type: "video/mp4",
    });

    fireEvent.change(fileInput, { target: { files: [testVideoFile] } });

    await waitFor(() => {
      expect(api.recognizeSentenceVideo).toHaveBeenCalledTimes(1);
    });
    expect(api.recognizeSentenceVideo.mock.calls[0][0].name).toBe("sentence_test.mp4");

    // Verifies displayed recognition results
    await waitFor(() => {
      expect(screen.getByTestId("recognized-sentence-text").textContent).toBe(
        "are you hiding something"
      );
      expect(screen.getByTestId("sentence-confidence-score").textContent).toContain("91.5%");
    });

    const glossContainer = screen.getByTestId("sentence-gloss-tokens");
    expect(within(glossContainer).getByText("YOU")).toBeDefined();
    expect(within(glossContainer).getByText("HIDE")).toBeDefined();
    expect(within(glossContainer).getByText("SOMETHING")).toBeDefined();

    // Verifies translation called
    await waitFor(() => {
      expect(api.translateText).toHaveBeenCalledWith("are you hiding something", "gu-IN", "en-IN");
    });
  });

  // ---------------------------------------------------------------------------
  // 4. Alternative Candidate Selection
  // ---------------------------------------------------------------------------
  it("allows clicking an alternative sentence candidate to re-translate and speak", async () => {
    render(
      <PanelYourSign
        otherLanguage="gu-IN"
        otherLanguageName="Gujarati"
        recognitionMode="sentence"
      />
    );

    // Run record to get results
    fireEvent.click(screen.getByTestId("record-sentence-btn"));
    await waitFor(() => {
      expect(screen.getByTestId("recognized-sentence-text")).toBeDefined();
    });

    // Clear translation mock calls to test alternative selection
    api.translateText.mockClear();
    api.synthesizeSpeech.mockClear();

    // Click candidate #2: "what are you doing"
    const altCandidate2 = screen.getByTestId("alt-candidate-1");
    expect(altCandidate2.textContent).toContain("what are you doing");
    fireEvent.click(altCandidate2);

    await waitFor(() => {
      expect(api.translateText).toHaveBeenCalledWith("what are you doing", "gu-IN", "en-IN");
    });
  });

  // ---------------------------------------------------------------------------
  // 5. Error Handling in Sentence Mode
  // ---------------------------------------------------------------------------
  it("displays error notice when sentence landmark recognition fails", async () => {
    api.recognizeSentenceLandmarks.mockRejectedValueOnce(
      new Error("Landmark sequence model timed out")
    );

    render(
      <PanelYourSign
        otherLanguage="gu-IN"
        otherLanguageName="Gujarati"
        recognitionMode="sentence"
      />
    );

    fireEvent.click(screen.getByTestId("record-sentence-btn"));

    await waitFor(() => {
      expect(screen.getByTestId("sentence-error-notice").textContent).toContain(
        "Landmark sequence model timed out"
      );
    });
  });

  // ---------------------------------------------------------------------------
  // 6. Isolated Mode Unchanged (100% Functionality Retained)
  // ---------------------------------------------------------------------------
  it("retains isolated mode UI and does not show sentence mode controls when recognitionMode=isolated", () => {
    render(
      <PanelYourSign
        otherLanguage="en-IN"
        otherLanguageName="English"
        recognitionMode="isolated"
      />
    );

    expect(screen.queryByTestId("sentence-mode-badge")).toBeNull();
    expect(screen.queryByTestId("sentence-actions-bar")).toBeNull();
    expect(screen.queryByTestId("record-sentence-btn")).toBeNull();
    expect(screen.queryByTestId("sentence-upload-btn")).toBeNull();
    expect(screen.getByTestId("panel-your-sign")).toBeDefined();
  });
});
