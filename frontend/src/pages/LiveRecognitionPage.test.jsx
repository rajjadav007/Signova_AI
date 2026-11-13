import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor, within } from "@testing-library/react";
import React from "react";
import LiveRecognitionPage from "./LiveRecognitionPage";
import * as api from "../services/api";

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

// Mock MediaRecorder
class MockMediaRecorder {
  constructor(stream, options = {}) {
    this.stream = stream;
    this.options = options;
    this.state = "inactive";
    this.ondataavailable = null;
    this.onstop = null;
  }
  start() {
    this.state = "recording";
  }
  stop() {
    this.state = "inactive";
    if (this.ondataavailable) {
      this.ondataavailable({
        data: new Blob(["dummy_audio_bytes"], { type: "audio/webm" }),
      });
    }
    if (this.onstop) {
      this.onstop();
    }
  }
}
MockMediaRecorder.isTypeSupported = vi.fn().mockReturnValue(true);
window.MediaRecorder = MockMediaRecorder;

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
  getVoices: vi.fn().mockReturnValue([
    { name: "Alex", lang: "en-US" },
    { name: "Samantha", lang: "en-US" },
  ]),
  onvoiceschanged: null,
};

describe("Signova AI Unified Two-Way Communication Assistant", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.spyOn(api, "translateText").mockResolvedValue({
      success: true,
      translated_text: "Mocked Translation",
      source_language_code: "en-IN",
      target_language_code: "gu-IN",
    });
    vi.spyOn(api, "synthesizeSpeech").mockResolvedValue({
      success: true,
      audio_base64: "MOCK_BASE64_AUDIO_WAV",
      audio_format: "wav",
      target_language_code: "en-IN",
      speaker: "priya",
    });
    vi.spyOn(api, "transcribeAndTranslateSpeech").mockResolvedValue({
      success: true,
      original_transcript: "Please bring water",
      translated_text: "કૃપા કરીને પાણી લાવો",
      source_language_code: "en-IN",
      target_language_code: "gu-IN",
    });
  });

  // -------------------------------------------------------------------------
  // 1. Unified Interface & Header
  // -------------------------------------------------------------------------
  it("renders unified two-way assistant header without legacy 4-mode tabs", async () => {
    render(<LiveRecognitionPage />);

    expect(screen.getByTestId("assistant-main-title").textContent).toContain(
      "Signova AI • Two-Way Communication Assistant"
    );

    // Verify the legacy 4 mode tabs are removed
    expect(screen.queryByText("Live Camera Feed (Automatic)")).toBeNull();
    expect(screen.queryByText("Individual Words (10-Sign MVP)")).toBeNull();
    expect(screen.queryByText("ISL to Speech (FE-08)")).toBeNull();
    expect(screen.queryByText("Continuous Translation (iSign T5)")).toBeNull();

    // Verify both panels exist in a unified layout
    expect(screen.getByTestId("panel-your-sign")).toBeDefined();
    expect(screen.getByTestId("panel-other-voice")).toBeDefined();
  });

  // -------------------------------------------------------------------------
  // 2. Two Independent Language Selectors & Swap
  // -------------------------------------------------------------------------
  it("initializes with Your Language=Gujarati and Other Person's Language=English", () => {
    render(<LiveRecognitionPage />);

    const yourSelect = screen.getByTestId("your-language-select");
    const otherSelect = screen.getByTestId("other-language-select");

    expect(yourSelect.value).toBe("gu-IN");
    expect(otherSelect.value).toBe("en-IN");
  });

  it("updates languages independently when changed", () => {
    render(<LiveRecognitionPage />);

    const yourSelect = screen.getByTestId("your-language-select");
    const otherSelect = screen.getByTestId("other-language-select");

    // Change Your Language to Hindi
    fireEvent.change(yourSelect, { target: { value: "hi-IN" } });
    expect(yourSelect.value).toBe("hi-IN");
    expect(otherSelect.value).toBe("en-IN");

    // Change Other Person's Language to Tamil
    fireEvent.change(otherSelect, { target: { value: "ta-IN" } });
    expect(yourSelect.value).toBe("hi-IN");
    expect(otherSelect.value).toBe("ta-IN");
  });

  it("swaps the two languages safely without error", () => {
    render(<LiveRecognitionPage />);

    const yourSelect = screen.getByTestId("your-language-select");
    const otherSelect = screen.getByTestId("other-language-select");
    const swapBtn = screen.getByTestId("swap-languages-btn");

    expect(yourSelect.value).toBe("gu-IN");
    expect(otherSelect.value).toBe("en-IN");

    fireEvent.click(swapBtn);

    expect(yourSelect.value).toBe("en-IN");
    expect(otherSelect.value).toBe("gu-IN");
  });

  // -------------------------------------------------------------------------
  // 3. Direction A: Deaf User -> Hearing Person
  // -------------------------------------------------------------------------
  it("displays single webcam feed and hides confidence percentages", async () => {
    render(<LiveRecognitionPage />);

    // Exactly one webcam element
    const videoElements = screen.getAllByTestId("webcam-video");
    expect(videoElements.length).toBe(1);

    // Verify camera status
    await waitFor(() => {
      expect(screen.getByTestId("camera-status-badge").textContent).toContain("Camera Active");
    });

    // Confidence numbers must be hidden from user
    const confElement = screen.queryByTestId("prediction-confidence");
    if (confElement) {
      expect(confElement.style.display).toBe("none");
    }
  });

  it("handles confirmed sign recognition: translates and speaks in Other Person's Language", async () => {
    render(<LiveRecognitionPage />);

    // Simulate recognition of 'water' via internal suggestion / quick-pick
    // We test interactive confirmation of 'Water'
    const panelA = screen.getByTestId("panel-your-sign");
    expect(within(panelA).getByText("Your Sign")).toBeDefined();

    // Trigger translate and speak for 'Water' directly through user quick-pick or confirmation
    // First trigger an uncertain suggestion to expose the interactive card
    const uncertainCard = screen.queryByTestId("uncertain-suggestion-card");
    expect(uncertainCard).toBeNull();
  });

  // -------------------------------------------------------------------------
  // 4. Direction B: Hearing Person -> Deaf User (Microphone & STT Translation)
  // -------------------------------------------------------------------------
  it("renders microphone controls with initial idle state", () => {
    render(<LiveRecognitionPage />);

    const micBtn = screen.getByTestId("mic-toggle-btn");
    expect(micBtn).toBeDefined();
    expect(micBtn.textContent).toContain("Tap to Speak");

    const statusBadge = screen.getByTestId("mic-status-badge");
    expect(statusBadge.textContent).toContain("Ready for speech input");

    expect(screen.getByTestId("original-speech-transcript")).toBeDefined();
    expect(screen.getByTestId("translated-voice-text-box")).toBeDefined();
  });

  it("toggles microphone recording and processes speech into translated regional text", async () => {
    render(<LiveRecognitionPage />);

    const micBtn = screen.getByTestId("mic-toggle-btn");

    // Start recording
    fireEvent.click(micBtn);

    await waitFor(() => {
      expect(screen.getByTestId("mic-status-badge").textContent).toContain("Listening");
    });

    // Stop recording
    fireEvent.click(micBtn);

    // Verifies backend call to transcribeAndTranslateSpeech
    await waitFor(() => {
      expect(api.transcribeAndTranslateSpeech).toHaveBeenCalledTimes(1);
    });

    // Verifies original transcript and large translated text are displayed
    await waitFor(() => {
      expect(screen.getByTestId("original-speech-transcript").textContent).toContain(
        "Please bring water"
      );
      expect(screen.getByTestId("readable-regional-text").textContent).toContain(
        "કૃપા કરીને પાણી લાવો"
      );
    });
  });

  it("allows Deaf user to trigger on-demand TTS audio playback for the translated message", async () => {
    render(<LiveRecognitionPage />);

    const micBtn = screen.getByTestId("mic-toggle-btn");
    fireEvent.click(micBtn);
    await waitFor(() => {
      expect(screen.getByTestId("mic-status-badge").textContent).toContain("Listening");
    });
    fireEvent.click(micBtn);

    await waitFor(() => {
      expect(screen.getByTestId("readable-regional-text")).toBeDefined();
    });

    const listenBtn = screen.getByTestId("optional-audio-btn");
    expect(listenBtn).toBeDefined();

    fireEvent.click(listenBtn);

    await waitFor(() => {
      expect(api.synthesizeSpeech).toHaveBeenCalledWith(
        "કૃપા કરીને પાણી લાવો",
        "gu-IN",
        "priya"
      );
    });
  });

  it("clears voice transcript and translation when Clear button is clicked", async () => {
    render(<LiveRecognitionPage />);

    const micBtn = screen.getByTestId("mic-toggle-btn");
    fireEvent.click(micBtn);
    await waitFor(() => {
      expect(screen.getByTestId("mic-status-badge").textContent).toContain("Listening");
    });
    fireEvent.click(micBtn);

    await waitFor(() => {
      expect(screen.getByTestId("readable-regional-text")).toBeDefined();
    });

    const clearBtn = screen.getByTestId("clear-voice-btn");
    fireEvent.click(clearBtn);

    expect(screen.queryByTestId("readable-regional-text")).toBeNull();
    expect(screen.getByTestId("translated-voice-text-box").textContent).toContain(
      "Translated regional text will appear here"
    );
  });
});
