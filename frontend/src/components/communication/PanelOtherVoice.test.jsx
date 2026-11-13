import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import React from "react";
import PanelOtherVoice from "./PanelOtherVoice";
import * as api from "../../services/api";

// Mock mediaDevices
if (!navigator.mediaDevices) {
  navigator.mediaDevices = {};
}
const mockAudioTrack = { stop: vi.fn(), kind: "audio", readyState: "live" };

navigator.mediaDevices.getUserMedia = vi.fn().mockImplementation((constraints) => {
  return Promise.resolve({
    getTracks: () => [mockAudioTrack],
  });
});

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
let mockAudioInstances = [];
window.Audio = vi.fn().mockImplementation((url) => {
  const instance = {
    url,
    play: vi.fn().mockResolvedValue(undefined),
    pause: vi.fn(),
    currentTime: 0,
    onended: null,
    onerror: null,
  };
  mockAudioInstances.push(instance);
  return instance;
});

const recordAndStopSpeech = async () => {
  const micBtn = screen.getByTestId("mic-toggle-btn");
  fireEvent.click(micBtn);
  await waitFor(() => {
    expect(screen.getByTestId("mic-status-badge").textContent).toContain("Listening");
  });
  fireEvent.click(micBtn);
};

describe("PanelOtherVoice - Automatic Translation Audio Playback", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockAudioInstances = [];

    vi.spyOn(api, "transcribeAndTranslateSpeech").mockResolvedValue({
      success: true,
      original_transcript: "हाँ जी बताइए क्या काम है",
      translated_text: "હા, કહો, શું કામ છે?",
      source_language_code: "hi-IN",
      target_language_code: "gu-IN",
    });

    vi.spyOn(api, "synthesizeSpeech").mockResolvedValue({
      success: true,
      audio_base64: "MOCK_BASE64_GUJARATI_AUDIO",
      audio_format: "wav",
      target_language_code: "gu-IN",
      speaker: "priya",
    });
  });

  it("1. Successful Hindi-to-Gujarati translation triggers automatic Sarvam Bulbul TTS and plays audio", async () => {
    render(
      <PanelOtherVoice
        otherLanguage="hi-IN"
        otherLanguageName="Hindi"
        yourLanguage="gu-IN"
        yourLanguageName="Gujarati"
      />
    );

    await recordAndStopSpeech();

    // Wait for translation to render
    await waitFor(() => {
      expect(screen.getByTestId("readable-regional-text").textContent).toBe("હા, કહો, શું કામ છે?");
    });

    // Verify Sarvam Bulbul TTS was automatically triggered with translated Gujarati text
    await waitFor(() => {
      expect(api.synthesizeSpeech).toHaveBeenCalledTimes(1);
      expect(api.synthesizeSpeech).toHaveBeenCalledWith(
        "હા, કહો, શું કામ છે?",
        "gu-IN",
        "priya"
      );
    });

    // Verify audio playback was initiated with generated base64 audio
    expect(mockAudioInstances.length).toBeGreaterThanOrEqual(1);
    const audio = mockAudioInstances[mockAudioInstances.length - 1];
    expect(audio.url).toContain("data:audio/wav;base64,MOCK_BASE64_GUJARATI_AUDIO");
    expect(audio.play).toHaveBeenCalled();
  });

  it("2. Successful English-to-Hindi translation triggers automatic Sarvam Bulbul TTS in Hindi", async () => {
    vi.spyOn(api, "transcribeAndTranslateSpeech").mockResolvedValue({
      success: true,
      original_transcript: "Hello how are you",
      translated_text: "नमस्ते आप कैसे हैं",
      source_language_code: "en-IN",
      target_language_code: "hi-IN",
    });

    vi.spyOn(api, "synthesizeSpeech").mockResolvedValue({
      success: true,
      audio_base64: "MOCK_BASE64_HINDI_AUDIO",
      audio_format: "wav",
      target_language_code: "hi-IN",
      speaker: "priya",
    });

    render(
      <PanelOtherVoice
        otherLanguage="en-IN"
        otherLanguageName="English"
        yourLanguage="hi-IN"
        yourLanguageName="Hindi"
      />
    );

    await recordAndStopSpeech();

    await waitFor(() => {
      expect(screen.getByTestId("readable-regional-text").textContent).toBe("नमस्ते आप कैसे हैं");
    });

    await waitFor(() => {
      expect(api.synthesizeSpeech).toHaveBeenCalledWith(
        "नमस्ते आप कैसे हैं",
        "hi-IN",
        "priya"
      );
    });

    const audio = mockAudioInstances[mockAudioInstances.length - 1];
    expect(audio.url).toContain("data:audio/wav;base64,MOCK_BASE64_HINDI_AUDIO");
    expect(audio.play).toHaveBeenCalled();
  });

  it("3. When Auto Play Audio setting is disabled, shows translated text without automatically playing audio", async () => {
    render(
      <PanelOtherVoice
        otherLanguage="hi-IN"
        otherLanguageName="Hindi"
        yourLanguage="gu-IN"
        yourLanguageName="Gujarati"
        defaultAutoPlayAudio={true}
      />
    );

    // Disable Auto Play Audio setting via toggle
    const autoPlayToggle = screen.getByTestId("auto-play-audio-toggle");
    expect(autoPlayToggle.checked).toBe(true);
    fireEvent.click(autoPlayToggle);
    expect(autoPlayToggle.checked).toBe(false);

    await recordAndStopSpeech();

    await waitFor(() => {
      expect(screen.getByTestId("readable-regional-text").textContent).toBe("હા, કહો, શું કામ છે?");
    });

    // Verify TTS was NOT automatically called
    expect(api.synthesizeSpeech).not.toHaveBeenCalled();
    expect(mockAudioInstances.length).toBe(0);

    // Manual Listen button is still available
    const listenBtn = screen.getByTestId("optional-audio-btn");
    expect(listenBtn).toBeDefined();
  });

  it("4. Re-rendering component does not cause repeated duplicate audio playback", async () => {
    const { rerender } = render(
      <PanelOtherVoice
        otherLanguage="hi-IN"
        otherLanguageName="Hindi"
        yourLanguage="gu-IN"
        yourLanguageName="Gujarati"
      />
    );

    await recordAndStopSpeech();

    await waitFor(() => {
      expect(api.synthesizeSpeech).toHaveBeenCalledTimes(1);
    });

    // Re-render multiple times
    rerender(
      <PanelOtherVoice
        otherLanguage="hi-IN"
        otherLanguageName="Hindi"
        yourLanguage="gu-IN"
        yourLanguageName="Gujarati"
      />
    );
    rerender(
      <PanelOtherVoice
        otherLanguage="hi-IN"
        otherLanguageName="Hindi"
        yourLanguage="gu-IN"
        yourLanguageName="Gujarati"
      />
    );

    // Verify synthesizeSpeech was not repeatedly invoked by re-renders
    expect(api.synthesizeSpeech).toHaveBeenCalledTimes(1);
  });

  it("5. Browser autoplay blocked displays 'Tap to play translated audio.' and fallback 'Play Audio' button", async () => {
    // Simulate browser blocking autoplay (play() rejects with NotAllowedError)
    window.Audio = vi.fn().mockImplementation((url) => {
      const err = new Error("The play() request was interrupted or prevented by browser policy.");
      err.name = "NotAllowedError";
      const instance = {
        url,
        play: vi.fn().mockRejectedValue(err),
        pause: vi.fn(),
        currentTime: 0,
        onended: null,
        onerror: null,
      };
      mockAudioInstances.push(instance);
      return instance;
    });

    render(
      <PanelOtherVoice
        otherLanguage="hi-IN"
        otherLanguageName="Hindi"
        yourLanguage="gu-IN"
        yourLanguageName="Gujarati"
      />
    );

    await recordAndStopSpeech();

    await waitFor(() => {
      expect(screen.getByTestId("readable-regional-text")).toBeDefined();
    });

    // Check banner and fallback button appear
    await waitFor(() => {
      expect(screen.getByTestId("autoplay-blocked-banner")).toBeDefined();
      expect(screen.getByText("Tap to play translated audio.")).toBeDefined();
    });

    const fallbackBtn = screen.getByTestId("unblock-voice-audio-btn");
    expect(fallbackBtn).toBeDefined();
    expect(fallbackBtn.textContent).toContain("Play Audio");

    // Listen button should NOT falsely show playing state
    const listenBtn = screen.getByTestId("optional-audio-btn");
    expect(listenBtn.className).not.toContain("playing");

    // Clicking fallback button allows user-gesture playback of cached audio
    window.Audio = vi.fn().mockImplementation((url) => {
      const instance = {
        url,
        play: vi.fn().mockResolvedValue(undefined),
        pause: vi.fn(),
        currentTime: 0,
        onended: null,
        onerror: null,
      };
      mockAudioInstances.push(instance);
      return instance;
    });

    fireEvent.click(fallbackBtn);

    // Verify it reuses cached audio without re-calling Sarvam TTS API
    expect(api.synthesizeSpeech).toHaveBeenCalledTimes(1);
  });

  it("6. Handles Sarvam Bulbul TTS API failure gracefully and allows retry via Listen button", async () => {
    api.synthesizeSpeech.mockRejectedValueOnce(
      new Error("Sarvam TTS service unavailable (503): Insufficient credits")
    );

    render(
      <PanelOtherVoice
        otherLanguage="hi-IN"
        otherLanguageName="Hindi"
        yourLanguage="gu-IN"
        yourLanguageName="Gujarati"
      />
    );

    await recordAndStopSpeech();

    // Translation should still be displayed
    await waitFor(() => {
      expect(screen.getByTestId("readable-regional-text").textContent).toBe("હા, કહો, શું કામ છે?");
    });

    // Error banner should display TTS failure
    await waitFor(() => {
      expect(screen.getByTestId("mic-error-banner")).toBeDefined();
      expect(screen.getByTestId("mic-error-banner").textContent).toContain("Insufficient credits");
    });

    // Mock succeeding for retry
    api.synthesizeSpeech.mockResolvedValueOnce({
      success: true,
      audio_base64: "RETRY_AUDIO_BASE64",
      audio_format: "wav",
      target_language_code: "gu-IN",
      speaker: "priya",
    });

    // Manual replay / retry via Listen button
    const listenBtn = screen.getByTestId("optional-audio-btn");
    fireEvent.click(listenBtn);

    await waitFor(() => {
      expect(api.synthesizeSpeech).toHaveBeenCalledTimes(2);
    });
  });

  it("7. Manual replay using Listen button reuses cached audio and stops audio when playing", async () => {
    render(
      <PanelOtherVoice
        otherLanguage="hi-IN"
        otherLanguageName="Hindi"
        yourLanguage="gu-IN"
        yourLanguageName="Gujarati"
      />
    );

    await recordAndStopSpeech();

    await waitFor(() => {
      expect(api.synthesizeSpeech).toHaveBeenCalledTimes(1);
    });

    // Click Listen for manual replay
    const listenBtn = screen.getByTestId("optional-audio-btn");
    fireEvent.click(listenBtn);

    // Replay should NOT make another TTS network call (uses cached audio)
    expect(api.synthesizeSpeech).toHaveBeenCalledTimes(1);

    // If audio is playing, clicking Listen button triggers stopAudio()
    const activeAudio = mockAudioInstances[mockAudioInstances.length - 1];
    fireEvent.click(listenBtn); // Stop
    expect(activeAudio.pause).toHaveBeenCalled();
  });

  it("8. Stops previous audio when a new translation arrives", async () => {
    render(
      <PanelOtherVoice
        otherLanguage="hi-IN"
        otherLanguageName="Hindi"
        yourLanguage="gu-IN"
        yourLanguageName="Gujarati"
      />
    );

    // First translation
    await recordAndStopSpeech();

    await waitFor(() => {
      expect(api.synthesizeSpeech).toHaveBeenCalledTimes(1);
    });
    const firstAudio = mockAudioInstances[mockAudioInstances.length - 1];

    // Second translation arrives
    vi.spyOn(api, "transcribeAndTranslateSpeech").mockResolvedValueOnce({
      success: true,
      original_transcript: "धन्यवाद",
      translated_text: "આભાર",
      source_language_code: "hi-IN",
      target_language_code: "gu-IN",
    });

    await recordAndStopSpeech();

    await waitFor(() => {
      expect(screen.getByTestId("readable-regional-text").textContent).toBe("આભાર");
    });

    // Previous audio should have been paused
    expect(firstAudio.pause).toHaveBeenCalled();

    // New translation audio generated
    await waitFor(() => {
      expect(api.synthesizeSpeech).toHaveBeenCalledWith("આભાર", "gu-IN", "priya");
    });
  });
});
