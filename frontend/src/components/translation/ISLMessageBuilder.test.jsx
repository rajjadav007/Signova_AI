import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import ISLMessageBuilder from "./ISLMessageBuilder";
import * as api from "../../services/api";

describe("ISLMessageBuilder Component (FE-08)", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.spyOn(api, "fetchSupportedLanguages").mockResolvedValue({
      "en-IN": "English (Indian)",
      "hi-IN": "Hindi (हिन्दी)",
      "gu-IN": "Gujarati (ગુજરાતી)",
      "kn-IN": "Kannada (ಕನ್ನಡ)",
      "ta-IN": "Tamil (தமிழ்)",
    });
    vi.spyOn(api, "recognizeIsolatedVideo").mockResolvedValue({
      success: true,
      status: "recognized",
      is_valid: true,
      confidence: 0.95,
      rejection_threshold: 0.80,
      predicted_sign: "hello",
      display_name: "Hello",
    });
    vi.spyOn(api, "translateText").mockResolvedValue({
      success: true,
      translated_text: "नमस्ते",
      target_language_code: "hi-IN",
      source_language_code: "en-IN",
      processing_time_ms: 100,
    });
    vi.spyOn(api, "synthesizeSpeech").mockResolvedValue({
      success: true,
      audio_base64: "UklGRiQAAABXQVZFZm10IBAAAAABAAEAESsAACJWAAACABAAZGF0YQAAAAA=",
      audio_format: "wav",
      target_language_code: "hi-IN",
      speaker: "meera",
      processing_time_ms: 200,
    });

    // Mock HTMLMediaElement prototype
    window.HTMLMediaElement.prototype.play = vi.fn().mockResolvedValue(undefined);
    window.HTMLMediaElement.prototype.pause = vi.fn();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders message builder layout with initial empty state and controls", () => {
    render(<ISLMessageBuilder />);

    expect(screen.getByText(/FE-08 Complete Direction:/i)).toBeDefined();
    expect(screen.getByText(/1\. ISL Gesture Recognition/i)).toBeDefined();
    expect(screen.getByText(/2\. Message Builder/i)).toBeDefined();
    expect(screen.getByText(/3\. Language Selection & Translation/i)).toBeDefined();
    expect(screen.getByText(/4\. Spoken Audio/i)).toBeDefined();
    expect(screen.getByPlaceholderText(/Editable sign message appears here\.\.\./i)).toBeDefined();
  });

  it("adds recognized word to sequence and preserves order", () => {
    render(<ISLMessageBuilder />);

    // Click quick-add buttons: Hello, Water, Please
    fireEvent.click(screen.getByTestId("quick-add-hello"));
    fireEvent.click(screen.getByTestId("quick-add-water"));
    fireEvent.click(screen.getByTestId("quick-add-please"));

    expect(screen.getByTestId("word-chip-0").textContent).toContain("Hello");
    expect(screen.getByTestId("word-chip-1").textContent).toContain("Water");
    expect(screen.getByTestId("word-chip-2").textContent).toContain("Please");

    const textarea = screen.getByTestId("editable-message-input");
    expect(textarea.value).toBe("Hello Water Please");
  });

  it("allows removing individual words without clearing entire message", () => {
    render(<ISLMessageBuilder />);

    fireEvent.click(screen.getByTestId("quick-add-food"));
    fireEvent.click(screen.getByTestId("quick-add-water"));
    fireEvent.click(screen.getByTestId("quick-add-yes"));

    expect(screen.getByTestId("editable-message-input").value).toBe("Food Water Yes");

    // Remove middle word "Water" (index 1)
    fireEvent.click(screen.getByTestId("remove-word-1"));

    expect(screen.queryByTestId("word-chip-2")).toBeNull();
    expect(screen.getByTestId("word-chip-0").textContent).toContain("Food");
    expect(screen.getByTestId("word-chip-1").textContent).toContain("Yes");
    expect(screen.getByTestId("editable-message-input").value).toBe("Food Yes");
  });

  it("clears entire message when Clear All is clicked", () => {
    render(<ISLMessageBuilder />);

    fireEvent.click(screen.getByTestId("quick-add-help"));
    fireEvent.click(screen.getByTestId("quick-add-stop"));
    expect(screen.getByTestId("editable-message-input").value).toBe("Help Stop");

    fireEvent.click(screen.getByTestId("clear-message-btn"));

    expect(screen.queryByTestId("word-chip-0")).toBeNull();
    expect(screen.getByTestId("editable-message-input").value).toBe("");
  });

  it("allows freehand manual editing of the message text", () => {
    render(<ISLMessageBuilder />);

    fireEvent.click(screen.getByTestId("quick-add-hello"));
    const textarea = screen.getByTestId("editable-message-input");

    fireEvent.change(textarea, { target: { value: "Hello, could you please help me find water?" } });

    expect(textarea.value).toBe("Hello, could you please help me find water?");
  });

  it("valid recognition response from webcam adds word, while rejected sign is ignored", async () => {
    render(<ISLMessageBuilder />);

    // 1. Valid recognition (confidence 0.95 >= 0.80)
    vi.spyOn(api, "recognizeIsolatedVideo").mockResolvedValueOnce({
      success: true,
      status: "recognized",
      is_valid: true,
      confidence: 0.95,
      rejection_threshold: 0.80,
      predicted_sign: "thank you",
      display_name: "Thank you",
    });

    const fileInput = document.querySelector('input[type="file"]');
    const fakeFile1 = new File(["video_bytes"], "thank_you.webm", { type: "video/webm" });
    fireEvent.change(fileInput, { target: { files: [fakeFile1] } });

    await waitFor(() => {
      expect(screen.getByTestId("editable-message-input").value).toBe("Thank you");
    });

    // 2. Uncertain / rejected recognition (confidence 0.45 < 0.80)
    vi.spyOn(api, "recognizeIsolatedVideo").mockResolvedValueOnce({
      success: true,
      status: "uncertain",
      is_valid: false,
      confidence: 0.45,
      rejection_threshold: 0.80,
      predicted_sign: "water",
      display_name: "Water",
    });

    const fakeFile2 = new File(["video_bytes_2"], "unclear.webm", { type: "video/webm" });
    fireEvent.change(fileInput, { target: { files: [fakeFile2] } });

    await waitFor(() => {
      // Must NOT add the rejected sign!
      expect(screen.getByTestId("editable-message-input").value).toBe("Thank you");
      expect(screen.getByText(/REJECTED \(UNCERTAIN\)/i)).toBeDefined();
    });
  });

  it("handles target language selection and successful translation", async () => {
    vi.spyOn(api, "translateText").mockResolvedValueOnce({
      success: true,
      translated_text: "नमस्ते, क्या आप मेरी मदद कर सकते हैं?",
      target_language_code: "hi-IN",
      source_language_code: "en-IN",
      processing_time_ms: 120,
    });

    render(<ISLMessageBuilder />);

    fireEvent.click(screen.getByTestId("quick-add-hello"));
    fireEvent.click(screen.getByTestId("quick-add-help"));

    const langSelect = screen.getByTestId("target-language-select");
    fireEvent.change(langSelect, { target: { value: "hi-IN" } });
    expect(langSelect.value).toBe("hi-IN");

    const translateBtn = screen.getByTestId("translate-btn");
    fireEvent.click(translateBtn);

    await waitFor(() => {
      expect(api.translateText).toHaveBeenCalledWith("Hello Help", "hi-IN", "en-IN");
      expect(screen.getByTestId("translated-text-display").textContent).toContain(
        "नमस्ते, क्या आप मेरी मदद कर सकते हैं?"
      );
    });
  });

  it("handles translation API failure without crashing", async () => {
    vi.spyOn(api, "translateText").mockRejectedValueOnce(
      new Error("Sarvam AI service is not configured. Please set SARVAM_API_KEY in the backend.")
    );

    render(<ISLMessageBuilder />);

    fireEvent.click(screen.getByTestId("quick-add-food"));
    fireEvent.click(screen.getByTestId("translate-btn"));

    await waitFor(() => {
      expect(screen.getByText(/Translation Error:/i)).toBeDefined();
      expect(
        screen.getByText(/Sarvam AI service is not configured/i)
      ).toBeDefined();
    });
  });

  it("handles speech generation and audio playback controls", async () => {
    vi.spyOn(api, "synthesizeSpeech").mockResolvedValueOnce({
      success: true,
      audio_base64: "UklGRiQAAABXQVZFZm10IBAAAAABAAEAESsAACJWAAACABAAZGF0YQAAAAA=",
      audio_format: "wav",
      target_language_code: "hi-IN",
      speaker: "meera",
      processing_time_ms: 250,
    });

    render(<ISLMessageBuilder />);

    // Add word and manually trigger speech
    fireEvent.click(screen.getByTestId("quick-add-water"));
    const generateSpeechBtn = screen.getByTestId("generate-speech-btn");
    fireEvent.click(generateSpeechBtn);

    await waitFor(() => {
      expect(api.synthesizeSpeech).toHaveBeenCalledWith("Water", "hi-IN", "meera", 1.0);
      expect(screen.getByTestId("replay-audio-btn")).toBeDefined();
    });

    // Test replay click
    fireEvent.click(screen.getByTestId("replay-audio-btn"));
    expect(window.HTMLMediaElement.prototype.play).toHaveBeenCalled();
  });

  it("handles speech synthesis failure gracefully", async () => {
    vi.spyOn(api, "synthesizeSpeech").mockRejectedValueOnce(
      new Error("Failed to generate speech: upstream rate limit.")
    );

    render(<ISLMessageBuilder />);

    fireEvent.click(screen.getByTestId("quick-add-yes"));
    fireEvent.click(screen.getByTestId("generate-speech-btn"));

    await waitFor(() => {
      expect(screen.getByText(/upstream rate limit/i)).toBeDefined();
    });
  });

  it("renders in embedded mode with hideCameraPreview={true} without video element", () => {
    const { container } = render(<ISLMessageBuilder hideCameraPreview={true} />);

    // Video element should not be rendered
    const video = container.querySelector("video");
    expect(video).toBeNull();

    // Message builder and quick add buttons remain functional
    expect(screen.getByTestId("quick-add-hello")).toBeDefined();
    expect(screen.getByTestId("editable-message-input")).toBeDefined();
    expect(screen.getByTestId("translate-btn")).toBeDefined();
    expect(screen.getByTestId("generate-speech-btn")).toBeDefined();
  });

  it("receives externalRecognizedSign from live camera and appends word to message", async () => {
    const { rerender } = render(
      <ISLMessageBuilder hideCameraPreview={true} externalRecognizedSign={null} />
    );

    expect(screen.getByTestId("editable-message-input").value).toBe("");

    rerender(
      <ISLMessageBuilder
        hideCameraPreview={true}
        externalRecognizedSign={{ word: "Hello", timestamp: 123456789 }}
      />
    );

    await waitFor(() => {
      expect(screen.getByTestId("editable-message-input").value).toBe("Hello");
      expect(screen.getByTestId("word-chip-0").textContent).toContain("Hello");
    });
  });

  it("displays amber suggestion banner for uncertain detection, never auto-adds, and allows user to add or dismiss", async () => {
    const { rerender } = render(
      <ISLMessageBuilder hideCameraPreview={true} externalUncertainSuggestion={null} />
    );

    expect(screen.queryByTestId("uncertain-suggestion-banner")).toBeNull();

    // Low confidence prediction arrives
    rerender(
      <ISLMessageBuilder
        hideCameraPreview={true}
        externalUncertainSuggestion={{ word: "Water", timestamp: 987654321 }}
      />
    );

    await waitFor(() => {
      const banner = screen.getByTestId("uncertain-suggestion-banner");
      expect(banner).toBeDefined();
      expect(banner.textContent).toContain("Uncertain detection:");
      expect(banner.textContent).toContain("Water");
    });

    // Word should NOT have been added automatically
    expect(screen.getByTestId("editable-message-input").value).toBe("");

    // User clicks "Add to Message"
    const addBtn = screen.getByTestId("add-uncertain-suggestion-btn");
    fireEvent.click(addBtn);

    // Now word is added and banner is dismissed
    expect(screen.getByTestId("editable-message-input").value).toBe("Water");
    expect(screen.queryByTestId("uncertain-suggestion-banner")).toBeNull();
  });
});
