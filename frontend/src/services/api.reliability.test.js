import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  translateText,
  synthesizeSpeech,
  transcribeAndTranslateSpeech,
  fetchSupportedLanguages,
  recognizeIsolatedLandmarks,
  requestWithTimeout,
  _clearInflightRequests,
} from "./api";

describe("Phase 6 Reliability and Error Handling Tests", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    _clearInflightRequests();
  });

  afterEach(() => {
    vi.restoreAllMocks();
    _clearInflightRequests();
  });

  // Focus 5: Sarvam API timeout & Client timeout
  it("rejects with clean timeout error when request exceeds timeout", async () => {
    global.fetch = vi.fn().mockImplementation((url, options) => {
      return new Promise((_, reject) => {
        if (options && options.signal) {
          options.signal.addEventListener("abort", () => {
            const err = new Error("The operation was aborted");
            err.name = "AbortError";
            reject(err);
          });
        }
      });
    });

    // Request with 50ms timeout
    await expect(
      requestWithTimeout("/api/v1/test", { method: "GET" }, 50)
    ).rejects.toThrow(/timed out after/i);
  });

  // Focus 6: API credit exhaustion (HTTP 402)
  it("handles 402 credit exhaustion with clear user message", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 402,
      json: () => Promise.resolve({ detail: "Insufficient credits or account subscription expired." }),
    });

    await expect(translateText("Hello", "hi-IN", "en-IN")).rejects.toThrow(
      /credit limit exhausted/i
    );
  });

  // Focus 6 (b): Rate limit (HTTP 429)
  it("handles 429 rate limiting with clear retry advice", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 429,
      json: () => Promise.resolve({ error: { message: "Too many requests. Rate limit reached." } }),
    });

    await expect(synthesizeSpeech("Hello", "hi-IN", "priya")).rejects.toThrow(
      /rate limit exceeded/i
    );
  });

  // Focus 7: Unsupported language combination
  it("surfaces server validation errors for unsupported languages cleanly", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 400,
      json: () =>
        Promise.resolve({
          detail: "Unsupported target language 'xx-XX'. Supported: ['en-IN', 'gu-IN', 'hi-IN', 'kn-IN', 'ta-IN']",
        }),
    });

    await expect(translateText("Hello", "xx-XX", "en-IN")).rejects.toThrow(
      /unsupported target language/i
    );
  });

  // Focus 8: Network disconnection
  it("detects network failure when fetch throws TypeError", async () => {
    global.fetch = vi.fn().mockRejectedValue(new TypeError("Failed to fetch"));

    await expect(fetchSupportedLanguages()).rejects.toThrow(
      /backend server|offline|network/i
    );
  });

  // Focus 9: Empty speech transcript does not display fake success
  it("returns explicit failure when speech transcript is empty", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: () =>
        Promise.resolve({
          success: false,
          original_transcript: "",
          translated_text: "",
          error_message: "No speech detected in audio recording. Please speak clearly into the microphone.",
          source_language_code: "en-IN",
          target_language_code: "gu-IN",
          processing_time_ms: 120.0,
        }),
    });

    const dummyBlob = new Blob([new Uint8Array([0, 0, 0, 0])], { type: "audio/wav" });
    const res = await transcribeAndTranslateSpeech(dummyBlob, "audio.wav", "en-IN", "gu-IN");

    expect(res.success).toBe(false);
    expect(res.original_transcript).toBe("");
    expect(res.translated_text).toBe("");
    expect(res.error_message).toMatch(/no speech detected/i);
  });

  // Focus 10: Invalid model prediction returns rejection structure
  it("receives invalid prediction rejection reason cleanly without throwing", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: () =>
        Promise.resolve({
          success: false,
          status: "uncertain",
          is_valid: false,
          predicted_sign: "unknown",
          confidence: 0.0,
          reason: "No valid landmarks detected in input sequence",
        }),
    });

    const res = await recognizeIsolatedLandmarks([]);
    expect(res.success).toBe(false);
    expect(res.status).toBe("uncertain");
    expect(res.is_valid).toBe(false);
    expect(res.reason).toMatch(/no valid landmarks/i);
  });

  // Focus 12: In-flight duplicate request deduplication
  it("deduplicates concurrent identical translation requests", async () => {
    let callCount = 0;
    global.fetch = vi.fn().mockImplementation(() => {
      callCount++;
      return new Promise((resolve) => {
        setTimeout(() => {
          resolve({
            ok: true,
            json: () =>
              Promise.resolve({
                success: true,
                translated_text: "નમસ્તે",
                target_language_code: "gu-IN",
              }),
          });
        }, 50);
      });
    });

    // Fire two identical calls concurrently
    const p1 = translateText("Water", "gu-IN", "en-IN");
    const p2 = translateText("Water", "gu-IN", "en-IN");

    const [r1, r2] = await Promise.all([p1, p2]);

    expect(r1.translated_text).toBe("નમસ્તે");
    expect(r2.translated_text).toBe("નમસ્તે");
    // Network fetch must only have been invoked ONCE
    expect(callCount).toBe(1);
  });
});
