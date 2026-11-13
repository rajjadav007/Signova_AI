/**
 * Signova AI Frontend API Service
 * Handles communication with FastAPI backend endpoints:
 * - /api/v1/recognize/isolated/video (Isolated sign recognition)
 * - /api/v1/recognize/isolated/landmarks (Direct landmark array)
 * - /api/v1/recognize/sentence/video (Continuous ISL sentence video recognition)
 * - /api/v1/recognize/sentence/landmarks (Continuous ISL sentence landmark sequence)
 * - /api/v1/recognize/sentence/sentences (Continuous ISL 101 sentence vocabulary)
 * - /api/v1/translate (Sarvam translation - FE-08)
 * - /api/v1/translate/languages (Supported languages)
 * - /api/v1/speech/synthesize (Sarvam Bulbul TTS - FE-08)
 * - /api/v1/speech/speakers (Supported speakers)
 * - /api/v1/speech/transcribe (Sarvam Saaras STT)
 * - /api/v1/speech/transcribe-and-translate (Direction B speech transcription & translation)
 * - /api/v1/predict/video (iSign T5 translation)
 * - /api/v1/predict/landmarks (iSign T5 landmarks)
 * - /api/v1/health (system health check)
 */

const API_BASE = "/api/v1";
export const API_BASE_URL = API_BASE;
const DEFAULT_TIMEOUT_MS = 20000;

// Track active in-flight requests to prevent duplicate network calls
const inflightRequests = new Map();

export function _clearInflightRequests() {
  inflightRequests.clear();
}

/**
 * Robust fetch wrapper with timeout, network offline detection, credit limit handling, and request deduplication.
 */
export async function requestWithTimeout(url, options = {}, timeoutMs = DEFAULT_TIMEOUT_MS, dedupeKey = null) {
  if (dedupeKey && inflightRequests.has(dedupeKey)) {
    return inflightRequests.get(dedupeKey);
  }

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  const fetchPromise = (async () => {
    try {
      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
      });
      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorBody = await response.json().catch(() => ({}));
        let rawMsg =
          errorBody.error?.message ||
          errorBody.detail ||
          errorBody.message ||
          `Request failed with status ${response.status}`;

        if (response.status === 402 || /credit|quota|balance/i.test(rawMsg)) {
          rawMsg = "API credit limit exhausted. Please top up your Sarvam AI credits to continue.";
        } else if (response.status === 429 || /rate limit/i.test(rawMsg)) {
          rawMsg = "Rate limit exceeded. Please wait a moment before trying again.";
        } else if (response.status === 504 || /timed out/i.test(rawMsg)) {
          rawMsg = "Service request timed out. Please check your connection and retry.";
        } else if (response.status === 503 || /not configured|service unavailable/i.test(rawMsg)) {
          rawMsg =
            errorBody.error?.message ||
            errorBody.detail ||
            "Service is currently unavailable. Please check backend settings.";
        }
        throw new Error(rawMsg);
      }

      return await response.json();
    } catch (err) {
      clearTimeout(timeoutId);
      if (err.name === "AbortError") {
        throw new Error(
          `Request timed out after ${Math.round(timeoutMs / 1000)}s. Please check your network and retry.`
        );
      }
      if (
        err instanceof TypeError &&
        (err.message.includes("fetch") || err.message.includes("NetworkError"))
      ) {
        if (typeof navigator !== "undefined" && !navigator.onLine) {
          throw new Error("Network disconnected. You are currently offline.");
        }
        throw new Error("Unable to reach backend server. Please verify network connection and retry.");
      }
      throw err;
    } finally {
      if (dedupeKey) {
        inflightRequests.delete(dedupeKey);
      }
    }
  })();

  if (dedupeKey) {
    inflightRequests.set(dedupeKey, fetchPromise);
  }

  return fetchPromise;
}

/**
 * Send video file to backend for MediaPipe extraction and 10-class Isolated ISL recognition.
 */
export async function recognizeIsolatedVideo(fileBlob, fileName = "recording.webm", category = null, topK = 3) {
  const formData = new FormData();
  formData.append("video", fileBlob, fileName);

  const queryParams = new URLSearchParams();
  if (category) queryParams.set("category", category);
  queryParams.set("top_k", topK);

  return requestWithTimeout(`${API_BASE}/recognize/isolated/video?${queryParams.toString()}`, {
    method: "POST",
    body: formData,
  }, 25000);
}

/**
 * Send pre-extracted landmark sequence directly for Isolated ISL sign recognition.
 */
export async function recognizeIsolatedLandmarks(landmarks, category = null, topK = 3) {
  return requestWithTimeout(`${API_BASE}/recognize/isolated/landmarks`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      landmarks,
      category,
      top_k: topK,
    }),
  });
}

/**
 * Send pre-extracted landmark sequence to backend for Continuous ISL Sentence Recognition (101 Sentences).
 */
export async function recognizeSentenceLandmarks(landmarks, options = {}) {
  const payload = {
    landmarks,
    timestamps_ms: options.timestamps !== undefined ? options.timestamps : (options.timestamps_ms || null),
    fps: options.fps || 25,
    top_k: options.top_k || options.topK || 3,
  };

  return requestWithTimeout(`${API_BASE_URL}/recognize/sentence/landmarks`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  }, 25000);
}

/**
 * Send uploaded video file to backend for Continuous ISL Sentence Recognition (101 Sentences).
 */
export async function recognizeSentenceVideo(videoFile, onProgress = null) {
  const formData = new FormData();
  formData.append("video", videoFile, videoFile.name || "sentence.mp4");

  if (onProgress && typeof XMLHttpRequest !== "undefined") {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `${API_BASE_URL}/recognize/sentence/video`);
      xhr.timeout = 45000;

      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable && onProgress) {
          const percent = Math.round((event.loaded / event.total) * 100);
          onProgress(percent, event.loaded, event.total);
        }
      };

      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            const data = JSON.parse(xhr.responseText);
            resolve(data);
          } catch {
            reject(new Error("Failed to parse JSON response"));
          }
        } else {
          try {
            const errData = JSON.parse(xhr.responseText);
            reject(new Error(errData.detail || errData.error?.message || `Upload failed with status ${xhr.status}`));
          } catch {
            reject(new Error(`Upload failed with status ${xhr.status}`));
          }
        }
      };

      xhr.onerror = () => reject(new Error("Network error during video upload"));
      xhr.ontimeout = () => reject(new Error("Video upload timed out"));
      xhr.send(formData);
    });
  }

  return requestWithTimeout(
    `${API_BASE_URL}/recognize/sentence/video`,
    {
      method: "POST",
      body: formData,
    },
    45000
  );
}

/**
 * Fetch the 101 ISL continuous sentence vocabulary and sentences from backend.
 */
export async function fetchSentenceVocabulary() {
  return requestWithTimeout(`${API_BASE_URL}/recognize/sentence/sentences`);
}

/**
 * Fetch the 5 supported continuous ISL sentence classes for Hackathon Demo Mode.
 */
export async function fetchDemoSentenceClasses() {
  return requestWithTimeout(`${API_BASE_URL}/recognize/sentence/demo/classes`);
}

/**
 * Send uploaded video file to backend for 5-Class Hackathon Demo Sentence Recognition.
 */
export async function recognizeDemoSentenceVideo(videoFile, options = {}) {
  const formData = new FormData();
  formData.append("video", videoFile, videoFile.name || "demo_sentence.mp4");

  const queryParams = new URLSearchParams();
  if (options.threshold !== undefined) {
    queryParams.set("threshold", options.threshold);
  }
  if (options.min_margin !== undefined) {
    queryParams.set("min_margin", options.min_margin);
  }

  const queryString = queryParams.toString();
  const url = `${API_BASE_URL}/recognize/sentence/demo/video${queryString ? `?${queryString}` : ""}`;

  return requestWithTimeout(
    url,
    {
      method: "POST",
      body: formData,
    },
    90000 // 90s timeout for MediaPipe video landmark extraction
  );
}

/**
 * Fetch the 5 supported ISL sentence classes for the 5-class sequence models.
 */
export async function fetchFiveClassClasses() {
  return requestWithTimeout(`${API_BASE_URL}/recognize/sentence/five-class/classes`);
}

/**
 * Send recorded webcam or uploaded video file to backend for 5-Class Sentence Recognition.
 * Supports selectable models: "bigru" (BiGRU Attention) or "dtw" (DTW-KNN).
 */
export async function recognizeFiveClassSentenceVideo(videoFile, model = "bigru", options = {}, onProgress = null) {
  const progressCallback = typeof options === "function" ? options : (onProgress || options?.onProgress || null);
  const opts = typeof options === "object" && options !== null ? options : {};

  const formData = new FormData();
  formData.append("video", videoFile, videoFile.name || "webcam_gesture.webm");

  const queryParams = new URLSearchParams();
  queryParams.set("model", model);
  if (opts.confidenceThreshold !== undefined) {
    queryParams.set("confidence_threshold", opts.confidenceThreshold);
  }

  const queryString = queryParams.toString();
  const url = `${API_BASE_URL}/recognize/sentence/five-class/video?${queryString}`;

  if (progressCallback && typeof XMLHttpRequest !== "undefined") {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", url);
      xhr.timeout = 90000;

      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable && progressCallback) {
          const percent = Math.round((event.loaded / event.total) * 100);
          progressCallback(percent, event.loaded, event.total);
        }
      };

      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            const data = JSON.parse(xhr.responseText);
            resolve(data);
          } catch {
            reject(new Error("Failed to parse JSON response"));
          }
        } else {
          try {
            const errData = JSON.parse(xhr.responseText);
            reject(new Error(errData.detail || errData.error?.message || `Upload failed with status ${xhr.status}`));
          } catch {
            reject(new Error(`Upload failed with status ${xhr.status}`));
          }
        }
      };

      xhr.onerror = () => reject(new Error("Network error during video upload"));
      xhr.ontimeout = () => reject(new Error("Video upload timed out"));
      xhr.send(formData);
    });
  }

  return requestWithTimeout(
    url,
    {
      method: "POST",
      body: formData,
    },
    90000 // 90s timeout for video decoding and MediaPipe extraction
  );
}

/**
 * Send pre-extracted landmark sequence directly for 5-Class Sentence Recognition.
 */
export async function recognizeFiveClassSentenceLandmarks(landmarks, model = "bigru", options = {}) {
  const queryParams = new URLSearchParams();
  queryParams.set("model", model);
  if (options.confidenceThreshold !== undefined) {
    queryParams.set("confidence_threshold", options.confidenceThreshold);
  }

  const queryString = queryParams.toString();
  const url = `${API_BASE_URL}/recognize/sentence/five-class/landmarks?${queryString}`;

  const payload = {
    landmarks,
    timestamps_ms: options.timestamps !== undefined ? options.timestamps : (options.timestamps_ms || null),
    fps: options.fps || 25,
    top_k: options.top_k || options.topK || 5,
  };

  return requestWithTimeout(
    url,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    },
    25000
  );
}


/**
 * Translate text into target language using Sarvam AI (FE-08).
 * Deduplicates concurrent identical translation requests.
 */
export async function translateText(text, targetLanguageCode, sourceLanguageCode = "en-IN") {
  const cleanText = (text || "").trim();
  const dedupeKey = `translate:${cleanText}:${targetLanguageCode}:${sourceLanguageCode}`;

  return requestWithTimeout(
    `${API_BASE}/translate`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        input: cleanText,
        target_language_code: targetLanguageCode,
        source_language_code: sourceLanguageCode,
      }),
    },
    DEFAULT_TIMEOUT_MS,
    dedupeKey
  );
}

/**
 * Fetch supported translation languages from backend.
 */
export async function fetchSupportedLanguages() {
  return requestWithTimeout(`${API_BASE}/translate/languages`);
}

/**
 * Synthesize text into spoken audio using Sarvam Bulbul TTS (FE-08).
 * Deduplicates concurrent identical synthesis requests.
 */
export async function synthesizeSpeech(text, targetLanguageCode, speaker = "meera", pace = 1.0) {
  const cleanText = (text || "").trim();
  const dedupeKey = `tts:${cleanText}:${targetLanguageCode}:${speaker}:${pace}`;

  return requestWithTimeout(
    `${API_BASE}/speech/synthesize`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        text: cleanText,
        target_language_code: targetLanguageCode,
        speaker,
        pace,
      }),
    },
    DEFAULT_TIMEOUT_MS,
    dedupeKey
  );
}

/**
 * Fetch supported TTS speakers from backend.
 */
export async function fetchSupportedSpeakers() {
  return requestWithTimeout(`${API_BASE}/speech/speakers`);
}

/**
 * Transcribe recorded audio file using Sarvam Saaras STT via backend.
 */
export async function transcribeSpeech(audioBlob, fileName = "recording.webm", languageCode = null) {
  const formData = new FormData();
  formData.append("audio", audioBlob, fileName);

  const queryParams = new URLSearchParams();
  if (languageCode) queryParams.set("language_code", languageCode);

  return requestWithTimeout(`${API_BASE}/speech/transcribe?${queryParams.toString()}`, {
    method: "POST",
    body: formData,
  }, 25000);
}

/**
 * Direction B: Transcribe Other Person's audio and translate into Your Language.
 */
export async function transcribeAndTranslateSpeech(
  audioBlob,
  fileName = "speech.webm",
  sourceLanguageCode = "en-IN",
  targetLanguageCode = "gu-IN"
) {
  const formData = new FormData();
  formData.append("audio", audioBlob, fileName);

  const queryParams = new URLSearchParams({
    source_language_code: sourceLanguageCode,
    target_language_code: targetLanguageCode,
  });

  return requestWithTimeout(`${API_BASE}/speech/transcribe-and-translate?${queryParams.toString()}`, {
    method: "POST",
    body: formData,
  }, 25000);
}

/**
 * Send a video or .pose file to backend for MediaPipe extraction and iSign T5 translation.
 */
export async function predictVideo(fileBlob, fileName = "recording.webm", decoding = "beam") {
  const formData = new FormData();
  formData.append("video", fileBlob, fileName);

  return requestWithTimeout(`${API_BASE}/predict/video?decoding=${encodeURIComponent(decoding)}`, {
    method: "POST",
    body: formData,
  }, 30000);
}

/**
 * Send pre-extracted 300-dim landmark sequence to backend.
 */
export async function predictLandmarks(landmarks, decoding = "beam") {
  return requestWithTimeout(`${API_BASE}/predict/landmarks`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      landmarks,
      decoding,
    }),
  });
}

/**
 * Check backend health status.
 */
export async function fetchHealth() {
  return requestWithTimeout(`${API_BASE}/health`, {}, 5000);
}
