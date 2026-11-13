import React, { useState, useRef, useEffect, useCallback } from "react";
import {
  Mic,
  MicOff,
  Volume2,
  VolumeX,
  RotateCcw,
  Square,
  AlertCircle,
  Loader2,
  CheckCircle2,
  Globe,
  Trash2,
  Play,
} from "lucide-react";
import { transcribeAndTranslateSpeech, synthesizeSpeech } from "../../services/api";

export default function PanelOtherVoice({
  yourLanguage = "gu-IN",
  yourLanguageName = "Gujarati",
  otherLanguage = "en-IN",
  otherLanguageName = "English",
  defaultAutoPlayAudio = true,
}) {
  const [isRecording, setIsRecording] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [statusState, setStatusState] = useState("idle"); // "idle" | "recording" | "processing" | "success" | "error"
  const [statusMessage, setStatusMessage] = useState("Ready for speech input");
  const [originalTranscript, setOriginalTranscript] = useState("");
  const [translatedText, setTranslatedText] = useState("");
  const [errorMessage, setErrorMessage] = useState(null);

  // Audio playback state (Sarvam Bulbul TTS playback in Your Language)
  const [isPlayingAudio, setIsPlayingAudio] = useState(false);
  const [isSynthesizingTTS, setIsSynthesizingTTS] = useState(false);
  const [audioBlocked, setAudioBlocked] = useState(false);
  const [autoPlayAudio, setAutoPlayAudio] = useState(defaultAutoPlayAudio);

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const streamRef = useRef(null);
  const recordingTimerRef = useRef(null);
  const autoStopTimeoutRef = useRef(null);
  const audioInstanceRef = useRef(null);
  const latestRequestIdRef = useRef(0);
  const audioCacheRef = useRef({ text: "", lang: "", audioBase64: null });

  // Online / offline network detection
  useEffect(() => {
    const handleOnline = () => {
      if (errorMessage && errorMessage.includes("offline")) {
        setErrorMessage(null);
      }
    };
    const handleOffline = () => {
      setErrorMessage("Network disconnected. You are currently offline. Check your internet connection.");
    };

    window.addEventListener("online", handleOnline);
    window.addEventListener("offline", handleOffline);
    return () => {
      window.removeEventListener("online", handleOnline);
      window.removeEventListener("offline", handleOffline);
    };
  }, [errorMessage]);

  useEffect(() => {
    return () => {
      stopRecordingCleanup();
      if (audioInstanceRef.current) {
        audioInstanceRef.current.pause();
        audioInstanceRef.current = null;
      }
    };
  }, []);

  const stopAudio = useCallback(() => {
    if (audioInstanceRef.current) {
      audioInstanceRef.current.pause();
      audioInstanceRef.current.currentTime = 0;
      audioInstanceRef.current = null;
    }
    setIsPlayingAudio(false);
  }, []);

  const stopRecordingCleanup = () => {
    if (recordingTimerRef.current) {
      clearInterval(recordingTimerRef.current);
      recordingTimerRef.current = null;
    }
    if (autoStopTimeoutRef.current) {
      clearTimeout(autoStopTimeoutRef.current);
      autoStopTimeoutRef.current = null;
    }
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
      try {
        mediaRecorderRef.current.stop();
      } catch {
        // ignore
      }
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => {
        track.onended = null;
        track.stop();
      });
      streamRef.current = null;
    }
  };

  const startRecording = async () => {
    setErrorMessage(null);
    setAudioBlocked(false);
    audioChunksRef.current = [];
    stopAudio();

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setErrorMessage("Microphone access is not supported by your browser.");
      setStatusState("error");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });
      streamRef.current = stream;

      // Handle microphone disconnection during active recording
      const audioTracks = stream.getAudioTracks
        ? stream.getAudioTracks()
        : (stream.getTracks ? stream.getTracks().filter((t) => t.kind === "audio") : []);
      const audioTrack = audioTracks[0];
      if (audioTrack) {
        audioTrack.onended = () => {
          stopRecordingCleanup();
          setIsRecording(false);
          setStatusState("error");
          setErrorMessage("Microphone was disconnected during recording. Please reconnect your microphone and retry.");
        };
      }

      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : MediaRecorder.isTypeSupported("audio/webm")
        ? "audio/webm"
        : "audio/wav";

      const recorder = new MediaRecorder(stream, { mimeType });
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      // Handle hardware / recording errors
      recorder.onerror = (event) => {
        stopRecordingCleanup();
        setIsRecording(false);
        setStatusState("error");
        setErrorMessage(
          event.error?.message || "Microphone recording failed due to an audio device error. Please retry."
        );
      };

      recorder.onstop = async () => {
        setIsRecording(false);
        const chunks = audioChunksRef.current;
        if (!chunks || chunks.length === 0) {
          setStatusState("error");
          setStatusMessage("No audio captured.");
          setErrorMessage("No audio captured from microphone. Please speak into the mic and retry.");
          return;
        }

        const audioBlob = new Blob(chunks, { type: mimeType });
        if (audioBlob.size === 0) {
          setStatusState("error");
          setStatusMessage("Empty audio recording.");
          setErrorMessage("Audio recording was empty. Please speak clearly into the microphone and retry.");
          return;
        }
        await processAudio(audioBlob, mimeType);
      };

      recorder.start(250);
      setIsRecording(true);
      setStatusState("recording");
      setStatusMessage(`Listening to ${otherLanguageName}...`);
      setRecordingSeconds(0);

      // Start recording duration timer
      recordingTimerRef.current = setInterval(() => {
        setRecordingSeconds((prev) => prev + 1);
      }, 1000);

      // 15 seconds auto-stop cap
      autoStopTimeoutRef.current = setTimeout(() => {
        stopRecording();
      }, 15000);
    } catch (err) {
      setIsRecording(false);
      setStatusState("error");
      if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
        setErrorMessage("Microphone permission was denied. Please allow microphone access in browser settings and click Retry.");
      } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
        setErrorMessage("No microphone found on this device. Please plug in or connect an audio input.");
      } else {
        setErrorMessage(err.message || "Could not access microphone.");
      }
    }
  };

  const stopRecording = () => {
    if (recordingTimerRef.current) {
      clearInterval(recordingTimerRef.current);
      recordingTimerRef.current = null;
    }
    if (autoStopTimeoutRef.current) {
      clearTimeout(autoStopTimeoutRef.current);
      autoStopTimeoutRef.current = null;
    }
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
      try {
        mediaRecorderRef.current.stop();
      } catch {
        // ignore
      }
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => {
        track.onended = null;
        track.stop();
      });
      streamRef.current = null;
    }
  };

  const toggleRecording = () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  const processAudio = async (audioBlob, mimeType) => {
    if (!audioBlob || audioBlob.size === 0) {
      setStatusState("error");
      setStatusMessage("Empty audio recording.");
      setErrorMessage("No audio captured. Please speak into the microphone and retry.");
      return;
    }

    const reqId = ++latestRequestIdRef.current;

    setStatusState("processing");
    setStatusMessage("Transcribing & translating speech...");

    const fileName = mimeType.includes("webm") ? "speech.webm" : "speech.wav";

    try {
      const response = await transcribeAndTranslateSpeech(
        audioBlob,
        fileName,
        otherLanguage,
        yourLanguage
      );

      // Discard stale result if a newer request began
      if (reqId !== latestRequestIdRef.current) return;

      if (response && response.success && response.original_transcript) {
        const translated = response.translated_text || "";
        setOriginalTranscript(response.original_transcript);
        setTranslatedText(translated);
        setStatusState("success");
        setStatusMessage("Message translated successfully.");
        setErrorMessage(null);

        // Stop previous audio and reset cache for new incoming translation
        stopAudio();
        audioCacheRef.current = { text: "", lang: "", audioBase64: null };

        // Automatically trigger Sarvam Bulbul TTS when autoPlayAudio is enabled
        if (autoPlayAudio && translated) {
          playBulbulAudio(translated, yourLanguage);
        }
      } else {
        // Empty speech transcript or failure - do NOT display fake translation results!
        setStatusState("error");
        const msg =
          response?.error_message ||
          "No speech detected in audio recording. Please speak clearly and try again.";
        setStatusMessage("No speech detected.");
        setErrorMessage(msg);
        setOriginalTranscript("");
        setTranslatedText("");
        stopAudio();
        audioCacheRef.current = { text: "", lang: "", audioBase64: null };
      }
    } catch (err) {
      if (reqId !== latestRequestIdRef.current) return;
      setStatusState("error");
      setStatusMessage("Speech processing error.");
      setErrorMessage(
        err.message || "Failed to transcribe speech. Check backend and Sarvam AI connection."
      );
      stopAudio();
    }
  };

  const playBulbulAudio = useCallback(
    async (text, lang) => {
      if (!text) return;

      // Stop previous audio before playing new
      if (audioInstanceRef.current) {
        audioInstanceRef.current.pause();
        audioInstanceRef.current.currentTime = 0;
        audioInstanceRef.current = null;
      }
      setIsPlayingAudio(false);
      setAudioBlocked(false);

      let base64 = null;
      if (
        audioCacheRef.current &&
        audioCacheRef.current.text === text &&
        audioCacheRef.current.lang === lang &&
        audioCacheRef.current.audioBase64
      ) {
        base64 = audioCacheRef.current.audioBase64;
      } else {
        setIsSynthesizingTTS(true);
        setErrorMessage(null);
        try {
          const res = await synthesizeSpeech(text, lang, "priya");
          if (res && res.audio_base64) {
            base64 = res.audio_base64;
            audioCacheRef.current = { text, lang, audioBase64: base64 };
          } else {
            setErrorMessage(res?.error_message || "Audio synthesis returned empty audio data.");
            return;
          }
        } catch (err) {
          setErrorMessage(err.message || "Speech synthesis failed. Please retry.");
          return;
        } finally {
          setIsSynthesizingTTS(false);
        }
      }

      if (!base64) return;

      try {
        const audioUrl = `data:audio/wav;base64,${base64}`;
        const audio = new Audio(audioUrl);
        audioInstanceRef.current = audio;

        audio.onended = () => {
          setIsPlayingAudio(false);
        };
        audio.onerror = () => {
          setIsPlayingAudio(false);
        };

        const playPromise = audio.play();
        if (playPromise !== undefined) {
          await playPromise;
          setIsPlayingAudio(true);
          setAudioBlocked(false);
        }
      } catch (err) {
        setIsPlayingAudio(false);
        if (
          err &&
          (err.name === "NotAllowedError" ||
            err.name === "AbortError" ||
            err.message?.toLowerCase().includes("interact") ||
            err.message?.toLowerCase().includes("user gesture"))
        ) {
          setAudioBlocked(true);
        } else {
          setErrorMessage(err.message || "Failed to play audio.");
        }
      }
    },
    []
  );

  // Manual replay TTS playback in Your Language (Deaf user can listen if desired)
  const handlePlayTranslatedAudio = () => {
    if (!translatedText) return;

    if (isPlayingAudio) {
      stopAudio();
      return;
    }

    playBulbulAudio(translatedText, yourLanguage);
  };

  const clearConversation = () => {
    stopAudio();
    stopRecordingCleanup();
    setIsRecording(false);
    setOriginalTranscript("");
    setTranslatedText("");
    setStatusState("idle");
    setStatusMessage("Ready for speech input");
    setErrorMessage(null);
    setAudioBlocked(false);
    audioCacheRef.current = { text: "", lang: "", audioBase64: null };
  };

  return (
    <div
      className="communication-panel panel-other-voice"
      data-testid="panel-other-voice"
      aria-label="Direction B: Other Person's Voice to Readable Regional Text"
    >
      <header className="panel-header">
        <div className="panel-title-group">
          <span className="panel-direction-badge hearing-badge">
            Hearing Person → Deaf Person
          </span>
          <h2 className="panel-title">Other Person's Voice</h2>
        </div>
        <span className="panel-target-lang-pill" data-testid="panel-b-target-lang">
          Input: <strong>{otherLanguageName}</strong> &bull; Output: <strong>{yourLanguageName}</strong>
        </span>
      </header>

      {/* Microphone Controls Card */}
      <section className="mic-control-card" aria-label="Microphone Voice Capture">
        <div className="mic-action-wrap">
          <button
            type="button"
            className={`mic-toggle-button ${isRecording ? "recording" : "idle"}`}
            data-testid="mic-toggle-btn"
            onClick={toggleRecording}
            disabled={statusState === "processing"}
            aria-label={isRecording ? "Stop recording speech" : "Start speaking into microphone"}
          >
            {isRecording ? (
              <>
                <Square size={22} className="mic-icon pulse" />
                <span className="mic-btn-label">
                  Stop ({15 - recordingSeconds}s)
                </span>
              </>
            ) : statusState === "processing" ? (
              <>
                <Loader2 size={22} className="rotating" />
                <span className="mic-btn-label">Processing...</span>
              </>
            ) : (
              <>
                <Mic size={22} className="mic-icon" />
                <span className="mic-btn-label">Tap to Speak</span>
              </>
            )}
          </button>
        </div>

        {/* Recording Status & Waveform Indicator */}
        <div className="mic-status-row">
          <div
            className={`mic-status-pill ${statusState}`}
            data-testid="mic-status-badge"
            role="status"
            aria-live="polite"
          >
            {isRecording && <span className="mic-recording-pulse-dot" />}
            <span>{statusMessage}</span>
          </div>

          {isRecording && (
            <div className="recording-wave-visualizer" aria-hidden="true">
              <span className="wave-bar bar-1" />
              <span className="wave-bar bar-2" />
              <span className="wave-bar bar-3" />
              <span className="wave-bar bar-4" />
              <span className="wave-bar bar-5" />
            </div>
          )}
        </div>

        {errorMessage && (
          <div className="mic-error-banner" data-testid="mic-error-banner" role="alert" style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: "10px" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "8px", flex: 1 }}>
              <AlertCircle size={18} style={{ flexShrink: 0 }} />
              <span>{errorMessage}</span>
            </div>
            <button
              type="button"
              className="mic-retry-btn"
              data-testid="mic-retry-btn"
              onClick={() => startRecording()}
              aria-label="Retry microphone"
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: "4px",
                padding: "4px 8px",
                borderRadius: "var(--radius-sm)",
                border: "1px solid rgba(239, 68, 68, 0.4)",
                backgroundColor: "rgba(239, 68, 68, 0.15)",
                color: "#DC2626",
                cursor: "pointer",
                fontSize: "0.78rem",
                fontWeight: "600",
                flexShrink: 0,
              }}
            >
              <RotateCcw size={13} />
              <span>Retry</span>
            </button>
          </div>
        )}
      </section>

      {/* Prominent Translated Text Display (Designed specifically for Deaf reader) */}
      <section
        className="voice-communication-display"
        data-testid="voice-communication-display"
        aria-label="Translated Message Display for Deaf User"
      >
        <div className="transcript-section">
          <div className="section-label-row">
            <span className="section-sub-label">
              Hearing Person Spoke ({otherLanguageName}):
            </span>
          </div>
          <div
            className="original-speech-transcript"
            data-testid="original-speech-transcript"
          >
            {originalTranscript ? (
              <p className="transcript-text">"{originalTranscript}"</p>
            ) : (
              <p className="transcript-placeholder">
                Original speech transcript will appear here when the hearing person speaks...
              </p>
            )}
          </div>
        </div>

        {/* Prominent Large Translated Regional Language Text */}
        <div className="translation-prominent-section">
          <div className="section-label-row">
            <span className="section-main-label">
              Readable Translation in {yourLanguageName}:
            </span>
            <span className="regional-tag">{yourLanguageName}</span>
          </div>

          <div
            className={`translated-text-display-box ${translatedText ? "has-content" : "empty"}`}
            data-testid="translated-voice-text-box"
          >
            {statusState === "processing" ? (
              <div className="translating-spinner-wrap">
                <Loader2 size={24} className="rotating" />
                <span>Translating message for you...</span>
              </div>
            ) : translatedText ? (
              <h3 className="readable-regional-text" data-testid="readable-regional-text">
                {translatedText}
              </h3>
            ) : (
              <p className="readable-regional-placeholder">
                Translated regional text will appear here prominently for easy reading.
              </p>
            )}
          </div>
        </div>

        {/* Browser Autoplay Blocked Notification */}
        {audioBlocked && (
          <div
            className="autoplay-blocked-banner"
            data-testid="autoplay-blocked-banner"
            role="alert"
            aria-live="polite"
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              gap: "10px",
              padding: "10px 14px",
              margin: "12px 0 0 0",
              backgroundColor: "rgba(245, 158, 11, 0.12)",
              border: "1px solid rgba(245, 158, 11, 0.4)",
              borderRadius: "var(--radius-md)",
              color: "#92400E",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <VolumeX size={18} style={{ color: "#D97706", flexShrink: 0 }} />
              <span style={{ fontWeight: "600", fontSize: "0.85rem" }}>
                Tap to play translated audio.
              </span>
            </div>
            <button
              type="button"
              className="optional-audio-btn unblock fallback-play-audio-btn"
              data-testid="unblock-voice-audio-btn"
              onClick={handlePlayTranslatedAudio}
              aria-label="Play Audio"
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: "5px",
                padding: "6px 14px",
                borderRadius: "var(--radius-sm)",
                backgroundColor: "#D97706",
                color: "#FFFFFF",
                border: "none",
                cursor: "pointer",
                fontWeight: "600",
                fontSize: "0.82rem",
                boxShadow: "0 1px 2px rgba(0,0,0,0.1)",
              }}
            >
              <Play size={14} />
              <span>Play Audio</span>
            </button>
          </div>
        )}

        {/* Action Controls: Optional Audio Playback, Auto Play Setting & Clear */}
        <div className="voice-actions-toolbar">
          <div className="voice-actions-left" style={{ display: "flex", gap: "10px", alignItems: "center", flexWrap: "wrap" }}>
            <button
              type="button"
              className={`optional-audio-btn ${isPlayingAudio ? "playing" : ""}`}
              data-testid="optional-audio-btn"
              onClick={handlePlayTranslatedAudio}
              disabled={!translatedText || statusState === "processing" || isSynthesizingTTS}
              aria-label={
                isPlayingAudio
                  ? "Stop reading audio"
                  : `Listen to translation in ${yourLanguageName}`
              }
              title={`Optional speech synthesis in ${yourLanguageName}`}
            >
              {isSynthesizingTTS ? (
                <>
                  <Loader2 size={16} className="rotating" />
                  <span>Generating Audio...</span>
                </>
              ) : isPlayingAudio ? (
                <>
                  <Square size={16} />
                  <span>Stop Audio</span>
                </>
              ) : (
                <>
                  <Volume2 size={16} />
                  <span>Listen ({yourLanguageName})</span>
                </>
              )}
            </button>

            {/* Auto Play Audio Setting Toggle */}
            <label
              className="auto-play-setting-toggle"
              data-testid="auto-play-setting-label"
              htmlFor="auto-play-audio-checkbox"
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: "6px",
                fontSize: "0.82rem",
                fontWeight: "600",
                color: "var(--color-text-secondary, #4B5563)",
                cursor: "pointer",
                userSelect: "none",
                padding: "6px 10px",
                borderRadius: "var(--radius-md)",
                backgroundColor: "var(--color-surface, #F9FAFB)",
                border: "1px solid var(--color-border, #E5E7EB)",
              }}
            >
              <input
                id="auto-play-audio-checkbox"
                type="checkbox"
                data-testid="auto-play-audio-toggle"
                checked={autoPlayAudio}
                onChange={(e) => setAutoPlayAudio(e.target.checked)}
                aria-label="Auto Play Audio"
                style={{
                  cursor: "pointer",
                  width: "15px",
                  height: "15px",
                  accentColor: "var(--color-primary, #4F46E5)",
                }}
              />
              <span>Auto Play Audio</span>
            </label>
          </div>

          <div className="voice-actions-right">
            <button
              type="button"
              className="clear-voice-btn"
              data-testid="clear-voice-btn"
              onClick={clearConversation}
              disabled={!originalTranscript && !translatedText}
              aria-label="Clear speech transcript"
              title="Clear current voice message"
            >
              <Trash2 size={14} />
              <span>Clear</span>
            </button>
          </div>
        </div>
      </section>
    </div>
  );
}
