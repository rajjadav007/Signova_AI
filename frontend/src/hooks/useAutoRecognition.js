import { useState, useRef, useEffect, useCallback } from "react";
import {
  RECOGNITION_STATE,
  RECOGNITION_STATE_LABELS,
  HandPresenceTracker,
  detectHandPresenceInCanvas,
  DEFAULT_FOCUS_ZONE,
} from "../utils/handPresence";
import { recognizeIsolatedVideo } from "../services/api";

export const MIN_GESTURE_DURATION_MS = 800; // 0.8s minimum gesture duration
export const MAX_GESTURE_DURATION_MS = 6000; // 6.0s duration safety cap
export const RESULT_DISPLAY_DURATION_MS = 2500; // 2.5s display window before next sign
export const RESET_SEQUENCE_DURATION_MS = 400; // 0.4s to display "Preparing next sign"
export const SEQUENCE_CAPACITY = 64;

export function useAutoRecognition({
  videoRef,
  stream = null,
  isCameraActive = true,
  enabled = true,
  autoSpeak = true,
  focusZone = DEFAULT_FOCUS_ZONE,
  onResetSequence = null,
  onSignRecognized = null,
  onUncertainSuggestion = null,
} = {}) {
  const stateMachineRef = useRef(
    isCameraActive ? RECOGNITION_STATE.NO_HAND : RECOGNITION_STATE.NO_HAND
  );
  const [state, setState] = useState(stateMachineRef.current);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [frameCount, setFrameCount] = useState(0);
  const [detectedHands, setDetectedHands] = useState([]);
  const [handConfidence, setHandConfidence] = useState(0);

  const trackerRef = useRef(new HandPresenceTracker({ enterThreshold: 3, lostThreshold: 5 }));
  const isMountedRef = useRef(true);
  const isRecognizingRef = useRef(false);
  const mediaRecorderRef = useRef(null);
  const recordedChunksRef = useRef([]);
  const recordingStartTimeRef = useRef(null);
  const frameCountRef = useRef(0);
  const nextSignTimerRef = useRef(null);
  const resetTimerRef = useRef(null);
  const onSignRecognizedRef = useRef(onSignRecognized);
  const onUncertainSuggestionRef = useRef(onUncertainSuggestion);

  useEffect(() => {
    onSignRecognizedRef.current = onSignRecognized;
    onUncertainSuggestionRef.current = onUncertainSuggestion;
  }, [onSignRecognized, onUncertainSuggestion]);

  const transitionState = useCallback((newState) => {
    stateMachineRef.current = newState;
    if (isMountedRef.current) {
      setState(newState);
    }
  }, []);

  // Helper to safely speak recognized words
  const speakPrediction = useCallback((text) => {
    if (!text || typeof window === "undefined" || !("speechSynthesis" in window)) {
      return;
    }
    try {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1.0;
      utterance.pitch = 1.0;
      utterance.lang = "en-US";
      window.speechSynthesis.speak(utterance);
    } catch {
      // Ignore speech synthesis errors
    }
  }, []);

  // Stop media recorder safely
  const stopRecorder = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
      try {
        mediaRecorderRef.current.stop();
      } catch {
        // ignore
      }
    }
    mediaRecorderRef.current = null;
  }, []);

  // Reset internal sequence buffer and temporary recording resources
  const resetSequence = useCallback(() => {
    stopRecorder();
    recordedChunksRef.current = [];
    recordingStartTimeRef.current = null;
    frameCountRef.current = 0;
    if (typeof onResetSequence === "function") {
      try {
        onResetSequence();
      } catch {
        // ignore
      }
    }
    if (isMountedRef.current) {
      setFrameCount(0);
    }
  }, [stopRecorder, onResetSequence]);

  // Execute isolated video recognition
  const dispatchRecognition = useCallback(
    async (videoBlob) => {
      if (isRecognizingRef.current || !videoBlob || videoBlob.size === 0) {
        return;
      }

      isRecognizingRef.current = true;
      transitionState(RECOGNITION_STATE.RECOGNIZING);
      if (isMountedRef.current) {
        setError(null);
      }

      try {
        const response = await recognizeIsolatedVideo(
          videoBlob,
          "auto_gesture.webm",
          null,
          1 // Request single highest-confidence candidate
        );

        if (!isMountedRef.current) return;

        setResult(response);

        if (response.status === "recognized" && response.confidence >= (response.rejection_threshold || 0.80)) {
          transitionState(RECOGNITION_STATE.SUCCESS);
          if (autoSpeak && response.predicted_sign) {
            speakPrediction(response.display_name || response.predicted_sign);
          }
          if (typeof onSignRecognizedRef.current === "function") {
            try {
              onSignRecognizedRef.current(response.display_name || response.predicted_sign, response);
            } catch {
              // ignore
            }
          }
        } else {
          transitionState(RECOGNITION_STATE.UNCERTAIN);
          if (
            response.predicted_sign &&
            response.predicted_sign !== "unknown" &&
            response.predicted_sign !== "uncertain" &&
            typeof onUncertainSuggestionRef.current === "function"
          ) {
            try {
              onUncertainSuggestionRef.current(response.display_name || response.predicted_sign, response);
            } catch {
              // ignore
            }
          }
        }

        // 2.5s display window before resetting sequence and preparing for next sign
        if (nextSignTimerRef.current) {
          clearTimeout(nextSignTimerRef.current);
        }
        nextSignTimerRef.current = setTimeout(() => {
          transitionState(RECOGNITION_STATE.RESET_SEQUENCE);
          resetSequence();

          if (resetTimerRef.current) {
            clearTimeout(resetTimerRef.current);
          }
          resetTimerRef.current = setTimeout(() => {
            transitionState(RECOGNITION_STATE.WAITING_FOR_NEXT_SIGN);
          }, RESET_SEQUENCE_DURATION_MS);
        }, RESULT_DISPLAY_DURATION_MS);
      } catch (err) {
        if (isMountedRef.current) {
          setResult(null);
          setError(err.message || "Recognition service unavailable.");
          transitionState(RECOGNITION_STATE.ERROR);

          if (nextSignTimerRef.current) {
            clearTimeout(nextSignTimerRef.current);
          }
          nextSignTimerRef.current = setTimeout(() => {
            transitionState(RECOGNITION_STATE.RESET_SEQUENCE);
            resetSequence();

            if (resetTimerRef.current) {
              clearTimeout(resetTimerRef.current);
            }
            resetTimerRef.current = setTimeout(() => {
              transitionState(RECOGNITION_STATE.WAITING_FOR_NEXT_SIGN);
            }, RESET_SEQUENCE_DURATION_MS);
          }, RESULT_DISPLAY_DURATION_MS);
        }
      } finally {
        isRecognizingRef.current = false;
      }
    },
    [autoSpeak, speakPrediction, resetSequence, transitionState]
  );

  // Handle recorder stop event: validate duration and size, then dispatch
  const handleRecorderStop = useCallback(
    (mimeType) => {
      const durationMs = recordingStartTimeRef.current
        ? Date.now() - recordingStartTimeRef.current
        : 0;
      const chunks = recordedChunksRef.current;

      transitionState(RECOGNITION_STATE.VALIDATING_RECORDING);

      const blob = chunks.length > 0 ? new Blob(chunks, { type: mimeType }) : null;

      if (!blob || blob.size === 0 || durationMs < MIN_GESTURE_DURATION_MS) {
        // Discard recording: too short, empty or zero-byte
        resetSequence();
        transitionState(RECOGNITION_STATE.RESET_SEQUENCE);
        if (resetTimerRef.current) {
          clearTimeout(resetTimerRef.current);
        }
        resetTimerRef.current = setTimeout(() => {
          transitionState(RECOGNITION_STATE.NO_HAND);
        }, RESET_SEQUENCE_DURATION_MS);
        return;
      }

      dispatchRecognition(blob);
    },
    [dispatchRecognition, resetSequence, transitionState]
  );

  // Initialize and start MediaRecorder for gesture recording
  const startRecording = useCallback(() => {
    if (!stream) return false;

    // Stop any existing recorder to avoid duplicate sessions
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.onstop = null;
    }
    stopRecorder();
    if (typeof onResetSequence === "function") {
      try {
        onResetSequence();
      } catch {
        // ignore
      }
    }
    recordedChunksRef.current = [];
    recordingStartTimeRef.current = Date.now();
    frameCountRef.current = 0;

    const mimeType =
      typeof MediaRecorder !== "undefined" && MediaRecorder.isTypeSupported("video/webm;codecs=vp9")
        ? "video/webm;codecs=vp9"
        : typeof MediaRecorder !== "undefined" && MediaRecorder.isTypeSupported("video/webm")
        ? "video/webm"
        : "video/mp4";

    try {
      const recorder = new MediaRecorder(stream, { mimeType });
      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          recordedChunksRef.current.push(event.data);
        }
      };

      recorder.onstop = () => {
        handleRecorderStop(mimeType);
      };

      mediaRecorderRef.current = recorder;
      recorder.start(100);
      return true;
    } catch {
      return false;
    }
  }, [stream, stopRecorder, handleRecorderStop]);

  // Main frame processing hook called by useFrameSampler on every frame
  const processFrame = useCallback(
    async (frameData) => {
      if (!isCameraActive || !enabled || !frameData || !frameData.canvas) {
        return;
      }

      // Step 1: Detect hand presence locally in focus zone
      const detection = detectHandPresenceInCanvas(frameData.canvas, focusZone);
      const trackerUpdate = trackerRef.current.update(detection.detected, detection.hands);

      if (isMountedRef.current) {
        setDetectedHands(trackerUpdate.hands);
        setHandConfidence(detection.confidence);
      }

      const currentState = stateMachineRef.current;

      // Handle WAITING_FOR_NEXT_SIGN release condition:
      // User must lower or remove hand before the next sign cycle can be triggered.
      if (currentState === RECOGNITION_STATE.WAITING_FOR_NEXT_SIGN) {
        if (!trackerUpdate.isPresent) {
          transitionState(RECOGNITION_STATE.NO_HAND);
        }
        return;
      }

      // Block frame ingestion during active recognition or while result is being displayed
      if (
        currentState === RECOGNITION_STATE.GESTURE_ENDED ||
        currentState === RECOGNITION_STATE.VALIDATING_RECORDING ||
        currentState === RECOGNITION_STATE.RECOGNIZING ||
        currentState === RECOGNITION_STATE.SUCCESS ||
        currentState === RECOGNITION_STATE.UNCERTAIN ||
        currentState === RECOGNITION_STATE.ERROR ||
        currentState === RECOGNITION_STATE.RESULT_DISPLAYED ||
        currentState === RECOGNITION_STATE.RESET_SEQUENCE
      ) {
        return;
      }

      // If recording is active
      if (
        currentState === RECOGNITION_STATE.RECORDING_GESTURE ||
        currentState === "COLLECTING_SEQUENCE"
      ) {
        const nextCount = frameCountRef.current + 1;
        frameCountRef.current = nextCount;
        if (isMountedRef.current) {
          setFrameCount(nextCount);
        }

        // Check maximum duration safety cap (6.0s)
        const elapsed = recordingStartTimeRef.current
          ? Date.now() - recordingStartTimeRef.current
          : 0;
        if (elapsed >= MAX_GESTURE_DURATION_MS) {
          transitionState(RECOGNITION_STATE.GESTURE_ENDED);
          stopRecorder();
          return;
        }

        // Hand absence confirmed stably (5 consecutive lost frames)
        if (!trackerUpdate.isPresent) {
          transitionState(RECOGNITION_STATE.GESTURE_ENDED);
          stopRecorder();
          return;
        }

        return;
      }

      // Hand detected & entering gesture recording
      if (trackerUpdate.isPresent) {
        if (
          currentState === RECOGNITION_STATE.NO_HAND ||
          currentState === RECOGNITION_STATE.HAND_DETECTED
        ) {
          // Clear stale result and errors when fresh gesture starts
          if (isMountedRef.current) {
            setResult(null);
            setError(null);
            setFrameCount(1);
          }
          frameCountRef.current = 1;
          startRecording();
          transitionState(RECOGNITION_STATE.RECORDING_GESTURE);
          return;
        }
      }
    },
    [isCameraActive, enabled, focusZone, startRecording, stopRecorder, transitionState]
  );

  // Reset when camera stopped or unmounted
  useEffect(() => {
    isMountedRef.current = true;
    if (!isCameraActive) {
      resetSequence();
      trackerRef.current.reset();
      transitionState(RECOGNITION_STATE.NO_HAND);
      setDetectedHands([]);
    }
    return () => {
      isMountedRef.current = false;
      stopRecorder();
      if (nextSignTimerRef.current) {
        clearTimeout(nextSignTimerRef.current);
      }
      if (resetTimerRef.current) {
        clearTimeout(resetTimerRef.current);
      }
    };
  }, [isCameraActive, resetSequence, stopRecorder, transitionState]);

  return {
    state,
    stateLabel: RECOGNITION_STATE_LABELS[state] || "Waiting for hand",
    result,
    error,
    frameCount,
    capacity: SEQUENCE_CAPACITY,
    detectedHands,
    handConfidence,
    isHandDetected: trackerRef.current.isHandPresent,
    processFrame,
    resetSequence,
    dispatchRecognition,
  };
}
