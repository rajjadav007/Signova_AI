import { useState, useEffect, useRef, useCallback } from "react";

export const CAMERA_STATUS = {
  INITIALIZING: "INITIALIZING",
  REQUESTING_PERMISSION: "REQUESTING_PERMISSION",
  ACTIVE: "ACTIVE",
  STOPPED: "STOPPED",
  PERMISSION_DENIED: "PERMISSION_DENIED",
  NOT_AVAILABLE: "NOT_AVAILABLE",
  ERROR: "ERROR",
};

export const CAMERA_STATUS_LABELS = {
  [CAMERA_STATUS.INITIALIZING]: "Initializing Camera",
  [CAMERA_STATUS.REQUESTING_PERMISSION]: "Requesting Permission",
  [CAMERA_STATUS.ACTIVE]: "Camera Active",
  [CAMERA_STATUS.STOPPED]: "Camera Stopped",
  [CAMERA_STATUS.PERMISSION_DENIED]: "Camera Permission Denied",
  [CAMERA_STATUS.NOT_AVAILABLE]: "Camera Not Available",
  [CAMERA_STATUS.ERROR]: "Camera Error",
};

export function useWebcam({
  autoStart = true,
  preferredWidth = 1280,
  preferredHeight = 720,
} = {}) {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const isMountedRef = useRef(true);
  const selectedDeviceIdRef = useRef("");

  const [status, setStatus] = useState(CAMERA_STATUS.INITIALIZING);
  const [stream, setStream] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);
  const [devices, setDevices] = useState([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState("");

  useEffect(() => {
    selectedDeviceIdRef.current = selectedDeviceId;
  }, [selectedDeviceId]);

  // Helper to enumerate available video devices without triggering stream re-acquisitions
  const updateDevices = useCallback(async () => {
    try {
      if (!navigator.mediaDevices?.enumerateDevices) return;
      const allDevices = await navigator.mediaDevices.enumerateDevices();
      const videoInputs = allDevices.filter((d) => d.kind === "videoinput");
      if (isMountedRef.current) {
        setDevices(videoInputs);
        if (videoInputs.length > 0 && !selectedDeviceIdRef.current) {
          selectedDeviceIdRef.current = videoInputs[0].deviceId;
          setSelectedDeviceId(videoInputs[0].deviceId);
        }
      }
    } catch {
      // Non-fatal device enumeration error
    }
  }, []);

  // Stop camera tracks cleanly
  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => {
        try {
          track.stop();
        } catch {
          // ignore already stopped tracks
        }
      });
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    if (isMountedRef.current) {
      setStream(null);
      setStatus(CAMERA_STATUS.STOPPED);
      setErrorMessage(null);
    }
  }, []);

  // Start camera stream with optimal resolution & fallback
  const startCamera = useCallback(
    async (deviceIdToUse) => {
      const targetDeviceId = deviceIdToUse || selectedDeviceIdRef.current;

      // Clean up any existing stream first to avoid duplicate streams
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }

      if (isMountedRef.current) {
        setStatus(CAMERA_STATUS.REQUESTING_PERMISSION);
        setErrorMessage(null);
      }

      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        if (isMountedRef.current) {
          setStatus(CAMERA_STATUS.NOT_AVAILABLE);
          setErrorMessage("Browser does not support mediaDevices.getUserMedia.");
        }
        return;
      }

      // Constraints with preferred resolution and fallback
      const constraints = {
        audio: false,
        video: targetDeviceId
          ? {
              deviceId: { exact: targetDeviceId },
              width: { ideal: preferredWidth },
              height: { ideal: preferredHeight },
            }
          : {
              facingMode: "user",
              width: { ideal: preferredWidth },
              height: { ideal: preferredHeight },
            },
      };

      try {
        let stream;
        try {
          stream = await navigator.mediaDevices.getUserMedia(constraints);
        } catch (initialErr) {
          // If exact device or ideal resolution fails, fallback to basic video constraints
          if (initialErr.name === "OverconstrainedError" || initialErr.name === "ConstraintNotSatisfiedError") {
            stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
          } else {
            throw initialErr;
          }
        }

        if (!isMountedRef.current) {
          // Component unmounted while waiting for user prompt
          stream.getTracks().forEach((t) => t.stop());
          return;
        }

        streamRef.current = stream;

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }

        setStream(stream);
        setStatus(CAMERA_STATUS.ACTIVE);
        setErrorMessage(null);
        await updateDevices();
      } catch (err) {
        if (!isMountedRef.current) return;

        if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
          setStatus(CAMERA_STATUS.PERMISSION_DENIED);
          setErrorMessage("Camera access denied. Please grant camera permission in your browser.");
        } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
          setStatus(CAMERA_STATUS.NOT_AVAILABLE);
          setErrorMessage("No video input devices found on this device.");
        } else {
          setStatus(CAMERA_STATUS.ERROR);
          setErrorMessage(err.message || "Failed to start camera preview.");
        }
      }
    },
    [preferredWidth, preferredHeight, updateDevices]
  );

  // Switch camera when multiple devices exist
  const switchCamera = useCallback(
    async (newDeviceId) => {
      selectedDeviceIdRef.current = newDeviceId;
      setSelectedDeviceId(newDeviceId);
      await startCamera(newDeviceId);
    },
    [startCamera]
  );

  // Auto-start exactly once on mount, clean up on unmount
  useEffect(() => {
    isMountedRef.current = true;
    if (autoStart) {
      startCamera();
    } else {
      setStatus(CAMERA_STATUS.STOPPED);
    }

    return () => {
      isMountedRef.current = false;
      // Stop all tracks when user navigates away
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
      }
      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
    };
  }, [autoStart, startCamera]);

  return {
    videoRef,
    stream,
    status,
    statusLabel: CAMERA_STATUS_LABELS[status] || "Camera Status",
    isActive: status === CAMERA_STATUS.ACTIVE,
    isInitializing:
      status === CAMERA_STATUS.INITIALIZING ||
      status === CAMERA_STATUS.REQUESTING_PERMISSION,
    errorMessage,
    devices,
    selectedDeviceId,
    startCamera,
    stopCamera,
    switchCamera,
  };
}
