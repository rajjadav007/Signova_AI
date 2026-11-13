import { useState, useEffect, useCallback, useRef } from "react";
import { fetchHealth } from "../services/api";

const HEALTH_TIMEOUT_MS = 5000;

export function useBackendHealth(autoFetch = true) {
  const [status, setStatus] = useState("checking"); // "checking" | "connected" | "disconnected"
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const isMountedRef = useRef(true);

  const checkHealth = useCallback(async () => {
    setStatus("checking");
    setError(null);

    // Timeout guard with AbortController
    const controller = new AbortController();
    const timer = setTimeout(() => {
      controller.abort();
    }, HEALTH_TIMEOUT_MS);

    try {
      const response = await Promise.race([
        fetchHealth(),
        new Promise((_, reject) => {
          controller.signal.addEventListener("abort", () => {
            reject(new Error("Health check timed out after 5s"));
          });
        }),
      ]);

      clearTimeout(timer);

      if (!isMountedRef.current) return;

      if (response && (response.status === "healthy" || response.status === "ok")) {
        setStatus("connected");
        setData(response);
      } else {
        setStatus("disconnected");
        setData(response || null);
        setError("Backend reported unhealthy status");
      }
    } catch (err) {
      clearTimeout(timer);
      if (!isMountedRef.current) return;
      setStatus("disconnected");
      setData(null);
      setError(err?.message || "Failed to reach backend service");
    }
  }, []);

  useEffect(() => {
    isMountedRef.current = true;
    if (autoFetch) {
      checkHealth();
    }
    return () => {
      isMountedRef.current = false;
    };
  }, [autoFetch, checkHealth]);

  return {
    status,
    data,
    error,
    isChecking: status === "checking",
    refetch: checkHealth,
  };
}
