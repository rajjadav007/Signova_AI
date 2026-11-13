import { describe, it, expect, beforeEach } from "vitest";
import {
  isSkinPixel,
  detectHandPresenceInCanvas,
  HandPresenceTracker,
  RECOGNITION_STATE,
  RECOGNITION_STATE_LABELS,
  DEFAULT_FOCUS_ZONE,
} from "./handPresence";

describe("Hand Presence Detector (FE-06)", () => {
  it("defines standard UI states and labels matching specification", () => {
    expect(RECOGNITION_STATE_LABELS[RECOGNITION_STATE.NO_HAND]).toBe("Waiting for hand");
    expect(RECOGNITION_STATE_LABELS[RECOGNITION_STATE.HAND_DETECTED]).toBe("Hand detected");
    expect(RECOGNITION_STATE_LABELS[RECOGNITION_STATE.RECORDING_GESTURE]).toBe("Recording sign");
    expect(RECOGNITION_STATE_LABELS[RECOGNITION_STATE.GESTURE_ENDED]).toBe("Analyzing sign");
    expect(RECOGNITION_STATE_LABELS[RECOGNITION_STATE.VALIDATING_RECORDING]).toBe("Analyzing sign");
    expect(RECOGNITION_STATE_LABELS[RECOGNITION_STATE.RECOGNIZING]).toBe("Recognizing");
    expect(RECOGNITION_STATE_LABELS[RECOGNITION_STATE.SUCCESS]).toBe("Sign recognized");
    expect(RECOGNITION_STATE_LABELS[RECOGNITION_STATE.UNCERTAIN]).toBe("Sign unclear, try again");
    expect(RECOGNITION_STATE_LABELS[RECOGNITION_STATE.ERROR]).toBe("Recognition unavailable");
    expect(RECOGNITION_STATE_LABELS[RECOGNITION_STATE.RESET_SEQUENCE]).toBe("Preparing next sign");
    expect(RECOGNITION_STATE_LABELS[RECOGNITION_STATE.WAITING_FOR_NEXT_SIGN]).toBe("Ready for next sign");
  });

  describe("isSkinPixel", () => {
    it("correctly identifies skin tone pixels", () => {
      // Typical skin tones
      expect(isSkinPixel(180, 130, 100)).toBe(true);
      expect(isSkinPixel(120, 80, 50)).toBe(true);
      expect(isSkinPixel(210, 160, 120)).toBe(true);
    });

    it("rejects non-skin pixels (black, white, green, blue)", () => {
      expect(isSkinPixel(0, 0, 0)).toBe(false); // black
      expect(isSkinPixel(255, 255, 255)).toBe(false); // pure white
      expect(isSkinPixel(0, 255, 0)).toBe(false); // pure green
      expect(isSkinPixel(0, 0, 255)).toBe(false); // pure blue
      expect(isSkinPixel(100, 100, 100)).toBe(false); // grey
    });
  });

  describe("detectHandPresenceInCanvas", () => {
    it("handles null or empty canvas safely", () => {
      const res = detectHandPresenceInCanvas(null);
      expect(res.detected).toBe(false);
      expect(res.confidence).toBe(0);
      expect(res.hands).toEqual([]);
    });

    it("detects no hand on blank canvas", () => {
      const mockCtx = {
        getImageData: vi.fn().mockReturnValue({
          data: new Uint8ClampedArray(100 * 100 * 4), // all 0s
        }),
      };
      const canvas = {
        width: 100,
        height: 100,
        getContext: vi.fn().mockReturnValue(mockCtx),
      };

      const res = detectHandPresenceInCanvas(canvas);
      expect(res.detected).toBe(false);
      expect(res.hands).toEqual([]);
    });

    it("detects hand presence when skin pixels populate focus zone", () => {
      // 100x100 buffer
      const buffer = new Uint8ClampedArray(100 * 100 * 4);
      // Fill with skin pixels: R=190, G=135, B=100
      for (let i = 0; i < buffer.length; i += 4) {
        buffer[i] = 190;
        buffer[i + 1] = 135;
        buffer[i + 2] = 100;
        buffer[i + 3] = 255;
      }
      const mockCtx = {
        getImageData: vi.fn().mockReturnValue({
          data: buffer,
        }),
      };
      const canvas = {
        width: 100,
        height: 100,
        getContext: vi.fn().mockReturnValue(mockCtx),
      };

      const res = detectHandPresenceInCanvas(canvas, DEFAULT_FOCUS_ZONE, 0.01);
      expect(res.detected).toBe(true);
      expect(res.confidence).toBeGreaterThan(0.5);
      expect(res.hands.length).toBeGreaterThanOrEqual(1);
    });

    it("isolates face in center and reports no hand when only head/face is present", () => {
      // 200x200 canvas
      const width = 200;
      const height = 200;
      const fzW = Math.floor(DEFAULT_FOCUS_ZONE.width * width);
      const fzH = Math.floor(DEFAULT_FOCUS_ZONE.height * height);
      const buffer = new Uint8ClampedArray(fzW * fzH * 4);

      // Draw head blob in center top (x: 45..95, y: 5..45 inside focus zone)
      for (let y = 5; y < 45; y++) {
        for (let x = 45; x < 95; x++) {
          const idx = (y * fzW + x) * 4;
          buffer[idx] = 190;
          buffer[idx + 1] = 135;
          buffer[idx + 2] = 100;
          buffer[idx + 3] = 255;
        }
      }

      const mockCtx = {
        getImageData: vi.fn().mockReturnValue({ data: buffer }),
      };
      const canvas = {
        width,
        height,
        getContext: vi.fn().mockReturnValue(mockCtx),
      };

      const res = detectHandPresenceInCanvas(canvas, DEFAULT_FOCUS_ZONE, 0.01);
      // Head must be isolated; hands must be empty so user sitting idle does not auto-record
      expect(res.detected).toBe(false);
      expect(res.hands).toEqual([]);
    });

    it("detects hand when hand blob appears outside the head region", () => {
      const width = 200;
      const height = 200;
      const fzW = Math.floor(DEFAULT_FOCUS_ZONE.width * width);
      const fzH = Math.floor(DEFAULT_FOCUS_ZONE.height * height);
      const buffer = new Uint8ClampedArray(fzW * fzH * 4);

      // Head in center
      for (let y = 5; y < 45; y++) {
        for (let x = 45; x < 95; x++) {
          const idx = (y * fzW + x) * 4;
          buffer[idx] = 190;
          buffer[idx + 1] = 135;
          buffer[idx + 2] = 100;
          buffer[idx + 3] = 255;
        }
      }

      // Hand on left side (x: 10..35, y: 60..100)
      for (let y = 60; y < 100; y++) {
        for (let x = 10; x < 35; x++) {
          const idx = (y * fzW + x) * 4;
          buffer[idx] = 190;
          buffer[idx + 1] = 135;
          buffer[idx + 2] = 100;
          buffer[idx + 3] = 255;
        }
      }

      const mockCtx = {
        getImageData: vi.fn().mockReturnValue({ data: buffer }),
      };
      const canvas = {
        width,
        height,
        getContext: vi.fn().mockReturnValue(mockCtx),
      };

      const res = detectHandPresenceInCanvas(canvas, DEFAULT_FOCUS_ZONE, 0.01);
      expect(res.detected).toBe(true);
      expect(res.hands.length).toBe(1);
      expect(res.hands[0].side).toBe("left");
    });

    describe("Full-Frame Hand Detection Across 11 Mandatory Positions", () => {
      // Helper to generate mock canvas with precise skin blobs
      function createMockCanvas(width, height, blobsToDraw = []) {
        const buffer = new Uint8ClampedArray(width * height * 4);
        for (const b of blobsToDraw) {
          const x0 = Math.max(0, Math.min(width - 1, b.x));
          const x1 = Math.max(0, Math.min(width, b.x + b.w));
          const y0 = Math.max(0, Math.min(height - 1, b.y));
          const y1 = Math.max(0, Math.min(height, b.y + b.h));
          for (let y = y0; y < y1; y++) {
            for (let x = x0; x < x1; x++) {
              const idx = (y * width + x) * 4;
              buffer[idx] = b.r ?? 190;
              buffer[idx + 1] = b.g ?? 135;
              buffer[idx + 2] = b.b ?? 100;
              buffer[idx + 3] = 255;
            }
          }
        }
        const mockCtx = {
          getImageData: vi.fn().mockImplementation((sx, sy, sw, sh) => {
            if (sx === 0 && sy === 0 && sw === width && sh === height) {
              return { data: buffer };
            }
            const sub = new Uint8ClampedArray(sw * sh * 4);
            for (let y = 0; y < sh; y++) {
              for (let x = 0; x < sw; x++) {
                const srcIdx = ((sy + y) * width + (sx + x)) * 4;
                const dstIdx = (y * sw + x) * 4;
                sub[dstIdx] = buffer[srcIdx];
                sub[dstIdx + 1] = buffer[srcIdx + 1];
                sub[dstIdx + 2] = buffer[srcIdx + 2];
                sub[dstIdx + 3] = buffer[srcIdx + 3];
              }
            }
            return { data: sub };
          }),
        };
        return {
          width,
          height,
          getContext: vi.fn().mockReturnValue(mockCtx),
        };
      }

      // Standard head blob for positions with user sitting in front of camera
      const standardHead = { x: 75, y: 10, w: 50, h: 45 };

      it("1. detects hand in center of frame (below face)", () => {
        const canvas = createMockCanvas(200, 200, [
          standardHead,
          { x: 80, y: 110, w: 40, h: 35 }, // Center hand
        ]);
        const res = detectHandPresenceInCanvas(canvas);
        expect(res.detected).toBe(true);
        expect(res.hands.length).toBe(1);
        expect(res.hands[0].y).toBeGreaterThan(0.45);
      });

      it("2. detects hand at far left edge of frame", () => {
        const canvas = createMockCanvas(200, 200, [
          standardHead,
          { x: 2, y: 90, w: 25, h: 35 }, // Far left (x <= 15% - previously blind zone)
        ]);
        const res = detectHandPresenceInCanvas(canvas);
        expect(res.detected).toBe(true);
        expect(res.hands.length).toBe(1);
        expect(res.hands[0].side).toBe("left");
        expect(res.hands[0].x).toBeLessThan(0.15);
      });

      it("3. detects hand at far right edge of frame", () => {
        const canvas = createMockCanvas(200, 200, [
          standardHead,
          { x: 172, y: 90, w: 25, h: 35 }, // Far right (x >= 85% - previously blind zone)
        ]);
        const res = detectHandPresenceInCanvas(canvas);
        expect(res.detected).toBe(true);
        expect(res.hands.length).toBe(1);
        expect(res.hands[0].side).toBe("right");
        expect(res.hands[0].x).toBeGreaterThan(0.80);
      });

      it("4. detects hand at top of frame (raised hand)", () => {
        const canvas = createMockCanvas(200, 200, [
          standardHead,
          { x: 10, y: 2, w: 30, h: 30 }, // Top raised hand (y <= 15%)
        ]);
        const res = detectHandPresenceInCanvas(canvas);
        expect(res.detected).toBe(true);
        expect(res.hands.length).toBe(1);
        expect(res.hands[0].y).toBeLessThan(0.15);
      });

      it("5. detects hand at bottom of frame", () => {
        const canvas = createMockCanvas(200, 200, [
          standardHead,
          { x: 85, y: 165, w: 30, h: 30 }, // Bottom hand (y >= 80%)
        ]);
        const res = detectHandPresenceInCanvas(canvas);
        expect(res.detected).toBe(true);
        expect(res.hands.length).toBe(1);
        expect(res.hands[0].y).toBeGreaterThan(0.75);
      });

      it("6a. detects hand at top-left corner", () => {
        const canvas = createMockCanvas(200, 200, [
          standardHead,
          { x: 2, y: 2, w: 26, h: 26 }, // Top-Left corner
        ]);
        const res = detectHandPresenceInCanvas(canvas);
        expect(res.detected).toBe(true);
        expect(res.hands[0].x).toBeLessThan(0.15);
        expect(res.hands[0].y).toBeLessThan(0.15);
      });

      it("6b. detects hand at top-right corner", () => {
        const canvas = createMockCanvas(200, 200, [
          standardHead,
          { x: 172, y: 2, w: 26, h: 26 }, // Top-Right corner
        ]);
        const res = detectHandPresenceInCanvas(canvas);
        expect(res.detected).toBe(true);
        expect(res.hands[0].x).toBeGreaterThan(0.80);
        expect(res.hands[0].y).toBeLessThan(0.15);
      });

      it("6c. detects hand at bottom-left corner", () => {
        const canvas = createMockCanvas(200, 200, [
          standardHead,
          { x: 2, y: 170, w: 26, h: 26 }, // Bottom-Left corner
        ]);
        const res = detectHandPresenceInCanvas(canvas);
        expect(res.detected).toBe(true);
        expect(res.hands[0].x).toBeLessThan(0.15);
        expect(res.hands[0].y).toBeGreaterThan(0.80);
      });

      it("6d. detects hand at bottom-right corner", () => {
        const canvas = createMockCanvas(200, 200, [
          standardHead,
          { x: 172, y: 170, w: 26, h: 26 }, // Bottom-Right corner
        ]);
        const res = detectHandPresenceInCanvas(canvas);
        expect(res.detected).toBe(true);
        expect(res.hands[0].x).toBeGreaterThan(0.80);
        expect(res.hands[0].y).toBeGreaterThan(0.80);
      });

      it("7. detects both hands simultaneously (left and right)", () => {
        const canvas = createMockCanvas(200, 200, [
          standardHead,
          { x: 15, y: 90, w: 30, h: 30 },  // Left hand
          { x: 155, y: 90, w: 30, h: 30 }, // Right hand
        ]);
        const res = detectHandPresenceInCanvas(canvas);
        expect(res.detected).toBe(true);
        expect(res.hands.length).toBe(2);
        const sides = res.hands.map((h) => h.side).sort();
        expect(sides).toEqual(["left", "right"]);
      });

      it("8. detects open palm and varying signing hand shapes", () => {
        // Large open palm
        const palmCanvas = createMockCanvas(200, 200, [
          standardHead,
          { x: 80, y: 95, w: 45, h: 45 },
        ]);
        const palmRes = detectHandPresenceInCanvas(palmCanvas);
        expect(palmRes.detected).toBe(true);
        expect(palmRes.confidence).toBeGreaterThan(0.4);

        // Compact signing shape / fist
        const fistCanvas = createMockCanvas(200, 200, [
          standardHead,
          { x: 85, y: 100, w: 26, h: 26 },
        ]);
        const fistRes = detectHandPresenceInCanvas(fistCanvas);
        expect(fistRes.detected).toBe(true);
      });

      it("9. tracks moving hand across the frame with centroid updates", () => {
        const tracker = new HandPresenceTracker({ enterThreshold: 2, lostThreshold: 3 });

        // Frame 1: hand on left
        const c1 = createMockCanvas(200, 200, [
          standardHead,
          { x: 20, y: 90, w: 30, h: 30 },
        ]);
        const r1 = detectHandPresenceInCanvas(c1);
        const u1 = tracker.update(r1.detected, r1.hands);

        // Frame 2: hand moves to center
        const c2 = createMockCanvas(200, 200, [
          standardHead,
          { x: 85, y: 90, w: 30, h: 30 },
        ]);
        const r2 = detectHandPresenceInCanvas(c2);
        const u2 = tracker.update(r2.detected, r2.hands);

        // Frame 3: hand moves to right
        const c3 = createMockCanvas(200, 200, [
          standardHead,
          { x: 150, y: 90, w: 30, h: 30 },
        ]);
        const r3 = detectHandPresenceInCanvas(c3);
        const u3 = tracker.update(r3.detected, r3.hands);

        expect(u3.isPresent).toBe(true);
        expect(u3.centroid).toBeDefined();
        // Centroid X should have shifted from ~0.17 to ~0.82
        expect(u3.centroid.x).toBeGreaterThan(0.70);
      });

      it("10. prevents false detection when only face and head are present", () => {
        const canvas = createMockCanvas(200, 200, [
          standardHead, // Only face in upper center
        ]);
        const res = detectHandPresenceInCanvas(canvas);
        expect(res.detected).toBe(false);
        expect(res.hands).toEqual([]);
      });

      it("11. prevents false detection on skin-colored background or empty scene", () => {
        // Blank non-skin canvas (white/grey)
        const blankCanvas = createMockCanvas(200, 200, []);
        const resBlank = detectHandPresenceInCanvas(blankCanvas);
        expect(resBlank.detected).toBe(false);
        expect(resBlank.hands).toEqual([]);

        // Canvas with blue / green / dark objects (curtains, clothing)
        const objectCanvas = createMockCanvas(200, 200, [
          { x: 0, y: 0, w: 200, h: 200, r: 40, g: 80, b: 180 }, // blue curtain
          { x: 50, y: 50, w: 80, h: 80, r: 20, g: 150, b: 30 }, // green clothing
        ]);
        const resObjects = detectHandPresenceInCanvas(objectCanvas);
        expect(resObjects.detected).toBe(false);
        expect(resObjects.hands).toEqual([]);
      });
    });
  });

  describe("HandPresenceTracker", () => {
    let tracker;

    beforeEach(() => {
      tracker = new HandPresenceTracker({ enterThreshold: 3, lostThreshold: 4 });
    });

    it("starts in NO_HAND state", () => {
      expect(tracker.isHandPresent).toBe(false);
    });

    it("requires consecutive detected frames to trigger HAND_ENTERED (debounced)", () => {
      // Frame 1
      let res = tracker.update(true, [{ side: "right" }]);
      expect(tracker.isHandPresent).toBe(false);
      expect(res.event).toBeNull();

      // Frame 2
      res = tracker.update(true, [{ side: "right" }]);
      expect(tracker.isHandPresent).toBe(false);
      expect(res.event).toBeNull();

      // Frame 3 (reaches enterThreshold: 3)
      res = tracker.update(true, [{ side: "right" }]);
      expect(tracker.isHandPresent).toBe(true);
      expect(res.stateChanged).toBe(true);
      expect(res.event).toBe("HAND_ENTERED");
    });

    it("ignores transient single-frame detection flicker", () => {
      tracker.update(true); // 1
      tracker.update(false); // resets counter
      tracker.update(true); // 1
      tracker.update(true); // 2
      expect(tracker.isHandPresent).toBe(false);
    });

    it("requires consecutive missing frames before confirming HAND_LOST", () => {
      // First establish presence
      tracker.update(true);
      tracker.update(true);
      tracker.update(true);
      expect(tracker.isHandPresent).toBe(true);

      // 1 frame lost - still present
      let res = tracker.update(false);
      expect(tracker.isHandPresent).toBe(true);
      expect(res.event).toBeNull();

      // 2 frames lost - still present
      res = tracker.update(false);
      expect(tracker.isHandPresent).toBe(true);

      // 3 frames lost - still present
      res = tracker.update(false);
      expect(tracker.isHandPresent).toBe(true);

      // 4 frames lost (reaches lostThreshold: 4)
      res = tracker.update(false);
      expect(tracker.isHandPresent).toBe(false);
      expect(res.stateChanged).toBe(true);
      expect(res.event).toBe("HAND_LOST");
    });

    it("resets state completely on reset()", () => {
      tracker.update(true);
      tracker.update(true);
      tracker.update(true);
      expect(tracker.isHandPresent).toBe(true);

      tracker.reset();
      expect(tracker.isHandPresent).toBe(false);
      expect(tracker.consecutiveDetected).toBe(0);
      expect(tracker.consecutiveLost).toBe(0);
    });
  });
});
