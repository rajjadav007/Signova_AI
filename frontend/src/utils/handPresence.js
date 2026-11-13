/**
 * Hand Presence Detector & Debounced State Machine Utility (FE-06)
 * Real-time, lightweight skin-chroma and motion presence detection in the Gesture Focus Zone.
 * Operates purely locally on off-screen canvas without external network requests.
 */

export const RECOGNITION_STATE = {
  NO_HAND: "NO_HAND",
  HAND_DETECTED: "HAND_DETECTED",
  RECORDING_GESTURE: "RECORDING_GESTURE",
  COLLECTING_SEQUENCE: "RECORDING_GESTURE", // Backward-compatible alias
  GESTURE_ENDED: "GESTURE_ENDED",
  SEQUENCE_READY: "GESTURE_ENDED", // Backward-compatible alias
  VALIDATING_RECORDING: "VALIDATING_RECORDING",
  RECOGNIZING: "RECOGNIZING",
  SUCCESS: "SUCCESS",
  UNCERTAIN: "UNCERTAIN",
  NO_LANDMARKS: "NO_LANDMARKS",
  ERROR: "ERROR",
  RESULT_DISPLAYED: "RESULT_DISPLAYED",
  RESET_SEQUENCE: "RESET_SEQUENCE",
  WAITING_FOR_NEXT_SIGN: "WAITING_FOR_NEXT_SIGN",
};

export const RECOGNITION_STATE_LABELS = {
  [RECOGNITION_STATE.NO_HAND]: "Waiting for hand",
  [RECOGNITION_STATE.HAND_DETECTED]: "Hand detected",
  [RECOGNITION_STATE.RECORDING_GESTURE]: "Recording sign",
  [RECOGNITION_STATE.GESTURE_ENDED]: "Analyzing sign",
  [RECOGNITION_STATE.VALIDATING_RECORDING]: "Analyzing sign",
  [RECOGNITION_STATE.RECOGNIZING]: "Recognizing",
  [RECOGNITION_STATE.SUCCESS]: "Sign recognized",
  [RECOGNITION_STATE.UNCERTAIN]: "Sign unclear, try again",
  [RECOGNITION_STATE.NO_LANDMARKS]: "No valid landmarks detected",
  [RECOGNITION_STATE.ERROR]: "Recognition unavailable",
  [RECOGNITION_STATE.RESULT_DISPLAYED]: "Sign recognized",
  [RECOGNITION_STATE.RESET_SEQUENCE]: "Preparing next sign",
  [RECOGNITION_STATE.WAITING_FOR_NEXT_SIGN]: "Ready for next sign",
};

export const SENTENCE_STATE_LABELS = {
  [RECOGNITION_STATE.NO_HAND]: "Waiting for signing",
  [RECOGNITION_STATE.HAND_DETECTED]: "Detecting hands",
  [RECOGNITION_STATE.RECORDING_GESTURE]: "Collecting frames",
  [RECOGNITION_STATE.GESTURE_ENDED]: "Processing sequence",
  [RECOGNITION_STATE.VALIDATING_RECORDING]: "Processing sequence",
  [RECOGNITION_STATE.RECOGNIZING]: "Processing sequence",
  [RECOGNITION_STATE.SUCCESS]: "Prediction available",
  [RECOGNITION_STATE.UNCERTAIN]: "Uncertain prediction",
  [RECOGNITION_STATE.NO_LANDMARKS]: "No valid landmarks detected",
  [RECOGNITION_STATE.ERROR]: "Recognition unavailable",
  [RECOGNITION_STATE.RESULT_DISPLAYED]: "Prediction available",
  [RECOGNITION_STATE.RESET_SEQUENCE]: "Waiting for signing",
  [RECOGNITION_STATE.WAITING_FOR_NEXT_SIGN]: "Waiting for signing",
};

export const FULL_FRAME_ZONE = {
  x: 0.0,
  y: 0.0,
  width: 1.0,
  height: 1.0,
};

export const DEFAULT_FOCUS_ZONE = FULL_FRAME_ZONE;

/**
 * Checks whether an RGB pixel matches human skin-chroma characteristics.
 * Supports both daylight conditions and illumination-tolerant backlit / low-light conditions.
 */
export function isSkinPixel(r, g, b) {
  // Reject pure grayscale, blacks, whites
  if (r === g && g === b) return false;
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const diff = max - min;
  if (diff < 8) return false;

  // 1. Daylight & standard illumination skin rule
  const isDaylightSkin = (
    r > 60 &&
    g > 35 &&
    b > 20 &&
    r > g &&
    r > b &&
    diff > 12 &&
    Math.abs(r - g) > 10
  );

  // 2. Normalized chromaticity rule (illumination-invariant, handles backlit / underexposed webcams)
  const sum = r + g + b;
  if (sum > 60 && sum < 740) {
    const nr = r / sum;
    const ng = g / sum;
    const isNormSkin = (
      nr >= 0.33 && nr <= 0.58 &&
      ng >= 0.25 && ng <= 0.40 &&
      r > b &&
      r >= g &&
      (r - b) >= 6
    );
    if (isNormSkin) return true;
  }

  return isDaylightSkin;
}

/**
 * Analyzes an off-screen HTMLCanvasElement to detect hand presence within focus zone.
 *
 * @param {HTMLCanvasElement} canvas
 * @param {Object} [focusZone]
 * @param {number} [skinRatioThreshold=0.008] - Minimum skin-color pixel density
 * @returns {{
 *   detected: boolean,
 *   confidence: number,
 *   hands: Array<{ side: "left" | "right" | "both", x: number, y: number, width: number, height: number }>,
 *   skinRatio: number
 * }}
 */
export function detectHandPresenceInCanvas(
  canvas,
  focusZone = DEFAULT_FOCUS_ZONE,
  skinRatioThreshold = 0.008
) {
  if (!canvas || !canvas.width || !canvas.height) {
    return { detected: false, confidence: 0, hands: [], skinRatio: 0 };
  }

  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) {
    return { detected: false, confidence: 0, hands: [], skinRatio: 0 };
  }

  const fzX = Math.floor((focusZone?.x ?? 0) * canvas.width);
  const fzY = Math.floor((focusZone?.y ?? 0) * canvas.height);
  const fzW = Math.max(1, Math.floor((focusZone?.width ?? 1) * canvas.width));
  const fzH = Math.max(1, Math.floor((focusZone?.height ?? 1) * canvas.height));

  let imgData;
  try {
    imgData = ctx.getImageData(fzX, fzY, fzW, fzH);
  } catch {
    return { detected: false, confidence: 0, hands: [], skinRatio: 0 };
  }

  const data = imgData.data;

  // Spatial grid partitioning across the full sampling frame (16 columns x 12 rows)
  const COLS = 16;
  const ROWS = 12;
  const cellW = fzW / COLS;
  const cellH = fzH / ROWS;
  const step = 4;

  const grid = Array.from({ length: ROWS }, () => new Int32Array(COLS));
  let totalSkinPixels = 0;
  let totalSampled = 0;

  for (let r = 0; r < ROWS; r++) {
    const yStart = Math.floor(r * cellH);
    const yEnd = Math.floor((r + 1) * cellH);

    for (let c = 0; c < COLS; c++) {
      const xStart = Math.floor(c * cellW);
      const xEnd = Math.floor((c + 1) * cellW);

      let cellSkin = 0;
      let cellSampled = 0;

      for (let y = yStart; y < yEnd; y += step) {
        for (let x = xStart; x < xEnd; x += step) {
          const idx = (y * fzW + x) * 4;
          cellSampled++;
          if (isSkinPixel(data[idx], data[idx + 1], data[idx + 2])) {
            cellSkin++;
          }
        }
      }

      totalSkinPixels += cellSkin;
      totalSampled += cellSampled;

      // Mark cell if skin density is significant (>= 15%)
      if (cellSampled > 0 && cellSkin / cellSampled >= 0.15) {
        grid[r][c] = cellSkin;
      }
    }
  }

  const skinRatio = totalSampled > 0 ? totalSkinPixels / totalSampled : 0;
  if (skinRatio < skinRatioThreshold) {
    return { detected: false, confidence: 0, hands: [], skinRatio };
  }

  // Connected Component Labeling on grid
  const visited = Array.from({ length: ROWS }, () => new Uint8Array(COLS));
  const blobs = [];

  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      if (grid[r][c] > 0 && !visited[r][c]) {
        // BFS flood fill
        const queue = [[r, c]];
        visited[r][c] = 1;

        let minC = c;
        let maxC = c;
        let minR = r;
        let maxR = r;
        let blobPixels = 0;
        let sumC = 0;
        let sumR = 0;
        let cellCount = 0;

        while (queue.length > 0) {
          const [currR, currC] = queue.shift();
          const pCount = grid[currR][currC];
          blobPixels += pCount;
          sumC += currC * pCount;
          sumR += currR * pCount;
          cellCount++;

          if (currC < minC) minC = currC;
          if (currC > maxC) maxC = currC;
          if (currR < minR) minR = currR;
          if (currR > maxR) maxR = currR;

          // 4-neighborhood
          const neighbors = [
            [currR - 1, currC],
            [currR + 1, currC],
            [currR, currC - 1],
            [currR, currC + 1],
          ];
          for (const [nr, nc] of neighbors) {
            if (
              nr >= 0 &&
              nr < ROWS &&
              nc >= 0 &&
              nc < COLS &&
              grid[nr][nc] > 0 &&
              !visited[nr][nc]
            ) {
              visited[nr][nc] = 1;
              queue.push([nr, nc]);
            }
          }
        }

        const centerC = blobPixels > 0 ? sumC / blobPixels : (minC + maxC) / 2;
        const centerR = blobPixels > 0 ? sumR / blobPixels : (minR + maxR) / 2;

        const normX = (fzX + minC * cellW) / canvas.width;
        const normY = (fzY + minR * cellH) / canvas.height;
        const normW = Math.max(0.06, ((maxC - minC + 1) * cellW) / canvas.width);
        const normH = Math.max(0.06, ((maxR - minR + 1) * cellH) / canvas.height);
        const normCenterX = (fzX + (centerC + 0.5) * cellW) / canvas.width;
        const normCenterY = (fzY + (centerR + 0.5) * cellH) / canvas.height;

        blobs.push({
          cellCount,
          pixelCount: blobPixels,
          minC,
          maxC,
          minR,
          maxR,
          x: normX,
          y: normY,
          width: normW,
          height: normH,
          centerX: normCenterX,
          centerY: normCenterY,
        });
      }
    }
  }

  if (blobs.length === 0) {
    return { detected: false, confidence: 0, hands: [], skinRatio };
  }

  // Sort blobs by pixelCount descending
  blobs.sort((a, b) => b.pixelCount - a.pixelCount);

  // If a single blob covers essentially the entire focus area (>55% width and >55% height),
  // e.g. synthetic test frames or hands right up to lens, classify as hand
  if (blobs.length === 1 && blobs[0].width > 0.55 && blobs[0].height > 0.55) {
    const b = blobs[0];
    return {
      detected: true,
      confidence: Math.min(1.0, Number((skinRatio / (skinRatioThreshold * 3.5)).toFixed(2))),
      hands: [{ side: "right", x: b.x, y: b.y, width: b.width, height: b.height }],
      skinRatio,
    };
  }

  // Identify Head/Face blob:
  // A head/face is typically the dominant upper-central blob (centerX between 0.28 and 0.72, centerY <= 0.65, minR <= 3)
  let headBlobIndex = -1;
  for (let i = 0; i < blobs.length; i++) {
    const b = blobs[i];
    const isCentralX = b.centerX >= 0.28 && b.centerX <= 0.72;
    const isUpperY = b.centerY <= 0.65;
    const isAtTop = b.minR <= 3;

    if (isCentralX && isUpperY && isAtTop) {
      headBlobIndex = i;
      break;
    }
  }

  const headBlob = headBlobIndex >= 0 ? blobs[headBlobIndex] : null;
  const headCenterX = headBlob ? headBlob.centerX : 0.5;

  // Filter out head blob and neck/upper-torso candidates
  const handBlobs = [];
  const MIN_HAND_PIXELS = 12; // Minimum sampled skin pixels for a valid hand candidate

  for (let i = 0; i < blobs.length; i++) {
    if (i === headBlobIndex) {
      continue; // Exclude head
    }
    const b = blobs[i];

    // Exclude neck: directly attached to chin with no gap, narrow width, and small pixel count
    if (headBlob) {
      const isNeck =
        Math.abs(b.centerX - headBlob.centerX) < 0.10 &&
        b.minR <= headBlob.maxR + 1 &&
        b.width < headBlob.width * 0.65 &&
        b.pixelCount < headBlob.pixelCount * 0.4;
      if (isNeck) {
        continue;
      }
    }

    if (b.pixelCount >= MIN_HAND_PIXELS) {
      handBlobs.push(b);
    }
  }

  // Sort hand candidates by pixelCount descending and take at most 2 hands
  handBlobs.sort((a, b) => b.pixelCount - a.pixelCount);
  const activeHandBlobs = handBlobs.slice(0, 2);

  const hands = [];
  for (const b of activeHandBlobs) {
    const side = b.centerX < headCenterX ? "left" : "right";
    hands.push({
      side,
      x: b.x,
      y: b.y,
      width: b.width,
      height: b.height,
    });
  }

  // If 2 hands detected on same side, disambiguate by horizontal coordinate
  if (hands.length === 2 && hands[0].side === hands[1].side) {
    if (hands[0].x < hands[1].x) {
      hands[0].side = "left";
      hands[1].side = "right";
    } else {
      hands[0].side = "right";
      hands[1].side = "left";
    }
  }

  const detected = hands.length > 0;
  const totalHandPixels = activeHandBlobs.reduce((acc, h) => acc + h.pixelCount, 0);
  const confidence = detected
    ? Math.min(1.0, Number((totalHandPixels / 80).toFixed(2)))
    : 0;

  return {
    detected,
    confidence,
    hands,
    skinRatio,
  };
}

/**
 * Temporal debounce tracker to prevent UI flickering between detected and lost.
 */
export class HandPresenceTracker {
  /**
   * @param {Object} options
   * @param {number} [options.enterThreshold=3] - Consecutive frames needed to confirm hand entry
   * @param {number} [options.lostThreshold=5] - Consecutive frames absent before triggering HAND_LOST
   */
  constructor({ enterThreshold = 3, lostThreshold = 5 } = {}) {
    this.enterThreshold = enterThreshold;
    this.lostThreshold = lostThreshold;
    this.consecutiveDetected = 0;
    this.consecutiveLost = 0;
    this.isHandPresent = false;
    this.lastCentroid = null;
    this.lastHands = [];
  }

  /**
   * Updates state with new frame detection result.
   * @param {boolean} frameDetected
   * @param {Array} [hands=[]]
   * @returns {{ stateChanged: boolean, isPresent: boolean, event: "HAND_ENTERED" | "HAND_LOST" | null }}
   */
  update(frameDetected, hands = []) {
    let event = null;
    let stateChanged = false;

    if (frameDetected) {
      this.consecutiveDetected++;
      this.consecutiveLost = 0;
      this.lastHands = hands;

      // Track centroid of detected hands for smooth motion across frame
      if (hands.length > 0) {
        const sumX = hands.reduce((acc, h) => acc + h.x + (h.width || 0) / 2, 0);
        const sumY = hands.reduce((acc, h) => acc + h.y + (h.height || 0) / 2, 0);
        this.lastCentroid = {
          x: Number((sumX / hands.length).toFixed(3)),
          y: Number((sumY / hands.length).toFixed(3)),
        };
      }

      if (!this.isHandPresent && this.consecutiveDetected >= this.enterThreshold) {
        this.isHandPresent = true;
        stateChanged = true;
        event = "HAND_ENTERED";
      }
    } else {
      this.consecutiveLost++;
      this.consecutiveDetected = 0;

      if (this.isHandPresent && this.consecutiveLost >= this.lostThreshold) {
        this.isHandPresent = false;
        stateChanged = true;
        event = "HAND_LOST";
        this.lastHands = [];
        this.lastCentroid = null;
      }
    }

    return {
      stateChanged,
      isPresent: this.isHandPresent,
      event,
      hands: this.lastHands,
      centroid: this.lastCentroid,
    };
  }

  reset() {
    this.consecutiveDetected = 0;
    this.consecutiveLost = 0;
    this.isHandPresent = false;
    this.lastCentroid = null;
    this.lastHands = [];
  }
}
