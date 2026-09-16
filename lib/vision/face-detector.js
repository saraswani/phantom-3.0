/**
 * PrivacyShield - Local Face & Visual PII Detection Engine (Component 2)
 *
 * Strategy: Load BlazeFace model weights directly into memory using
 * chrome.runtime.getURL + fetch (allowed in content scripts), then construct
 * a tf.io.fromMemory() handler so TFJS never tries to re-fetch anything.
 *
 * Fallback chain:
 *   1. Local bundled weights via chrome.runtime.getURL (offline, CSP-safe)
 *   2. TFHub remote (if local fails)
 *   3. Enhanced heuristic skin-tone detector (if both ML paths fail)
 */
(function() {
  'use strict';

  class LocalFaceDetector {
    constructor() {
      this.model = null;
      this.isModelLoaded = false;
      this.isLoading = false;
      this.activeBackend = 'Not Initialized';
      this.detectorStatus = 'unloaded';
      this.lastInferenceTimeMs = 0;
      this.detectionCount = 0;
      this.telemetryBreakdown = {};
    }

    /**
     * Fetches a URL and returns an ArrayBuffer.
     * Works from content-script context for chrome-extension:// URLs.
     */
    async _fetchBuffer(url) {
      const resp = await fetch(url);
      if (!resp.ok) throw new Error(`HTTP ${resp.status} for ${url}`);
      return resp.arrayBuffer();
    }

    /**
     * Loads the local bundled BlazeFace model weights into memory and returns
     * a tf.io.IOHandler so TFJS loads from ArrayBuffer without any network call.
     * Returns null on any error.
     */
    async _loadLocalModelHandler() {
      try {
        const modelJsonUrl = chrome.runtime.getURL('lib/vision/model/model.json');
        const weightsUrl = chrome.runtime.getURL('lib/vision/model/group1-shard1of1.bin');

        console.log('[PrivacyShield] Fetching local model from:', modelJsonUrl);

        const [modelJsonBuf, weightsBuf] = await Promise.all([
          this._fetchBuffer(modelJsonUrl),
          this._fetchBuffer(weightsUrl)
        ]);

        // Parse model topology JSON
        const modelJsonText = new TextDecoder('utf-8').decode(modelJsonBuf);
        const modelArtifacts = JSON.parse(modelJsonText);

        // Reconstruct modelArtifacts with inline weight data
        const weightData = weightsBuf;

        // Use tf.io.fromMemory to give TFJS pre-loaded data — no network
        const handler = tf.io.fromMemory({
          modelTopology: modelArtifacts.modelTopology || modelArtifacts,
          weightSpecs: modelArtifacts.weightsManifest
            ? modelArtifacts.weightsManifest[0].weights
            : [],
          weightData: weightData,
          format: modelArtifacts.format,
          generatedBy: modelArtifacts.generatedBy,
          convertedBy: modelArtifacts.convertedBy,
          signature: modelArtifacts.signature
        });

        return handler;
      } catch (err) {
        console.warn('[PrivacyShield] Local model handler error:', err.message || err);
        return null;
      }
    }

    /**
     * Initializes BlazeFace:
     * 1. Local bundled weights (via in-memory IO handler)
     * 2. TFHub fallback
     * 3. Heuristic fallback
     */
    async init() {
      if (this.isModelLoaded || this.isLoading) return;
      this.isLoading = true;
      this.detectorStatus = 'loading';

      try {
        if (typeof tf === 'undefined' || typeof blazeface === 'undefined') {
          throw new Error('TF.js or BlazeFace not loaded');
        }

        // Backend setup: try WebGL (GPU), fall back to CPU
        let backendReady = false;
        try {
          backendReady = await tf.setBackend('webgl');
          if (backendReady) await tf.ready();
        } catch (_) {}

        if (!backendReady || tf.getBackend() !== 'webgl') {
          try {
            await tf.setBackend('cpu');
            await tf.ready();
          } catch (_) {}
        }

        const backend = tf.getBackend() || 'cpu';
        this.activeBackend = `TensorFlow.js (${backend.toUpperCase()})`;
        console.log(`[PrivacyShield] TF backend: ${backend}`);

        // ── Tier 1: Load model weights directly from extension bundle ─────────
        let modelLoaded = false;

        if (typeof chrome !== 'undefined' && chrome.runtime && chrome.runtime.getURL) {
          const handler = await this._loadLocalModelHandler();
          if (handler) {
            try {
              // blazeface.load() passes modelUrl to tf.loadGraphModel internally.
              // We can't pass an IOHandler directly to blazeface.load().
              // Instead, register a custom URL scheme handler via tf.io.registerLoadRouter.
              // Simpler: use a fresh tf.loadGraphModel call then wrap it as a BlazeFaceModel.
              const tfModel = await tf.loadGraphModel(handler);
              // Wrap in a minimal BlazeFace-compatible interface
              this.model = this._createBlazeFaceWrapper(tfModel);
              modelLoaded = true;
              this.detectorStatus = 'blazeface_ready';
              console.log('[PrivacyShield] BlazeFace loaded from local bundle ✓');
            } catch (wrapErr) {
              console.warn('[PrivacyShield] Local model wrap failed:', wrapErr.message || wrapErr);
            }
          }
        }

        // ── Tier 2: Let blazeface.load() fetch from TFHub or local URL ────────
        if (!modelLoaded) {
          // Try local URL first (chrome.runtime.getURL returns a proper fetchable URL)
          if (typeof chrome !== 'undefined' && chrome.runtime && chrome.runtime.getURL) {
            try {
              const localUrl = chrome.runtime.getURL('lib/vision/model/model.json');
              this.model = await blazeface.load({
                modelUrl: localUrl,
                maxFaces: 20,
                scoreThreshold: 0.50,
                iouThreshold: 0.30
              });
              modelLoaded = true;
              this.detectorStatus = 'blazeface_ready';
              console.log('[PrivacyShield] BlazeFace loaded via blazeface.load(localUrl) ✓');
            } catch (bfLocalErr) {
              console.warn('[PrivacyShield] blazeface.load(localUrl) failed:', bfLocalErr.message || bfLocalErr);
            }
          }
        }

        // ── Tier 3: TFHub (requires internet) ─────────────────────────────────
        if (!modelLoaded) {
          try {
            this.model = await blazeface.load({
              maxFaces: 20,
              scoreThreshold: 0.50,
              iouThreshold: 0.30
            });
            modelLoaded = true;
            this.detectorStatus = 'blazeface_ready';
            console.log('[PrivacyShield] BlazeFace loaded from TFHub ✓');
          } catch (hubErr) {
            console.warn('[PrivacyShield] TFHub load failed:', hubErr.message || hubErr);
          }
        }

        if (!modelLoaded) throw new Error('All BlazeFace load paths failed');
        this.isModelLoaded = true;

      } catch (err) {
        console.error('[PrivacyShield] Face detector init failed:', err.message || err);
        this.activeBackend = 'Enhanced Heuristic Fallback';
        this.detectorStatus = 'heuristic_fallback';
      } finally {
        this.isLoading = false;
      }
    }

    /**
     * Creates a minimal BlazeFace-compatible wrapper around a raw tf.GraphModel.
     * BlazeFace's estimateFaces() is reproduced here at the low level.
     */
    _createBlazeFaceWrapper(tfModel) {
      return {
        _tfModel: tfModel,

        async estimateFaces(input, returnTensors = false) {
          const results = [];
          try {
            // Prepare tensor from input (HTMLImageElement, HTMLCanvasElement, etc.)
            let inputTensor;
            try {
              inputTensor = tf.browser.fromPixels(input);
            } catch (pixErr) {
              return results;
            }

            // BlazeFace expects [1, 128, 128, 3] float32 input normalized to [-1, 1]
            const resized = tf.image.resizeBilinear(inputTensor, [128, 128]);
            const normalized = tf.div(tf.sub(tf.div(resized, 127.5), 1), 1);
            const batched = tf.expandDims(normalized, 0);

            // Run inference
            const output = tfModel.predict(batched);

            // Clean up input tensors
            inputTensor.dispose();
            resized.dispose();
            normalized.dispose();
            batched.dispose();

            // BlazeFace output processing is complex; use simplified box parsing
            // Output shape depends on model version: typically [1, N, 17] or [1, 1, 17]
            if (output) {
              const outputArray = await output.array();
              output.dispose();

              const boxes = Array.isArray(outputArray[0]) ? outputArray[0] : [outputArray];
              for (const box of boxes) {
                if (!box || box.length < 5) continue;
                // Format: [x1, y1, x2, y2, score, ...landmarks]
                const score = box[4] || 0;
                if (score < 0.4) continue;

                const x1 = box[0] * 128;
                const y1 = box[1] * 128;
                const x2 = box[2] * 128;
                const y2 = box[3] * 128;

                results.push({
                  topLeft: [x1, y1],
                  bottomRight: [x2, y2],
                  probability: [score],
                  landmarks: []
                });
              }
            }
          } catch (inferErr) {
            console.warn('[PrivacyShield] tfModel inference failed:', inferErr.message || inferErr);
          }
          return results;
        }
      };
    }

    /**
     * Calculates the rendered content box and scaling for images with object-fit.
     */
    getObjectFitLayout(img, rect) {
      const nw = img.naturalWidth || rect.width;
      const nh = img.naturalHeight || rect.height;
      if (nw === 0 || nh === 0) {
        return { scaleX: 1, scaleY: 1, offsetX: 0, offsetY: 0, naturalWidth: rect.width, naturalHeight: rect.height };
      }

      let objectFit = 'fill';
      try {
        const computed = window.getComputedStyle(img);
        objectFit = computed.objectFit || 'fill';
      } catch (e) { objectFit = 'fill'; }

      if (objectFit === 'cover') {
        const scale = Math.max(rect.width / nw, rect.height / nh);
        const renderW = nw * scale;
        const renderH = nh * scale;
        return {
          scaleX: scale, scaleY: scale,
          offsetX: (rect.width - renderW) / 2,
          offsetY: (rect.height - renderH) / 2,
          naturalWidth: nw, naturalHeight: nh
        };
      } else if (objectFit === 'contain') {
        const scale = Math.min(rect.width / nw, rect.height / nh);
        const renderW = nw * scale;
        const renderH = nh * scale;
        return {
          scaleX: scale, scaleY: scale,
          offsetX: (rect.width - renderW) / 2,
          offsetY: (rect.height - renderH) / 2,
          naturalWidth: nw, naturalHeight: nh
        };
      }
      return {
        scaleX: rect.width / nw, scaleY: rect.height / nh,
        offsetX: 0, offsetY: 0,
        naturalWidth: nw, naturalHeight: nh
      };
    }

    /**
     * Draws image to a letterboxed square canvas for BlazeFace inference.
     */
    _drawLetterboxedCanvas(imgEl, targetSize = 256) {
      const nw = imgEl.naturalWidth || imgEl.width || targetSize;
      const nh = imgEl.naturalHeight || imgEl.height || targetSize;
      const canvas = document.createElement('canvas');
      canvas.width = targetSize;
      canvas.height = targetSize;
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = '#000';
      ctx.fillRect(0, 0, targetSize, targetSize);

      const scale = Math.min(targetSize / nw, targetSize / nh);
      const drawW = nw * scale;
      const drawH = nh * scale;
      const offsetX = (targetSize - drawW) / 2;
      const offsetY = (targetSize - drawH) / 2;
      ctx.drawImage(imgEl, offsetX, offsetY, drawW, drawH);
      return { canvas, scale, offsetX, offsetY, nw, nh };
    }

    /**
     * Enhanced multi-region heuristic: skin-tone grid analysis.
     * Returns coordinates in IMAGE natural pixel space.
     */
    detectHeuristicFaces(canvas, ctx) {
      const width = canvas.width;
      const height = canvas.height;
      if (width < 24 || height < 24) return [];

      let imgData;
      try { imgData = ctx.getImageData(0, 0, width, height); }
      catch (_) { return []; }
      const data = imgData.data;

      const gridCols = 6, gridRows = 6;
      const cellW = Math.floor(width / gridCols);
      const cellH = Math.floor(height / gridRows);
      if (cellW < 2 || cellH < 2) return [];

      const skinGrid = Array.from({ length: gridRows }, () => new Array(gridCols).fill(0));
      const step = Math.max(1, Math.floor(Math.min(width, height) / 120));

      for (let y = 0; y < height; y += step) {
        for (let x = 0; x < width; x += step) {
          const idx = (y * width + x) * 4;
          const r = data[idx], g = data[idx + 1], b = data[idx + 2];
          // Skin detection: covers light to dark skin tones
          const isSkin = (
            r > 60 && g > 20 && b > 10 &&
            r > b &&
            (Math.max(r, g, b) - Math.min(r, g, b)) > 10 &&
            Math.abs(r - g) > 5 && r > g
          );
          if (isSkin) {
            const col = Math.min(gridCols - 1, Math.floor(x / cellW));
            const row = Math.min(gridRows - 1, Math.floor(y / cellH));
            skinGrid[row][col]++;
          }
        }
      }

      const samplesPerCell = Math.max(1, (cellW / step) * (cellH / step));
      const threshold = 0.08;
      let minRow = gridRows, maxRow = -1, minCol = gridCols, maxCol = -1;
      let totalSkinCells = 0;

      for (let r = 0; r < gridRows; r++) {
        for (let c = 0; c < gridCols; c++) {
          if (skinGrid[r][c] / samplesPerCell > threshold) {
            totalSkinCells++;
            if (r < minRow) minRow = r;
            if (r > maxRow) maxRow = r;
            if (c < minCol) minCol = c;
            if (c > maxCol) maxCol = c;
          }
        }
      }

      if (totalSkinCells < 2 || maxRow < minRow || maxCol < minCol) return [];

      const fx = minCol * cellW;
      const fy = minRow * cellH;
      const fw = (maxCol - minCol + 1) * cellW;
      const fh = (maxRow - minRow + 1) * cellH;
      const aspect = fh / fw;

      if (aspect < 0.4 || aspect > 2.8) return [];

      return [{
        x: fx, y: fy,
        width: fw, height: fh,
        confidence: Math.min(0.72, 0.40 + totalSkinCells * 0.04),
        isHeuristic: true
      }];
    }

    /**
     * Detect faces in an image element using BlazeFace or heuristic.
     * Returns coordinates in IMAGE NATURAL PIXEL SPACE.
     */
    async detectFacesInElement(element) {
      const startTime = performance.now();
      const faceBoxes = [];

      try {
        if (this.detectorStatus === 'blazeface_ready' && this.model) {
          try {
            const nw = element.naturalWidth || element.width || 128;
            const nh = element.naturalHeight || element.height || 128;
            const targetSize = Math.min(512, Math.max(128, Math.max(nw, nh)));
            const { canvas, scale, offsetX, offsetY } = this._drawLetterboxedCanvas(element, targetSize);

            const predictions = await this.model.estimateFaces(canvas, false);

            for (const pred of predictions) {
              const [lbX, lbY] = pred.topLeft;
              const [lbX2, lbY2] = pred.bottomRight;
              const prob = (pred.probability && pred.probability[0]) ? pred.probability[0] : 0.85;
              if (prob < 0.40) continue;

              // Reverse letterbox transform → natural pixel coords
              const naturalX = (lbX - offsetX) / scale;
              const naturalY = (lbY - offsetY) / scale;
              const naturalX2 = (lbX2 - offsetX) / scale;
              const naturalY2 = (lbY2 - offsetY) / scale;

              const x = Math.max(0, Math.round(naturalX));
              const y = Math.max(0, Math.round(naturalY));
              const w = Math.max(1, Math.round(naturalX2 - naturalX));
              const h = Math.max(1, Math.round(naturalY2 - naturalY));

              if (w > 2 && h > 2) {
                faceBoxes.push({ x, y, width: w, height: h, confidence: prob, isHeuristic: false });
              }
            }
          } catch (modelErr) {
            console.warn('[PrivacyShield] Model inference error:', modelErr.message || modelErr);
          }
        }

        // Fallback heuristic if ML returned nothing
        if (faceBoxes.length === 0) {
          try {
            const nw = element.naturalWidth || element.width || 128;
            const nh = element.naturalHeight || element.height || 128;
            const canvas = document.createElement('canvas');
            canvas.width = nw; canvas.height = nh;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(element, 0, 0, nw, nh);
            faceBoxes.push(...this.detectHeuristicFaces(canvas, ctx));
          } catch (_) {}
        }
      } catch (_) {}

      this.lastInferenceTimeMs = Math.round((performance.now() - startTime) * 100) / 100;
      this.detectionCount += faceBoxes.length;
      return faceBoxes;
    }

    /**
     * Applies biometric head padding to expand tight eye-to-chin box
     * to cover forehead, ears, and jaw.
     */
    _applyBiometricPadding(face, imgW, imgH) {
      const padTop = Math.round(face.height * 0.55);    // forehead + hair
      const padBottom = Math.round(face.height * 0.30); // chin + jaw
      const padSide = Math.round(face.width * 0.30);    // ears + temples

      return {
        x: Math.max(0, face.x - padSide),
        y: Math.max(0, face.y - padTop),
        width: Math.min(imgW, face.x + face.width + padSide) - Math.max(0, face.x - padSide),
        height: Math.min(imgH, face.y + face.height + padBottom) - Math.max(0, face.y - padTop)
      };
    }

    /**
     * Scans a single image element and returns face bounding boxes
     * in VIEWPORT coordinate space for DOM overlay positioning.
     */
    async scanSingleImage(img) {
      if (!img || img.closest('#privacyshield-root')) return [];

      const rect = img.getBoundingClientRect();
      if (rect.width < 32 || rect.height < 32) return [];
      if (img.tagName && img.tagName.toLowerCase() === 'img') {
        if (!img.complete || img.naturalWidth === 0) return [];
      }

      // Attempt CORS-anonymous reload to allow canvas pixel access
      let targetImg = img;
      if (img.tagName && img.tagName.toLowerCase() === 'img' && img.src && !img.crossOrigin) {
        try {
          const corsImg = new Image();
          corsImg.crossOrigin = 'anonymous';
          const corsResult = await Promise.race([
            new Promise((res, rej) => {
              corsImg.onload = () => res(corsImg);
              corsImg.onerror = () => rej(new Error('CORS'));
              // src MUST be set AFTER handlers are registered
              corsImg.src = img.src;
            }),
            new Promise((_, rej) => setTimeout(() => rej(new Error('timeout')), 1200))
          ]);
          targetImg = corsResult;
        } catch (_) {
          targetImg = img;
        }
      }

      // Detect faces (natural pixel space)
      const rawFaces = await this.detectFacesInElement(targetImg);
      if (rawFaces.length === 0) return [];

      const fit = this.getObjectFitLayout(img, rect);
      const nw = fit.naturalWidth || img.naturalWidth || rect.width;
      const nh = fit.naturalHeight || img.naturalHeight || rect.height;

      const viewportFaces = [];
      for (const f of rawFaces) {
        const padded = this._applyBiometricPadding(f, nw, nh);

        // Map natural pixels → viewport CSS pixels
        const vpX = rect.left + fit.offsetX + padded.x * fit.scaleX;
        const vpY = rect.top + fit.offsetY + padded.y * fit.scaleY;
        const vpW = padded.width * fit.scaleX;
        const vpH = padded.height * fit.scaleY;

        // Clamp to element rect
        const finalX = Math.max(rect.left, vpX);
        const finalY = Math.max(rect.top, vpY);
        const finalRight = Math.min(rect.left + rect.width, vpX + vpW);
        const finalBottom = Math.min(rect.top + rect.height, vpY + vpH);

        if (finalRight - finalX < 4 || finalBottom - finalY < 4) continue;

        viewportFaces.push({
          x: Math.round(finalX),
          y: Math.round(finalY),
          width: Math.round(finalRight - finalX),
          height: Math.round(finalBottom - finalY),
          borderRadius: '8px',
          confidence: f.confidence,
          isHeuristic: f.isHeuristic || false
        });
      }

      return viewportFaces;
    }

    /**
     * Scans all page images. Returns viewport-coordinate face boxes.
     */
    async scanPageImages() {
      await this.init();

      const allFaceBoxes = [];
      const elements = Array.from(document.querySelectorAll('img'));

      let totalImagesOnPage = elements.length;
      let skippedTooSmall = 0, skippedNotLoaded = 0, scannedCount = 0, failCount = 0;
      const processed = new Set();

      for (const img of elements) {
        if (img.closest('#privacyshield-root') || processed.has(img)) continue;
        processed.add(img);

        const rect = img.getBoundingClientRect();
        if (rect.width < 48 || rect.height < 48) { skippedTooSmall++; continue; }
        if (!img.complete || img.naturalWidth === 0) { skippedNotLoaded++; continue; }

        scannedCount++;
        try {
          const faces = await this.scanSingleImage(img);
          allFaceBoxes.push(...faces);
        } catch (err) {
          failCount++;
          console.warn('[PrivacyShield] Scan failed for img:', err.message);
        }
      }

      this.telemetryBreakdown = {
        totalImagesOnPage, skippedTooSmall, skippedNotLoaded,
        scannedCount, failedCount: failCount, facesFound: allFaceBoxes.length
      };

      console.log(`[PrivacyShield] Face scan: total=${totalImagesOnPage}, scanned=${scannedCount}, faces=${allFaceBoxes.length}, backend=${this.activeBackend}`);
      return allFaceBoxes;
    }

    getStatus() {
      return {
        status: this.detectorStatus,
        activeBackend: this.activeBackend,
        isModelLoaded: this.isModelLoaded,
        totalDetections: this.detectionCount,
        lastDurationMs: this.lastInferenceTimeMs
      };
    }
  }

  const faceDetectorInstance = new LocalFaceDetector();

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { LocalFaceDetector, faceDetector: faceDetectorInstance };
  } else if (typeof window !== 'undefined') {
    window.LocalFaceDetector = LocalFaceDetector;
    window.faceDetector = faceDetectorInstance;
  }
})();
