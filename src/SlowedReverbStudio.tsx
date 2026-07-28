import React, { useState, useRef, useEffect, useMemo, useCallback } from "react";
import { Play, Pause, Square, Upload, Download, Loader2, Music2, Film } from "lucide-react";

// ---------------------------------------------------------------------------
// Palette tokens — Beige Grey / Stone Grey / Tarpaulin Grey + brass hardware
// ---------------------------------------------------------------------------
const COLOR = {
  beige: "#CDC5B4",
  beigeLight: "#E4DECE",
  stone: "#8F8A7E",
  stoneDark: "#5D5A50",
  tarp: "#3A4240",
  tarpDeep: "#232826",
  brass: "#B08D4F",
  brassLight: "#D1AE73",
  rust: "#8B3A2E",
};

// ---------------------------------------------------------------------------
// Pipeline core — kept the same shape as the original Python pipeline:
// room impulse responses, EQ bands, widening, limiter. Only the engine
// changed (Web Audio instead of numpy/scipy).
// ---------------------------------------------------------------------------
type RoomType = "small_room" | "medium_room" | "large_hall" | "cathedral" | "warehouse";

const ROOM_PARAMS: Record<RoomType, { earlyDelay: number; decayTime: number; label: string }> = {
  small_room: { earlyDelay: 0.3, decayTime: 0.5, label: "Small Room" },
  medium_room: { earlyDelay: 0.5, decayTime: 1.5, label: "Medium Room" },
  large_hall: { earlyDelay: 0.8, decayTime: 2.5, label: "Large Hall" },
  cathedral: { earlyDelay: 1.0, decayTime: 4.0, label: "Cathedral" },
  warehouse: { earlyDelay: 1.2, decayTime: 3.5, label: "Warehouse" },
};

const ROOM_ORDER: RoomType[] = ["small_room", "medium_room", "large_hall", "cathedral", "warehouse"];

// Reference pitch tuning. Standard concert pitch is A4 = 440 Hz; these are
// alternate reference pitches some producers retune to. Retuning is done by
// resampling (the same mechanism as the Speed control), so it's expressed
// as a ratio against 440 Hz and combined with the speed's stretch factor.
const REFERENCE_HZ = 440;
const TUNING_PRESETS = [396, 417, 432, 440, 444, 528, 639, 741, 852, 963];

/**
 * Generates a diffuse, noise-based impulse response per room type.
 * A bare exponential envelope with no noise rings like a comb filter
 * (the "synthetic" artifact from the old pipeline). Multiplying the
 * decay envelope by filtered noise gives a natural, diffuse tail.
 */
function generateImpulseResponse(ctx: BaseAudioContext, roomType: RoomType): AudioBuffer {
  const { earlyDelay, decayTime } = ROOM_PARAMS[roomType];
  const sr = ctx.sampleRate;
  const length = Math.max(1, Math.floor(decayTime * sr));
  const buffer = ctx.createBuffer(2, length, sr);

  for (let ch = 0; ch < 2; ch++) {
    const data = buffer.getChannelData(ch);
    data[0] = 1;

    const numReflections = 4;
    for (let i = 1; i < numReflections; i++) {
      const delaySample = Math.floor((earlyDelay * sr * i) / numReflections);
      if (delaySample < length) data[delaySample] = Math.pow(0.5, i);
    }

    const tailStart = Math.floor(0.1 * sr);
    const tailLength = Math.max(1, length - tailStart);
    let prev = 0;
    for (let i = 0; i < tailLength; i++) {
      const white = Math.random() * 2 - 1;
      const smoothed = white + 0.6 * prev; // light one-pole smoothing
      prev = smoothed;
      const decay = Math.exp((-5 * i) / tailLength);
      const idx = tailStart + i;
      if (idx < length) data[idx] += smoothed * decay * 0.3;
    }

    let max = 0;
    for (let i = 0; i < length; i++) max = Math.max(max, Math.abs(data[i]));
    if (max > 0) for (let i = 0; i < length; i++) data[i] /= max;
  }

  return buffer;
}

type ChainParams = {
  reverbAmount: number; // 0..1
  roomType: RoomType;
  eqOn: boolean;
  wideningOn: boolean;
};

type ChainRefs = {
  bassEQ: BiquadFilterNode;
  trebleEQ: BiquadFilterNode;
  dryGain: GainNode;
  wetGain: GainNode;
  convolver: ConvolverNode;
  output: AudioNode;
};

/**
 * Builds the effects chain. Works for both a live AudioContext and an
 * OfflineAudioContext, so the exact same graph is used for playback and
 * for rendering the downloadable file.
 *
 * source -> EQ -> [dry / convolver-reverb] -> (optional) stereo widener
 *   -> limiter -> master gain
 */
function buildProcessingChain(ctx: BaseAudioContext, input: AudioNode, params: ChainParams): ChainRefs {
  const bassEQ = ctx.createBiquadFilter();
  bassEQ.type = "peaking";
  bassEQ.frequency.value = 100;
  bassEQ.Q.value = 0.9;
  bassEQ.gain.value = params.eqOn ? 6 : 0;

  const trebleEQ = ctx.createBiquadFilter();
  trebleEQ.type = "peaking";
  trebleEQ.frequency.value = 4000;
  trebleEQ.Q.value = 0.9;
  trebleEQ.gain.value = params.eqOn ? 4 : 0;

  input.connect(bassEQ);
  bassEQ.connect(trebleEQ);

  const dryGain = ctx.createGain();
  dryGain.gain.value = 1 - params.reverbAmount;
  const wetGain = ctx.createGain();
  wetGain.gain.value = params.reverbAmount;

  const convolver = ctx.createConvolver();
  convolver.normalize = false;
  convolver.buffer = generateImpulseResponse(ctx, params.roomType);

  trebleEQ.connect(dryGain);
  trebleEQ.connect(convolver);
  convolver.connect(wetGain);

  const reverbSum = ctx.createGain();
  dryGain.connect(reverbSum);
  wetGain.connect(reverbSum);

  let widenOutput: AudioNode = reverbSum;
  if (params.wideningOn) {
    const leftTap = ctx.createGain();
    const rightTap = ctx.createGain();
    reverbSum.connect(leftTap);
    reverbSum.connect(rightTap);
    const delay = ctx.createDelay(0.05);
    delay.delayTime.value = 0.02;
    rightTap.connect(delay);
    const merger = ctx.createChannelMerger(2);
    leftTap.connect(merger, 0, 0);
    delay.connect(merger, 0, 1);
    widenOutput = merger;
  }

  const limiter = ctx.createDynamicsCompressor();
  limiter.threshold.value = -6;
  limiter.knee.value = 0;
  limiter.ratio.value = 20;
  limiter.attack.value = 0.003;
  limiter.release.value = 0.15;
  widenOutput.connect(limiter);

  const masterGain = ctx.createGain();
  masterGain.gain.value = 0.9;
  limiter.connect(masterGain);

  return { bassEQ, trebleEQ, dryGain, wetGain, convolver, output: masterGain };
}

function normalizeBuffer(buffer: AudioBuffer, target = 0.97) {
  let peak = 0;
  for (let ch = 0; ch < buffer.numberOfChannels; ch++) {
    const data = buffer.getChannelData(ch);
    for (let i = 0; i < data.length; i++) peak = Math.max(peak, Math.abs(data[i]));
  }
  if (peak > 0) {
    const gain = target / peak;
    for (let ch = 0; ch < buffer.numberOfChannels; ch++) {
      const data = buffer.getChannelData(ch);
      for (let i = 0; i < data.length; i++) data[i] *= gain;
    }
  }
}

function encodeWav(buffer: AudioBuffer): Blob {
  const numChannels = buffer.numberOfChannels;
  const sampleRate = buffer.sampleRate;
  const numFrames = buffer.length;
  const blockAlign = numChannels * 2;
  const dataSize = numFrames * blockAlign;
  const arrayBuffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(arrayBuffer);

  const writeStr = (offset: number, str: string) => {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  };

  writeStr(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);
  writeStr(36, "data");
  view.setUint32(40, dataSize, true);

  const channels: Float32Array[] = [];
  for (let ch = 0; ch < numChannels; ch++) channels.push(buffer.getChannelData(ch));

  let offset = 44;
  for (let i = 0; i < numFrames; i++) {
    for (let ch = 0; ch < numChannels; ch++) {
      const s = Math.max(-1, Math.min(1, channels[ch][i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      offset += 2;
    }
  }

  return new Blob([arrayBuffer], { type: "audio/wav" });
}

function computePeaks(buffer: AudioBuffer, numBars: number): number[] {
  const data = buffer.getChannelData(0);
  const blockSize = Math.max(1, Math.floor(data.length / numBars));
  const peaks: number[] = [];
  for (let i = 0; i < numBars; i++) {
    const start = i * blockSize;
    let max = 0;
    for (let j = 0; j < blockSize; j++) {
      const v = Math.abs(data[start + j] ?? 0);
      if (v > max) max = v;
    }
    peaks.push(max);
  }
  const maxPeak = Math.max(...peaks, 0.0001);
  return peaks.map((p) => p / maxPeak);
}

function formatTime(t: number): string {
  if (!isFinite(t) || t < 0) t = 0;
  const m = Math.floor(t / 60);
  const s = Math.floor(t % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

const VIDEO_EXTS = [
  ".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv", ".m4v",
  ".mpg", ".mpeg", ".3gp", ".3g2", ".ts", ".mts", ".m2ts", ".ogv",
  ".vob", ".asf", ".divx", ".f4v", ".mxf",
];
const AUDIO_EXTS = [
  ".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".wma", ".aiff",
  ".aif", ".opus", ".oga", ".caf", ".mp2", ".amr", ".ac3", ".weba",
];

/** Same intent as the old load_media(): decide audio vs. video by extension,
 * then MIME type, defaulting to audio if neither is conclusive. */
function detectMediaType(file: File): "audio" | "video" {
  const name = file.name.toLowerCase();
  const dot = name.lastIndexOf(".");
  const ext = dot >= 0 ? name.slice(dot) : "";
  if (VIDEO_EXTS.includes(ext)) return "video";
  if (AUDIO_EXTS.includes(ext)) return "audio";
  if (file.type.startsWith("video/")) return "video";
  if (file.type.startsWith("audio/")) return "audio";
  return "audio";
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export default function SlowedReverbStudio() {
  const [fileName, setFileName] = useState<string | null>(null);
  const [mediaType, setMediaType] = useState<"audio" | "video" | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [audioBuffer, setAudioBuffer] = useState<AudioBuffer | null>(null);
  const [peaks, setPeaks] = useState<number[] | null>(null);
  const [isDecoding, setIsDecoding] = useState(false);
  const [isRendering, setIsRendering] = useState(false);
  const [isRenderingVideo, setIsRenderingVideo] = useState(false);
  const [videoRenderProgress, setVideoRenderProgress] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [level, setLevel] = useState(0);

  const [slowFactor, setSlowFactor] = useState(1.4);
  const [targetHz, setTargetHz] = useState(REFERENCE_HZ);
  const [reverbAmount, setReverbAmount] = useState(0.35);
  const [roomType, setRoomType] = useState<RoomType>("cathedral");
  const [eqOn, setEqOn] = useState(true);
  const [wideningOn, setWideningOn] = useState(true);

  const audioCtxRef = useRef<AudioContext | null>(null);
  const sourceRef = useRef<AudioBufferSourceNode | null>(null);
  const chainRef = useRef<ChainRefs | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const rafRef = useRef<number | null>(null);
  const startCtxTimeRef = useRef(0);
  const pausedAtRef = useRef(0); // position on the *stretched* timeline
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // The Speed control and the Hz tuning control both work by resampling, so
  // their effects combine into one playbackRate. pitchRatio > 1 raises the
  // reference pitch (e.g. 528/440), < 1 lowers it (e.g. 432/440). stretchFactor
  // is the total timeline stretch (used for duration/offset math, like
  // slowFactor alone used to be); effectiveRate is what actually goes into
  // AudioBufferSourceNode/HTMLVideoElement .playbackRate.
  const pitchRatio = targetHz / REFERENCE_HZ;
  const stretchFactor = slowFactor / pitchRatio;
  const effectiveRate = 1 / stretchFactor;

  const duration = audioBuffer ? audioBuffer.duration * stretchFactor : 0;

  const idlePeaks = useMemo(
    () => Array.from({ length: 96 }, (_, i) => 0.22 + 0.16 * Math.sin(i * 0.35) + 0.06 * Math.sin(i * 0.9)),
    []
  );
  const displayPeaks = peaks ?? idlePeaks;

  const ensureContext = useCallback(async () => {
    if (!audioCtxRef.current) {
      const Ctor = window.AudioContext || (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
      if (!Ctor) throw new Error("This browser doesn't support the Web Audio API.");
      audioCtxRef.current = new Ctor();
      console.log("AudioContext created, state:", audioCtxRef.current.state);
    }
    // Resume audio context if suspended (Safari and other browsers require user interaction)
    if (audioCtxRef.current.state === "suspended") {
      console.log("Resuming suspended AudioContext");
      try {
        await audioCtxRef.current.resume();
        console.log("AudioContext resumed successfully");
      } catch (err) {
        console.error("Failed to resume AudioContext:", err);
        throw new Error("Failed to resume audio context - please ensure sound is not muted");
      }
    }
    return audioCtxRef.current;
  }, []);

  const stopSource = useCallback(() => {
    if (sourceRef.current) {
      try {
        sourceRef.current.onended = null;
        sourceRef.current.stop();
      } catch {
        /* already stopped */
      }
      sourceRef.current.disconnect();
      sourceRef.current = null;
    }
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
  }, []);

  const tick = useCallback(() => {
    const ctx = audioCtxRef.current;
    if (!ctx || !audioBuffer) return;
    const elapsed = pausedAtRef.current + (ctx.currentTime - startCtxTimeRef.current);
    const total = audioBuffer.duration * stretchFactor;
    if (elapsed >= total) {
      setCurrentTime(total);
      setIsPlaying(false);
      stopSource();
      pausedAtRef.current = 0;
      return;
    }
    setCurrentTime(elapsed);

    const analyser = analyserRef.current;
    if (analyser) {
      const data = new Uint8Array(analyser.fftSize);
      analyser.getByteTimeDomainData(data);
      let sumSq = 0;
      for (let i = 0; i < data.length; i++) {
        const v = (data[i] - 128) / 128;
        sumSq += v * v;
      }
      setLevel(Math.sqrt(sumSq / data.length));
    }

    rafRef.current = requestAnimationFrame(tick);
  }, [audioBuffer, stretchFactor, stopSource]);

  const startPlaybackFrom = useCallback(
    async (positionOnStretchedTimeline: number) => {
      if (!audioBuffer) return;
      try {
        const ctx = await ensureContext();
        stopSource();

        const source = ctx.createBufferSource();
        source.buffer = audioBuffer;
        source.playbackRate.value = effectiveRate;

        const chain = buildProcessingChain(ctx, source, { reverbAmount: reverbAmount * 0.5, roomType, eqOn, wideningOn });
        chainRef.current = chain;

        const analyser = ctx.createAnalyser();
        analyser.fftSize = 256;
        chain.output.connect(analyser);
        analyser.connect(ctx.destination);
        analyserRef.current = analyser;

        source.onended = () => {
          /* handled in tick loop to keep timeline math consistent */
        };

        const originalOffset = Math.min(positionOnStretchedTimeline / stretchFactor, audioBuffer.duration);
        source.start(0, originalOffset);
        sourceRef.current = source;
        pausedAtRef.current = positionOnStretchedTimeline;
        startCtxTimeRef.current = ctx.currentTime;

        setIsPlaying(true);
        rafRef.current = requestAnimationFrame(tick);
      } catch (err) {
        console.error("Playback failed:", err);
        setIsPlaying(false);
        setLoadError(
          err instanceof Error ? `Playback couldn't start: ${err.message}` : "Playback couldn't start unexpectedly."
        );
      }
    },
    [audioBuffer, effectiveRate, stretchFactor, reverbAmount, roomType, eqOn, wideningOn, ensureContext, stopSource, tick]
  );

  const handlePlayPause = useCallback(() => {
    if (!audioBuffer) return;
    if (isPlaying) {
      const ctx = audioCtxRef.current;
      const elapsed = ctx ? pausedAtRef.current + (ctx.currentTime - startCtxTimeRef.current) : pausedAtRef.current;
      stopSource();
      pausedAtRef.current = Math.min(elapsed, duration);
      setCurrentTime(pausedAtRef.current);
      setIsPlaying(false);
      setLevel(0);
    } else {
      startPlaybackFrom(pausedAtRef.current >= duration ? 0 : pausedAtRef.current);
    }
  }, [audioBuffer, isPlaying, duration, stopSource, startPlaybackFrom]);

  const handleStop = useCallback(() => {
    stopSource();
    pausedAtRef.current = 0;
    setCurrentTime(0);
    setIsPlaying(false);
    setLevel(0);
  }, [stopSource]);

  const handleSeek = useCallback(
    (fraction: number) => {
      if (!audioBuffer) return;
      const target = Math.max(0, Math.min(1, fraction)) * duration;
      if (isPlaying) {
        startPlaybackFrom(target);
      } else {
        pausedAtRef.current = target;
        setCurrentTime(target);
      }
    },
    [audioBuffer, duration, isPlaying, startPlaybackFrom]
  );

  // Live-update parameters that don't require rebuilding the graph.
  useEffect(() => {
    if (chainRef.current) chainRef.current.dryGain.gain.value = 1 - reverbAmount;
    if (chainRef.current) chainRef.current.wetGain.gain.value = reverbAmount;
  }, [reverbAmount]);

  useEffect(() => {
    if (chainRef.current && audioCtxRef.current) {
      chainRef.current.convolver.buffer = generateImpulseResponse(audioCtxRef.current, roomType);
    }
  }, [roomType]);

  useEffect(() => {
    if (chainRef.current) {
      chainRef.current.bassEQ.gain.value = eqOn ? 6 : 0;
      chainRef.current.trebleEQ.gain.value = eqOn ? 4 : 0;
    }
  }, [eqOn]);

  // Structural changes (speed/tuning rewrite the timeline, widening rebuilds
  // the graph) restart playback smoothly from the current position.
  const restartKey = `${slowFactor}|${targetHz}|${wideningOn}`;
  const prevRestartKeyRef = useRef(restartKey);
  
  // Initialize audio context early and ensure Safari compatibility with user gesture
  useEffect(() => {
    const handleInitAudioContext = async () => {
      try {
        // Try to create/resume audio context - this may fail if not called from user gesture
        await ensureContext();
      } catch (err) {
        // This is expected - will be retried on play button click
        console.log("Early AudioContext initialization deferred (requires user interaction)");
      }
    };

    // Add click listener to document for Safari user gesture requirement
    document.addEventListener("click", handleInitAudioContext, { once: true, capture: true });
    
    return () => {
      document.removeEventListener("click", handleInitAudioContext, { capture: true });
    };
  }, [ensureContext]);

  useEffect(() => {
    if (prevRestartKeyRef.current !== restartKey) {
      prevRestartKeyRef.current = restartKey;
      if (isPlaying) {
        const ctx = audioCtxRef.current;
        const elapsed = ctx ? pausedAtRef.current + (ctx.currentTime - startCtxTimeRef.current) : pausedAtRef.current;
        const clamped = Math.min(elapsed, audioBuffer ? audioBuffer.duration * stretchFactor : 0);
        startPlaybackFrom(clamped);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [restartKey]);

  // Revoke the previous preview URL whenever it changes (or on unmount).
  useEffect(() => {
    if (!videoUrl) return;
    return () => URL.revokeObjectURL(videoUrl);
  }, [videoUrl]);

  const handleFile = useCallback(
    async (file: File) => {
      setIsDecoding(true);
      setLoadError(null);
      handleStop();
      setAudioBuffer(null);
      setPeaks(null);

      const type = detectMediaType(file);
      let timeoutId: ReturnType<typeof setTimeout> | undefined;

      try {
        const ctx = await ensureContext();
        const arrayBuffer = await file.arrayBuffer();

        // decodeAudioData pulls the audio track straight out of a video
        // container too (mp4/webm with AAC/Opus audio decode fine in
        // Chromium browsers), so the same call handles both media types.
        // Some unsupported/odd codecs can stall the decoder instead of
        // rejecting, so race it against a timeout rather than waiting forever.
        const decoded = await Promise.race([
          ctx.decodeAudioData(arrayBuffer),
          new Promise<never>((_, reject) => {
            timeoutId = setTimeout(() => reject(new Error("Decoding timed out")), 30000);
          }),
        ]);

        setAudioBuffer(decoded);
        setPeaks(computePeaks(decoded, 96));
        setFileName(file.name);
        setMediaType(type);
        setVideoUrl(type === "video" ? URL.createObjectURL(file) : null);
      } catch (err) {
        console.error(err);
        setFileName(null);
        setMediaType(null);
        setVideoUrl(null);
        setLoadError(
          type === "video"
            ? "This browser couldn't decode audio from that video. The container is fine — it's likely an audio codec it doesn't support (e.g. ProRes/PCM in some .mov files). Try re-exporting the clip with AAC or Opus audio, or convert it with HandBrake/ffmpeg first."
            : "This browser couldn't decode that audio file. Try a more common format like WAV, MP3, FLAC, or AAC."
        );
      } finally {
        if (timeoutId) clearTimeout(timeoutId);
        setIsDecoding(false);
      }
    },
    [ensureContext, handleStop]
  );

  const handleExport = useCallback(async () => {
    if (!audioBuffer) return;
    setIsRendering(true);
    try {
      const sr = audioBuffer.sampleRate;
      const outLength = Math.ceil(audioBuffer.duration * stretchFactor * sr) + Math.ceil(4 * sr);
      const offlineCtx = new OfflineAudioContext(2, outLength, sr);
      const source = offlineCtx.createBufferSource();
      source.buffer = audioBuffer;
      source.playbackRate.value = effectiveRate;
      const chain = buildProcessingChain(offlineCtx, source, { reverbAmount: reverbAmount * 0.5, roomType, eqOn, wideningOn });
      chain.output.connect(offlineCtx.destination);
      source.start();
      const rendered = await offlineCtx.startRendering();
      normalizeBuffer(rendered);
      const blob = encodeWav(rendered);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      const base = (fileName ?? "audio").replace(/\.[^/.]+$/, "");
      a.href = url;
      a.download = `${base}_slowed_reverb.wav`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setTimeout(() => URL.revokeObjectURL(url), 5000);
    } catch (err) {
      console.error(err);
    } finally {
      setIsRendering(false);
    }
  }, [audioBuffer, stretchFactor, effectiveRate, reverbAmount, roomType, eqOn, wideningOn, fileName]);

  /**
   * Video export can't be done offline the way the WAV export is — there's
   * no OfflineAudioContext equivalent for video frames — so this plays the
   * clip through in real time at the slowed rate while MediaRecorder
   * captures the visual frames and the processed audio together. A 15s
   * source at 1.5x slowdown takes ~22.5s to render, since it's a live
   * capture, not a fast offline bounce.
   */
  const handleExportVideo = useCallback(async () => {
    if (!videoUrl || !audioBuffer) return;
    setIsRenderingVideo(true);
    setVideoRenderProgress(0);

    let off: HTMLVideoElement | null = null;
    try {
      const ctx = await ensureContext();

      off = document.createElement("video");
      off.src = videoUrl;
      off.muted = true; // native audio is silenced; processed audio comes from the Web Audio graph
      off.playsInline = true;
      await new Promise<void>((resolve, reject) => {
        off!.onloadedmetadata = () => resolve();
        off!.onerror = () => reject(new Error("Could not load video for export"));
      });
      off.playbackRate = effectiveRate;

      const source = ctx.createMediaElementSource(off);
      const chain = buildProcessingChain(ctx, source, { reverbAmount: reverbAmount * 0.5, roomType, eqOn, wideningOn });
      const streamDest = ctx.createMediaStreamDestination();
      chain.output.connect(streamDest);

      const anyOff = off as unknown as { captureStream?: () => MediaStream; mozCaptureStream?: () => MediaStream };
      const videoStream = anyOff.captureStream ? anyOff.captureStream() : anyOff.mozCaptureStream?.();
      if (!videoStream) throw new Error("This browser can't capture video frames for export.");

      const combined = new MediaStream([...videoStream.getVideoTracks(), ...streamDest.stream.getAudioTracks()]);

      const mimeCandidates = ["video/webm;codecs=vp9,opus", "video/webm;codecs=vp8,opus", "video/webm"];
      const mimeType = mimeCandidates.find((m) => MediaRecorder.isTypeSupported(m)) ?? "";
      const recorder = mimeType ? new MediaRecorder(combined, { mimeType }) : new MediaRecorder(combined);

      const chunks: BlobPart[] = [];
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.push(e.data);
      };
      const stopped = new Promise<void>((resolve) => {
        recorder.onstop = () => resolve();
      });

      const totalDuration = off.duration * stretchFactor;
      const trackProgress = () => {
        if (!off || off.ended) return;
        setVideoRenderProgress(Math.min(1, (off.currentTime * stretchFactor) / Math.max(totalDuration, 0.001)));
        requestAnimationFrame(trackProgress);
      };

      recorder.start();
      await off.play();
      requestAnimationFrame(trackProgress);

      await new Promise<void>((resolve) => {
        off!.onended = () => resolve();
      });
      recorder.stop();
      await stopped;

      const blob = new Blob(chunks, { type: mimeType || "video/webm" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      const base = (fileName ?? "video").replace(/\.[^/.]+$/, "");
      a.href = url;
      a.download = `${base}_slowed_reverb.webm`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setTimeout(() => URL.revokeObjectURL(url), 5000);
    } catch (err) {
      console.error(err);
      setLoadError("Video export failed in this browser. Chrome or Edge work best for this feature.");
    } finally {
      setIsRenderingVideo(false);
      setVideoRenderProgress(0);
    }
  }, [videoUrl, audioBuffer, stretchFactor, effectiveRate, reverbAmount, roomType, eqOn, wideningOn, fileName, ensureContext]);

  useEffect(() => {
    return () => {
      stopSource();
      audioCtxRef.current?.close().catch(() => {});
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const progressFraction = duration > 0 ? currentTime / duration : 0;

  return (
    <div
      style={{ background: COLOR.tarpDeep, minHeight: "100%", fontFamily: "'Inter', ui-sans-serif, system-ui" }}
      className="w-full min-h-screen p-4 sm:p-6 md:p-8"
    >
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
        .font-display { font-family: 'Oswald', ui-sans-serif, system-ui; }
        .font-tag { font-family: 'IBM Plex Mono', ui-monospace, monospace; }
        .slider-track::-webkit-slider-thumb {
          -webkit-appearance: none;
          width: 18px; height: 18px; border-radius: 9999px;
          background: ${COLOR.brass};
          border: 2px solid ${COLOR.tarpDeep};
          cursor: pointer;
          margin-top: -1px;
        }
        .slider-track::-moz-range-thumb {
          width: 18px; height: 18px; border-radius: 9999px;
          background: ${COLOR.brass};
          border: 2px solid ${COLOR.tarpDeep};
          cursor: pointer;
        }
        .slider-track { -webkit-appearance: none; appearance: none; height: 4px; border-radius: 4px; }
      `}</style>

      <div className="max-w-5xl mx-auto">
        {/* Header */}
        <div className="flex items-start justify-between mb-6 gap-4 flex-wrap">
          <div className="flex items-center gap-3">
            <div className="flex gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full" style={{ background: COLOR.brass }} />
              <span className="w-2.5 h-2.5 rounded-full" style={{ background: COLOR.stone }} />
            </div>
            <div>
              <h1
                className="font-display uppercase tracking-wide text-xl sm:text-2xl"
                style={{ color: COLOR.beigeLight, letterSpacing: "0.04em" }}
              >
                Slowed &amp; Reverbed
              </h1>
              <p className="font-tag text-[11px] uppercase tracking-widest" style={{ color: COLOR.stone }}>
                tape-speed audio studio
              </p>
            </div>
          </div>
          <Music2 size={22} style={{ color: COLOR.stone, flexShrink: 0 }} />
        </div>

        <div
          className="grid grid-cols-1 lg:grid-cols-[1.35fr_1fr] gap-5"
          style={{
            display: "grid",
            gridTemplateColumns: "1.35fr 1fr",
          }}
        >
          {/* Hero: drape visualization */}
          <div
            className="rounded-2xl p-4 sm:p-5 relative overflow-hidden"
            style={{ background: COLOR.tarp, border: `1px solid ${COLOR.stoneDark}` }}
          >
            <DrapeCanvas
              peaks={displayPeaks}
              slowFactor={slowFactor}
              progressFraction={audioBuffer ? progressFraction : 0}
              hasAudio={!!audioBuffer}
              isPlaying={isPlaying}
            />

            <div className="mt-4 flex items-center gap-4">
              <button
                onClick={handlePlayPause}
                disabled={!audioBuffer}
                className="flex items-center justify-center rounded-full w-12 h-12 shrink-0 transition disabled:opacity-40"
                style={{ background: COLOR.brass, color: COLOR.tarpDeep }}
                aria-label={isPlaying ? "Pause" : "Play"}
              >
                {isPlaying ? <Pause size={20} /> : <Play size={20} />}
              </button>
              <button
                onClick={handleStop}
                disabled={!audioBuffer}
                className="flex items-center justify-center rounded-full w-10 h-10 shrink-0 transition disabled:opacity-40"
                style={{ background: "transparent", border: `1px solid ${COLOR.stoneDark}`, color: COLOR.beige }}
                aria-label="Stop"
              >
                <Square size={16} />
              </button>

              <div className="flex-1">
                <div
                  className="h-2 rounded-full cursor-pointer relative"
                  style={{ background: COLOR.tarpDeep }}
                  onClick={(e) => {
                    const rect = e.currentTarget.getBoundingClientRect();
                    handleSeek((e.clientX - rect.left) / rect.width);
                  }}
                >
                  <div
                    className="h-2 rounded-full"
                    style={{ width: `${progressFraction * 100}%`, background: COLOR.brass }}
                  />
                </div>
                <div className="flex justify-between mt-1.5 font-tag text-[11px]" style={{ color: COLOR.stone }}>
                  <span>{formatTime(currentTime)}</span>
                  <span>{formatTime(duration)}</span>
                </div>
              </div>

              <div className="w-16 shrink-0">
                <div className="h-1.5 rounded-full overflow-hidden" style={{ background: COLOR.tarpDeep }}>
                  <div
                    className="h-full transition-[width] duration-75"
                    style={{
                      width: `${Math.min(100, level * 220)}%`,
                      background: level > 0.35 ? COLOR.rust : COLOR.brassLight,
                    }}
                  />
                </div>
                <div className="font-tag text-[10px] mt-1 text-center" style={{ color: COLOR.stone }}>
                  OUT
                </div>
              </div>
            </div>
          </div>

          {/* Controls panel */}
          <div
            className="rounded-2xl p-4 sm:p-5 flex flex-col gap-5"
            style={{ background: COLOR.beige, border: `2px dashed ${COLOR.stoneDark}` }}
          >
            {/* Upload */}
            <div>
              <input
                ref={fileInputRef}
                type="file"
                accept="audio/*,video/*,.mov,.mkv,.avi,.wmv,.flv,.m4v,.3gp,.3g2,.ts,.mts,.m2ts,.ogv,.vob,.asf,.divx,.f4v,.mxf,.opus,.oga,.caf,.amr,.ac3,.weba"
                className="hidden"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) handleFile(f);
                  e.target.value = ""; // allow re-selecting the same file
                }}
              />
              <button
                onClick={() => {
                  ensureContext(); // tie AudioContext creation/resume to this click
                  fileInputRef.current?.click();
                }}
                className="w-full flex items-center justify-center gap-2 rounded-lg py-2.5 font-display uppercase text-sm tracking-wide transition"
                style={{ background: COLOR.tarp, color: COLOR.beigeLight }}
              >
                {isDecoding ? <Loader2 size={16} className="animate-spin" /> : <Upload size={16} />}
                {isDecoding ? "Loading…" : "Load audio or recording"}
              </button>
              <p className="font-tag text-[11px] mt-1.5" style={{ color: COLOR.stoneDark, maxWidth: "150px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", display: "block" }}>
                {fileName ? `${fileName}${mediaType === "video" ? " · screen recording" : ""}` : "No file loaded"}
              </p>
              {loadError && (
                <p className="font-tag text-[11px] mt-1 leading-snug" style={{ color: COLOR.rust }}>
                  {loadError}
                </p>
              )}
            </div>

            <Divider />

            {/* Speed */}
            <div>
              <RowLabel label="Speed" value={`${slowFactor.toFixed(2)}x`} sub={`${(-12 * Math.log2(slowFactor)).toFixed(1)} st`} />
              <input
                type="range"
                min={0.5}
                max={2.5}
                step={0.01}
                value={slowFactor}
                onChange={(e) => setSlowFactor(parseFloat(e.target.value))}
                className="slider-track w-full mt-2"
                style={{ background: COLOR.stone }}
              />
              <div className="flex justify-between font-tag text-[10px] mt-1" style={{ color: COLOR.stoneDark }}>
                <span>faster / higher</span>
                <span>slower / deeper</span>
              </div>
            </div>

            {/* Hz tuning */}
            <div>
              <RowLabel
                label="Tuning"
                value={`${targetHz % 1 === 0 ? targetHz : targetHz.toFixed(1)} Hz`}
                sub={targetHz === REFERENCE_HZ ? "standard" : `${(12 * Math.log2(pitchRatio)).toFixed(1)} st`}
              />
              <div className="flex flex-wrap gap-1.5 mt-2">
                {TUNING_PRESETS.map((hz) => (
                  <button
                    key={hz}
                    onClick={() => setTargetHz(hz)}
                    className="px-2.5 py-1 rounded-md font-tag text-[11px] tracking-wide transition"
                    style={
                      targetHz === hz
                        ? { background: COLOR.brass, color: COLOR.tarpDeep }
                        : { background: COLOR.beigeLight, color: COLOR.stoneDark, border: `1px solid ${COLOR.stone}` }
                    }
                  >
                    {hz}
                    {hz === REFERENCE_HZ ? " std" : ""}
                  </button>
                ))}
              </div>
              <div className="flex items-center gap-2 mt-2">
                <input
                  type="number"
                  min={200}
                  max={700}
                  step={0.1}
                  value={targetHz}
                  onChange={(e) => {
                    const v = parseFloat(e.target.value);
                    if (!isNaN(v)) setTargetHz(Math.min(700, Math.max(200, v)));
                  }}
                  className="w-20 rounded-md px-2 py-1 font-tag text-xs"
                  style={{ background: COLOR.beigeLight, border: `1px solid ${COLOR.stone}`, color: COLOR.tarp }}
                />
                <span className="font-tag text-[10px]" style={{ color: COLOR.stoneDark }}>
                  custom Hz (A4 = 440 is standard concert pitch)
                </span>
              </div>
            </div>

            {/* Reverb amount */}
            <div>
              <RowLabel label="Reverb" value={`${Math.round(reverbAmount * 100)}%`} />
              <input
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={reverbAmount}
                onChange={(e) => setReverbAmount(parseFloat(e.target.value))}
                className="slider-track w-full mt-2"
                style={{ background: COLOR.stone }}
              />
            </div>

            {/* Room type tabs */}
            <div>
              <RowLabel label="Room" value={ROOM_PARAMS[roomType].label} />
              <div className="flex flex-wrap gap-1.5 mt-2">
                {ROOM_ORDER.map((rt) => (
                  <button
                    key={rt}
                    onClick={() => setRoomType(rt)}
                    className="px-2.5 py-1 rounded-md font-tag text-[11px] uppercase tracking-wide transition"
                    style={
                      roomType === rt
                        ? { background: COLOR.brass, color: COLOR.tarpDeep }
                        : { background: COLOR.beigeLight, color: COLOR.stoneDark, border: `1px solid ${COLOR.stone}` }
                    }
                  >
                    {ROOM_PARAMS[rt].label}
                  </button>
                ))}
              </div>
            </div>

            <Divider />

            {/* Toggles */}
            <div className="flex flex-col gap-3">
              <ToggleRow label="EQ boost" sub="bass + air" checked={eqOn} onChange={setEqOn} />
              <ToggleRow label="Stereo widen" sub="haas delay" checked={wideningOn} onChange={setWideningOn} />
            </div>

            <div className="flex-1" />

            <button
              onClick={handleExport}
              disabled={!audioBuffer || isRendering}
              className="w-full flex items-center justify-center gap-2 rounded-lg py-2.5 font-display uppercase text-sm tracking-wide transition disabled:opacity-40"
              style={{ background: COLOR.rust, color: COLOR.beigeLight }}
            >
              {isRendering ? <Loader2 size={16} className="animate-spin" /> : <Download size={16} />}
              {isRendering ? "Rendering…" : "Render & download audio"}
            </button>

            {mediaType === "video" && (
              <div>
                <button
                  onClick={handleExportVideo}
                  disabled={!audioBuffer || isRenderingVideo}
                  className="w-full flex items-center justify-center gap-2 rounded-lg py-2.5 font-display uppercase text-sm tracking-wide transition disabled:opacity-40"
                  style={{ background: COLOR.tarp, color: COLOR.beigeLight }}
                >
                  {isRenderingVideo ? <Loader2 size={16} className="animate-spin" /> : <Film size={16} />}
                  {isRenderingVideo ? `Capturing video… ${Math.round(videoRenderProgress * 100)}%` : "Download slowed video"}
                </button>
                <p className="font-tag text-[10px] mt-1.5 leading-snug text-center" style={{ color: COLOR.stoneDark }}>
                  Captures playback live, so it takes as long as the slowed clip. Works best in Chrome or Edge.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Small presentational pieces
// ---------------------------------------------------------------------------
function Divider() {
  return <div className="h-px" style={{ background: COLOR.stone, opacity: 0.5 }} />;
}

function RowLabel({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="flex items-baseline justify-between">
      <span className="font-display uppercase text-xs tracking-wider" style={{ color: COLOR.tarp }}>
        {label}
      </span>
      <span className="font-tag text-sm" style={{ color: COLOR.stoneDark }}>
        {value}
        {sub ? <span className="ml-1.5 opacity-70">({sub})</span> : null}
      </span>
    </div>
  );
}

function ToggleRow({
  label,
  sub,
  checked,
  onChange,
}: {
  label: string;
  sub: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <div className="flex items-center justify-between">
      <div className="flex-1">
        <div className="font-display uppercase text-xs tracking-wider" style={{ color: COLOR.tarp }}>
          {label}
        </div>
        <div className="font-tag text-[10px]" style={{ color: COLOR.stoneDark }}>
          {sub}
        </div>
      </div>
      <button
        onClick={() => onChange(!checked)}
        className="relative w-12 h-6 rounded-full transition shrink-0 border-0 p-0"
        style={{ background: checked ? COLOR.brass : COLOR.stone }}
        aria-pressed={checked}
        aria-label={label}
      >
        <span
          className="absolute w-5 h-5 rounded-full transition-transform"
          style={{
            background: "white",
            transform: checked ? "translateX(23px) translateY(-50%)" : "translateX(2px) translateY(-50%)",
            top: "50%",
            left: 0,
          }}
        />
      </button>
    </div>
  );
}

function DrapeCanvas({
  peaks,
  slowFactor,
  progressFraction,
  hasAudio,
  isPlaying,
}: {
  peaks: number[];
  slowFactor: number;
  progressFraction: number;
  hasAudio: boolean;
  isPlaying: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const midY = H * 0.56;
    const sagMax = 46;
    const sagStrength = Math.max(0, slowFactor - 1);

    const n = peaks.length;
    const barW = W / n;
    const maxBarH = H * 0.62;

    // Drape curve + grommets
    ctx.beginPath();
    for (let i = 0; i <= 40; i++) {
      const f = i / 40;
      const x = f * W;
      const sag = Math.sin(f * Math.PI) * sagMax * sagStrength;
      const y = midY + sag;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = COLOR.stoneDark;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([2, 4]);
    ctx.stroke();
    ctx.setLineDash([]);

    const grommetCount = 7;
    for (let g = 0; g < grommetCount; g++) {
      const f = g / (grommetCount - 1);
      const x = f * W;
      const sag = Math.sin(f * Math.PI) * sagMax * sagStrength;
      const y = midY + sag;
      ctx.beginPath();
      ctx.arc(x, y, 4.5, 0, Math.PI * 2);
      ctx.fillStyle = COLOR.brass;
      ctx.fill();
      ctx.beginPath();
      ctx.arc(x, y, 2, 0, Math.PI * 2);
      ctx.fillStyle = COLOR.tarpDeep;
      ctx.fill();
    }

    // Waveform bars, centered on the sagging drape line
    for (let i = 0; i < n; i++) {
      const f = i / (n - 1);
      const x = i * barW;
      const sag = Math.sin(f * Math.PI) * sagMax * sagStrength;
      const y = midY + sag;
      const h = Math.max(2, peaks[i] * maxBarH);

      const depth = Math.min(1, sagStrength / 1.5);
      const r1 = parseInt(COLOR.brassLight.slice(1, 3), 16);
      const g1 = parseInt(COLOR.brassLight.slice(3, 5), 16);
      const b1 = parseInt(COLOR.brassLight.slice(5, 7), 16);
      const r2 = parseInt(COLOR.beige.slice(1, 3), 16);
      const g2 = parseInt(COLOR.beige.slice(3, 5), 16);
      const b2 = parseInt(COLOR.beige.slice(5, 7), 16);
      const r = Math.round(r1 + (r2 - r1) * (1 - depth) * 0.4 + r2 * 0);
      const g = Math.round(g1 + (g2 - g1) * (1 - depth) * 0.4);
      const b = Math.round(b1 + (b2 - b1) * (1 - depth) * 0.4);

      ctx.fillStyle = `rgba(${r},${g},${b},${hasAudio ? 0.9 : 0.45})`;
      ctx.fillRect(x + barW * 0.15, y - h / 2, Math.max(1, barW * 0.7), h);
    }

    // Playhead
    if (hasAudio) {
      const x = progressFraction * W;
      ctx.beginPath();
      ctx.moveTo(x, H * 0.08);
      ctx.lineTo(x, H * 0.98);
      ctx.strokeStyle = isPlaying ? COLOR.brassLight : COLOR.brass;
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }, [peaks, slowFactor, progressFraction, hasAudio, isPlaying]);

  return (
    <canvas
      ref={canvasRef}
      width={900}
      height={230}
      className="w-full h-[220px] sm:h-[240px] rounded-lg"
      style={{ background: COLOR.tarpDeep }}
    />
  );
}