import { useEffect, useRef, useState } from "react";
import {
  loadAndCompile,
  loadLiteRt,
  Tensor,
} from "@litertjs/core";
import {
  FaceDetector,
  FilesetResolver,
} from "@mediapipe/tasks-vision";
import "./App.css";

const EMOTIONS = [
  "Anger",
  "Contempt",
  "Disgust",
  "Fear",
  "Happy",
  "Neutral",
  "Sad",
  "Surprised",
];

const MODEL_PATH =
  "/models/emotion_model_v1_web.tflite";

const FACE_MODEL_PATH =
  "/models/blaze_face_short_range.tflite";

const LITERT_WASM_PATH =
  "/litert-wasm/";

const MEDIAPIPE_WASM_PATH =
  "/mediapipe/wasm/";

let liteRtPromise = null;
let emotionModelPromise = null;
let faceDetectorPromise = null;

function getLiteRt() {
  if (!liteRtPromise) {
    liteRtPromise = loadLiteRt(
      LITERT_WASM_PATH,
      {
        jspi: true,
      }
    );
  }

  return liteRtPromise;
}

function getEmotionModel() {
  if (!emotionModelPromise) {
    emotionModelPromise = getLiteRt().then(() =>
      loadAndCompile(
        MODEL_PATH,
        {
          accelerator: "wasm",
        }
      )
    );
  }

  return emotionModelPromise;
}

function getFaceDetector() {
  if (!faceDetectorPromise) {
    faceDetectorPromise =
      FilesetResolver.forVisionTasks(
        MEDIAPIPE_WASM_PATH
      ).then((vision) =>
        FaceDetector.createFromOptions(
          vision,
          {
            baseOptions: {
              modelAssetPath:
                FACE_MODEL_PATH,
            },
            runningMode: "VIDEO",
            minDetectionConfidence: 0.5,
          }
        )
      );
  }

  return faceDetectorPromise;
}

function clamp(value, min, max) {
  return Math.max(
    min,
    Math.min(value, max)
  );
}

function preprocessFace(
  video,
  boundingBox,
  canvas,
  ctx
) {
  const width = video.videoWidth;
  const height = video.videoHeight;

  if (!width || !height) {
    return null;
  }

  const x = clamp(
    Math.round(boundingBox.originX),
    0,
    width - 1
  );

  const y = clamp(
    Math.round(boundingBox.originY),
    0,
    height - 1
  );

  const faceWidth = clamp(
    Math.round(boundingBox.width),
    1,
    width - x
  );

  const faceHeight = clamp(
    Math.round(boundingBox.height),
    1,
    height - y
  );

  ctx.clearRect(
    0,
    0,
    48,
    48
  );

  ctx.drawImage(
    video,
    x,
    y,
    faceWidth,
    faceHeight,
    0,
    0,
    48,
    48
  );

  const imageData =
    ctx.getImageData(
      0,
      0,
      48,
      48
    );

  const inputData =
    new Float32Array(
      48 * 48
    );

  for (
    let i = 0;
    i < 48 * 48;
    i++
  ) {
    const index = i * 4;

    const r =
      imageData.data[index];

    const g =
      imageData.data[index + 1];

    const b =
      imageData.data[index + 2];

    const grayscale =
      0.299 * r +
      0.587 * g +
      0.114 * b;

    inputData[i] =
      grayscale / 255;
  }

  return inputData;
}

function drawDetection(
  ctx,
  detection,
  emotion,
  confidence,
  mirrored,
  videoWidth,
  videoHeight
) {
  if (!detection?.boundingBox) {
    return;
  }

  const box =
    detection.boundingBox;

  let x = box.originX;

  if (mirrored) {
    x =
      videoWidth -
      (box.originX + box.width);
  }

  const y =
    box.originY;

  const width =
    box.width;

  const height =
    box.height;

  x = clamp(
    x,
    0,
    videoWidth - 1
  );

  ctx.strokeStyle =
    "#00ff66";

  ctx.lineWidth = 3;

  ctx.strokeRect(
    x,
    y,
    width,
    height
  );

  const label =
    `${emotion} ${confidence.toFixed(1)}%`;

  ctx.font =
    "700 18px Arial";

  const paddingX = 10;
  const labelHeight = 30;

  const textWidth =
    ctx.measureText(
      label
    ).width;

  const labelWidth =
    textWidth +
    paddingX * 2;

  let labelX = x;

  if (
    labelX + labelWidth >
    videoWidth
  ) {
    labelX =
      videoWidth -
      labelWidth -
      6;
  }

  labelX =
    Math.max(6, labelX);

  let labelY =
    y -
    labelHeight -
    8;

  if (labelY < 6) {
    labelY = y + 6;
  }

  ctx.fillStyle =
    "rgba(0, 0, 0, 0.78)";

  ctx.fillRect(
    labelX,
    labelY,
    labelWidth,
    labelHeight
  );

  ctx.fillStyle =
    "#ffffff";

  ctx.fillText(
    label,
    labelX + paddingX,
    labelY + 21
  );
}

export default function App() {
  const videoRef =
    useRef(null);

  const canvasRef =
    useRef(null);

  const faceCanvasRef =
    useRef(null);

  const streamRef =
    useRef(null);

  const animationFrameRef =
    useRef(null);

  const cameraRequestIdRef =
    useRef(0);

  const inferenceBusyRef =
    useRef(false);

  const emotionRef =
    useRef("Waiting");

  const confidenceRef =
    useRef(0);

  const cameraFacingRef =
    useRef("user");

  const lastDetectionRef =
    useRef(null);

  const lastDetectionTimeRef =
    useRef(0);

  const lastInferenceTimeRef =
    useRef(0);

  const [cameraFacing, setCameraFacing] =
    useState("user");

  const [status, setStatus] =
    useState("Loading AI...");

  const [error, setError] =
    useState("");

  const [faces, setFaces] =
    useState(0);

  const [emotion, setEmotion] =
    useState("Waiting");

  const [confidence, setConfidence] =
    useState(0);

  useEffect(() => {
    cameraFacingRef.current =
      cameraFacing;
  }, [cameraFacing]);

  useEffect(() => {
    const canvas =
      document.createElement(
        "canvas"
      );

    canvas.width = 48;
    canvas.height = 48;

    faceCanvasRef.current =
      canvas;
  }, []);

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current
        .getTracks()
        .forEach((track) =>
          track.stop()
        );

      streamRef.current =
        null;
    }

    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.srcObject =
        null;
    }
  };

  const drawFrame = (
    detection
  ) => {
    const video =
      videoRef.current;

    const canvas =
      canvasRef.current;

    if (
      !video ||
      !canvas ||
      !video.videoWidth ||
      !video.videoHeight
    ) {
      return;
    }

    const ctx =
      canvas.getContext("2d");

    if (!ctx) {
      return;
    }

    if (
      canvas.width !==
        video.videoWidth ||
      canvas.height !==
        video.videoHeight
    ) {
      canvas.width =
        video.videoWidth;

      canvas.height =
        video.videoHeight;
    }

    ctx.clearRect(
      0,
      0,
      canvas.width,
      canvas.height
    );

    if (!detection) {
      return;
    }

    drawDetection(
      ctx,
      detection,
      emotionRef.current,
      confidenceRef.current,
      cameraFacingRef.current ===
        "user",
      video.videoWidth,
      video.videoHeight
    );
  };

  const predictEmotion = async (
    detection,
    model
  ) => {
    if (
      inferenceBusyRef.current ||
      !detection
    ) {
      return;
    }

    const faceCanvas =
      faceCanvasRef.current;

    const video =
      videoRef.current;

    if (!faceCanvas || !video) {
      return;
    }

    const ctx =
      faceCanvas.getContext(
        "2d",
        {
          willReadFrequently:
            true,
        }
      );

    if (!ctx) {
      return;
    }

    inferenceBusyRef.current =
      true;

    let inputTensor = null;
    let results = null;

    try {
      const inputData =
        preprocessFace(
          video,
          detection.boundingBox,
          faceCanvas,
          ctx
        );

      if (!inputData) {
        return;
      }

      inputTensor =
        new Tensor(
          inputData,
          [1, 48, 48, 1]
        );

      results =
        await model.run(
          inputTensor
        );

      const outputTensor =
        results[0];

      const output =
        await outputTensor.data();

      let bestIndex = 0;

      for (
        let i = 1;
        i < output.length;
        i++
      ) {
        if (
          output[i] >
          output[bestIndex]
        ) {
          bestIndex = i;
        }
      }

      const predictedEmotion =
        EMOTIONS[bestIndex];

      const predictedConfidence =
        output[bestIndex] * 100;

      emotionRef.current =
        predictedEmotion;

      confidenceRef.current =
        predictedConfidence;

      setEmotion(
        predictedEmotion
      );

      setConfidence(
        predictedConfidence
      );
    } catch (err) {
      console.error(
        "Emotion inference error:",
        err
      );
    } finally {
      if (
        inputTensor &&
        typeof inputTensor.delete ===
          "function"
      ) {
        inputTensor.delete();
      }

      if (
        results &&
        typeof results.delete ===
          "function"
      ) {
        results.delete();
      }
    }

    inferenceBusyRef.current =
      false;
  };

  const processFrame = async (
    model,
    detector
  ) => {
    const video =
      videoRef.current;

    if (
      !video ||
      video.readyState < 2
    ) {
      animationFrameRef.current =
        requestAnimationFrame(
          () =>
            processFrame(
              model,
              detector
            )
        );

      return;
    }

    const now =
      performance.now();

    if (
      now -
        lastDetectionTimeRef.current >=
      60
    ) {
      lastDetectionTimeRef.current =
        now;

      try {
        const result =
          detector.detectForVideo(
            video,
            now
          );

        const detections =
          result?.detections || [];

        setFaces(
          detections.length
        );

        if (detections.length) {
          const detection =
            detections[0];

          lastDetectionRef.current =
            detection;

          drawFrame(
            detection
          );

          if (
            now -
              lastInferenceTimeRef.current >=
            250
          ) {
            lastInferenceTimeRef.current =
              now;

            predictEmotion(
              detection,
              model
            );
          }
        } else {
          lastDetectionRef.current =
            null;

          drawFrame(null);
        }
      } catch (err) {
        console.error(
          "Face detection error:",
          err
        );
      }
    }

    if (
      lastDetectionRef.current
    ) {
      drawFrame(
        lastDetectionRef.current
      );
    }

    animationFrameRef.current =
      requestAnimationFrame(
        () =>
          processFrame(
            model,
            detector
          )
      );
  };

  const startCamera = async (
    facingMode
  ) => {
    const requestId =
      ++cameraRequestIdRef.current;

    stopCamera();

    setError("");

    setStatus(
      "Requesting camera..."
    );

    try {
      const stream =
        await navigator.mediaDevices.getUserMedia(
          {
            video: {
              facingMode,
              width: {
                ideal: 640,
              },
              height: {
                ideal: 480,
              },
            },
            audio: false,
          }
        );

      if (
        requestId !==
        cameraRequestIdRef.current
      ) {
        stream
          .getTracks()
          .forEach((track) =>
            track.stop()
          );

        return;
      }

      streamRef.current =
        stream;

      const video =
        videoRef.current;

      if (!video) {
        return;
      }

      video.srcObject =
        stream;

      await video.play();

      setStatus(
        "Camera and AI ready."
      );
    } catch (err) {
      console.error(
        "Camera error:",
        err
      );

      setStatus(
        "Camera unavailable."
      );

      if (
        err?.name ===
        "NotAllowedError"
      ) {
        setError(
          "Camera permission was denied. Please allow camera access and restart the camera."
        );
      } else if (
        err?.name ===
        "NotFoundError"
      ) {
        setError(
          "No camera was found on this device."
        );
      } else {
        setError(
          "Unable to start the camera. Please check your camera settings."
        );
      }
    }
  };

  const switchCamera = async () => {
    const nextFacing =
      cameraFacingRef.current ===
      "user"
        ? "environment"
        : "user";

    cameraFacingRef.current =
      nextFacing;

    setCameraFacing(
      nextFacing
    );

    await startCamera(
      nextFacing
    );
  };

  const restartCamera = async () => {
    await startCamera(
      cameraFacingRef.current
    );
  };

  useEffect(() => {
    let cancelled = false;

    const initialize = async () => {
      try {
        setStatus(
          "Loading AI model..."
        );

        const [
          model,
          detector,
        ] = await Promise.all([
          getEmotionModel(),
          getFaceDetector(),
        ]);

        if (cancelled) {
          return;
        }

        setStatus(
          "AI loaded. Starting camera..."
        );

        await startCamera(
          cameraFacingRef.current
        );

        if (cancelled) {
          return;
        }

        animationFrameRef.current =
          requestAnimationFrame(
            () =>
              processFrame(
                model,
                detector
              )
          );
      } catch (err) {
        console.error(
          "Initialization error:",
          err
        );

        if (!cancelled) {
          setStatus(
            "Initialization failed."
          );

          setError(
            err?.message ||
              "Unable to load the AI model or face detector."
          );
        }
      }
    };

    initialize();

    return () => {
      cancelled = true;

      cameraRequestIdRef.current++;

      if (
        animationFrameRef.current
      ) {
        cancelAnimationFrame(
          animationFrameRef.current
        );

        animationFrameRef.current =
          null;
      }

      stopCamera();
    };
  }, []);

  return (
    <div className="app">
      <header className="header">
        <h1>
          Human Emotion Detection
        </h1>

        <p>
          Real-time facial emotion
          recognition
        </p>
      </header>

      <main className="main">
        <div className="camera-wrapper">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className={
              cameraFacing === "user"
                ? "mirror"
                : ""
            }
          />

          <canvas
            ref={canvasRef}
            className="overlay"
          />
        </div>

        <div className="stats">
          <div className="stat">
            <span>Faces</span>
            <strong>
              {faces}
            </strong>
          </div>

          <div className="stat">
            <span>Emotion</span>
            <strong>
              {emotion}
            </strong>
          </div>

          <div className="stat">
            <span>Confidence</span>
            <strong>
              {confidence.toFixed(1)}%
            </strong>
          </div>
        </div>

        <div className="status">
          <span>{status}</span>
        </div>

        {error && (
          <div className="error">
            {error}
          </div>
        )}

        <div className="controls">
          <button
            type="button"
            onClick={
              switchCamera
            }
          >
            Switch Camera
          </button>

          <button
            type="button"
            onClick={
              restartCamera
            }
          >
            Restart Camera
          </button>
        </div>
      </main>

      <footer className="footer">
        <span>
          Developed &amp; created with
        </span>

        <span className="footer-heart">
          ♥
        </span>

        <span>
          by Utsadeep Kundu
        </span>
      </footer>
    </div>
  );
}