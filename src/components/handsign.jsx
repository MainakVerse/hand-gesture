import React, { useRef, useState, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";
import * as handpose from "@tensorflow-models/handpose";
import Webcam from "react-webcam";
import * as fp from "fingerpose";
import "../index.css";
import { drawHand } from "../utilities";

// Utility functions for gesture creation
const createGesture = (name) => new fp.GestureDescription(name);

const setFullCurl = (gesture, fingers) => {
  fingers.forEach(finger => {
    gesture.addCurl(finger, fp.FingerCurl.FullCurl, 1.0);
  });
};

const setNoCurl = (gesture, fingers) => {
  fingers.forEach(finger => {
    gesture.addCurl(finger, fp.FingerCurl.NoCurl, 1.0);
  });
};

const setDirection = (gesture, finger, direction, weight = 1.0) => {
  gesture.addDirection(finger, direction, weight);
};

// Define All Fingers array for convenience
const AllFingers = [
  fp.Finger.Thumb,
  fp.Finger.Index,
  fp.Finger.Middle,
  fp.Finger.Ring,
  fp.Finger.Pinky
];

// Gestures definitions
const victoryGesture = createGesture('victory');
setNoCurl(victoryGesture, [fp.Finger.Index, fp.Finger.Middle]);
setFullCurl(victoryGesture, [fp.Finger.Ring, fp.Finger.Pinky]);
setDirection(victoryGesture, fp.Finger.Index, fp.FingerDirection.VerticalUp);
setDirection(victoryGesture, fp.Finger.Middle, fp.FingerDirection.VerticalUp);
victoryGesture.addCurl(fp.Finger.Thumb, fp.FingerCurl.HalfCurl);

const thumbsUpGesture = createGesture('thumbs_up');
setFullCurl(thumbsUpGesture, [fp.Finger.Index, fp.Finger.Middle, fp.Finger.Ring, fp.Finger.Pinky]);
thumbsUpGesture.addCurl(fp.Finger.Thumb, fp.FingerCurl.NoCurl, 1.0);
setDirection(thumbsUpGesture, fp.Finger.Thumb, fp.FingerDirection.VerticalUp, 1.0);

const openPalmGesture = createGesture('open_palm');
setNoCurl(openPalmGesture, AllFingers);
AllFingers.forEach(finger => {
  setDirection(openPalmGesture, finger, fp.FingerDirection.VerticalUp);
});

const closedFistGesture = createGesture('closed_fist');
setFullCurl(closedFistGesture, AllFingers);

const pointUpGesture = createGesture('point_up');
setFullCurl(pointUpGesture, [fp.Finger.Middle, fp.Finger.Ring, fp.Finger.Pinky]);
pointUpGesture.addCurl(fp.Finger.Index, fp.FingerCurl.NoCurl, 1.0);
pointUpGesture.addCurl(fp.Finger.Thumb, fp.FingerCurl.HalfCurl, 0.8);
setDirection(pointUpGesture, fp.Finger.Index, fp.FingerDirection.VerticalUp);

const okSignGesture = createGesture('ok_sign');
okSignGesture.addCurl(fp.Finger.Index, fp.FingerCurl.HalfCurl, 1.0);
okSignGesture.addCurl(fp.Finger.Thumb, fp.FingerCurl.HalfCurl, 1.0);
setNoCurl(okSignGesture, [fp.Finger.Middle, fp.Finger.Ring, fp.Finger.Pinky]);
setDirection(okSignGesture, fp.Finger.Middle, fp.FingerDirection.VerticalUp);
setDirection(okSignGesture, fp.Finger.Ring, fp.FingerDirection.VerticalUp);
setDirection(okSignGesture, fp.Finger.Pinky, fp.FingerDirection.VerticalUp);

const rockOnGesture = createGesture('rock_on');
setNoCurl(rockOnGesture, [fp.Finger.Index, fp.Finger.Pinky]);
setFullCurl(rockOnGesture, [fp.Finger.Middle, fp.Finger.Ring]);
setDirection(rockOnGesture, fp.Finger.Index, fp.FingerDirection.VerticalUp);
setDirection(rockOnGesture, fp.Finger.Pinky, fp.FingerDirection.VerticalUp);
rockOnGesture.addCurl(fp.Finger.Thumb, fp.FingerCurl.HalfCurl);

const callMeGesture = createGesture('call_me');
setNoCurl(callMeGesture, [fp.Finger.Thumb, fp.Finger.Pinky]);
setFullCurl(callMeGesture, [fp.Finger.Index, fp.Finger.Middle, fp.Finger.Ring]);
setDirection(callMeGesture, fp.Finger.Thumb, fp.FingerDirection.DiagonalUpLeft);
setDirection(callMeGesture, fp.Finger.Pinky, fp.FingerDirection.HorizontalRight);

async function runHandpose(detect) {
  await tf.setBackend('webgl');
  await tf.ready();
  const net = await handpose.load();
  console.log("Handpose model loaded with backend:", tf.getBackend());
  const interval = setInterval(() => {
    detect(net);
  }, 100);
  return () => clearInterval(interval);
}

function HandSign() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [gesture, setGesture] = useState(null);
  const [gestureConfidence, setGestureConfidence] = useState(0);

  const detect = async (net) => {
    if (
      typeof webcamRef.current !== "undefined" &&
      webcamRef.current !== null &&
      webcamRef.current.video.readyState === 4
    ) {
      const video = webcamRef.current.video;
      const videoWidth = webcamRef.current.video.videoWidth;
      const videoHeight = webcamRef.current.video.videoHeight;

      webcamRef.current.video.width = videoWidth;
      webcamRef.current.video.height = videoHeight;
      canvasRef.current.width = videoWidth;
      canvasRef.current.height = videoHeight;

      const hand = await net.estimateHands(video);
      if (hand.length > 0) {
        const GE = new fp.GestureEstimator([
          victoryGesture,
          thumbsUpGesture,
          openPalmGesture,
          closedFistGesture,
          pointUpGesture,
          okSignGesture,
          rockOnGesture,
          callMeGesture
        ]);

        const estimatedGestures = await GE.estimate(hand[0].landmarks, 7.5);
        if (estimatedGestures.gestures?.length > 0) {
          const gesturesByConfidence = estimatedGestures.gestures.sort(
            (a, b) => b.confidence - a.confidence
          );

          const bestGesture = gesturesByConfidence[0];
          if (bestGesture.confidence > 7.5) {
            setGesture(bestGesture.name);
            setGestureConfidence(bestGesture.confidence);
          } else {
            setGesture(null);
            setGestureConfidence(0);
          }
        } else {
          setGesture(null);
          setGestureConfidence(0);
        }
      } else {
        setGesture(null);
        setGestureConfidence(0);
      }

      const ctx = canvasRef.current.getContext("2d");
      drawHand(hand, ctx);
    }
  };

  useEffect(() => {
    const interval = runHandpose(detect);
    return () => clearInterval(interval); // This cleans up the interval when the component unmounts
  }, []);

  return (
    <><div className="bento" id="B0">
      <h1>HAND GESTURE RECOGNITION SYSTEM</h1>
    </div>
    <br />
    <div className="bentoWrapper">
      <div className="bento ml-8 w-full" id="B1"><Webcam ref={webcamRef} style={{ borderRadius: '20px' }} />
        <canvas ref={canvasRef} style={{ position: "fixed" }} />
      </div>
      <div className="bento min-h-[150px] flex flex-col p-4" id="B2">
        <div className="h-8 text-xl font-bold mb-2">
          {gesture ? gesture.replace(/_/g, ' ').toUpperCase() : "NO GESTURE DETECTED"}
        </div>
        <div className="h-12 text-base py-5">
          {gesture ? `Gesture: ${gesture}` : "Show a hand gesture to begin"}
        </div>
        <div className="text-sm text-green-300 mt-auto">
          {gesture && `Confidence: ${gestureConfidence.toFixed(2)}%`}
        </div>
      </div>
      <div className="bento" id="B3"><div style={{ fontWeight: "bold", fontSize: '20px', textAlign: 'center', paddingBottom: '10px' }}>AVAILABLE GESTURE CLASSES</div>
        {Object.keys(victoryGesture, thumbsUpGesture, openPalmGesture, closedFistGesture, pointUpGesture, okSignGesture, rockOnGesture, callMeGesture).map(key => (
          <div key={key} style={{ marginBottom: "3px", fontSize: '15px', padding: '5px', textAlign: 'center' }}>
            {key.replace(/_/g, ' ')}
          </div>
        ))}
      </div>
      <div className="bento flex items-center justify-center" id="B4">
        <div style={{ fontWeight: "bold", fontSize: '20px', marginBottom: "5px", textAlign: 'center' }}>REPRESENTATIONAL EMOJI</div>
        <div style={{
          fontSize: "200px",
          width: "200px",
          height: "200px",
          display: "flex",
          paddingLeft: '50px',
          alignItems: "center",
          justifyContent: "center"
        }}>
          {gesture ? '👍' : "🫡"} {/* Update as needed */}
        </div>
      </div>
    </div>
  </>);
}

export default HandSign;
