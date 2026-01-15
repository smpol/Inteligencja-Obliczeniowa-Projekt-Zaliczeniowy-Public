import React, { useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import { Camera, Upload, Image } from 'lucide-react';

export const CameraView = ({ onCapture, isStreaming }) => {
  const videoRef = useRef(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let stream = null;

    const startCamera = async () => {
      // Only try camera on desktop (will be hidden on mobile via CSS)
      if (window.innerWidth < 768) {
        return;
      }

      try {
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
          stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "environment" }
          });
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
          }
        }
      } catch (err) {
        console.error("Error accessing camera:", err);
        setError("Camera access denied");
      }
    };

    if (isStreaming) {
      startCamera();
    }

    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, [isStreaming]);

  const handleCapture = () => {
    if (videoRef.current) {
      const canvas = document.createElement('canvas');
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(videoRef.current, 0, 0);

      canvas.toBlob((blob) => {
        if (blob && onCapture) {
          const file = new File([blob], "capture.jpg", { type: "image/jpeg" });
          const previewUrl = canvas.toDataURL('image/jpeg');
          onCapture(file, previewUrl, { width: canvas.width, height: canvas.height });
        }
      }, 'image/jpeg', 0.9);
    }
  };

  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      const url = URL.createObjectURL(file);
      onCapture(file, url, { width: 0, height: 0 });
    }
  };

  return (
    <div className="w-full">
      {/* Mobile View - md:hidden */}
      <div className="md:hidden ios-card-solid flex flex-col items-center justify-center p-8" style={{ minHeight: '50vh' }}>
        <div className="w-20 h-20 rounded-full bg-[var(--ios-blue)]/10 flex items-center justify-center mb-6">
          <Camera className="w-10 h-10 text-[var(--ios-blue)]" />
        </div>

        <h2 className="ios-title2 text-white text-center mb-2">Scan Your Waste</h2>
        <p className="ios-subhead text-[var(--text-secondary)] text-center mb-8 max-w-xs">
          Take a photo or choose from your library to identify the waste type
        </p>

        {/* Take Photo Button */}
        <label className="w-full cursor-pointer mb-3">
          <div className="ios-button ios-button-primary w-full">
            <Camera className="w-5 h-5" />
            Take Photo
          </div>
          <input
            type="file"
            accept="image/*"
            capture="environment"
            className="hidden"
            onChange={handleFileSelect}
          />
        </label>

        {/* Choose from Library */}
        <label className="w-full cursor-pointer">
          <div className="ios-button ios-button-secondary w-full">
            <Image className="w-5 h-5" />
            Choose from Library
          </div>
          <input
            type="file"
            accept="image/*"
            className="hidden"
            onChange={handleFileSelect}
          />
        </label>
      </div>

      {/* Desktop View - hidden md:block */}
      <div className="hidden md:block relative w-full aspect-[4/3] ios-card-solid overflow-hidden">
        {!error ? (
          <>
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="w-full h-full object-cover"
            />

            {/* Scanning Line */}
            <motion.div
              className="absolute w-full h-0.5 bg-[var(--ios-blue)] opacity-80"
              style={{ boxShadow: '0 0 20px var(--ios-blue)' }}
              animate={{ top: ["10%", "90%", "10%"] }}
              transition={{ duration: 2.5, repeat: Infinity, ease: "linear" }}
            />

            {/* Corner Guides */}
            <div className="absolute inset-4 pointer-events-none">
              <div className="absolute top-0 left-0 w-8 h-8 border-t-2 border-l-2 border-white/60 rounded-tl-lg" />
              <div className="absolute top-0 right-0 w-8 h-8 border-t-2 border-r-2 border-white/60 rounded-tr-lg" />
              <div className="absolute bottom-0 left-0 w-8 h-8 border-b-2 border-l-2 border-white/60 rounded-bl-lg" />
              <div className="absolute bottom-0 right-0 w-8 h-8 border-b-2 border-r-2 border-white/60 rounded-br-lg" />
            </div>
          </>
        ) : (
          <div className="w-full h-full flex flex-col items-center justify-center p-6 text-center">
            <Camera className="w-12 h-12 text-[var(--text-tertiary)] mb-4" />
            <p className="ios-body text-[var(--text-secondary)] mb-2">Camera unavailable</p>
            <p className="ios-caption1 text-[var(--text-tertiary)]">Use the button below to upload</p>
          </div>
        )}

        {/* Capture/Upload Button */}
        <div className="absolute bottom-6 left-0 right-0 flex justify-center gap-4">
          {!error && (
            <button
              onClick={handleCapture}
              className="w-16 h-16 rounded-full bg-white/10 backdrop-blur-md border-4 border-white flex items-center justify-center active:scale-95 transition-transform"
            >
              <div className="w-12 h-12 bg-white rounded-full" />
            </button>
          )}
          <button
            onClick={() => document.getElementById('desktop-upload').click()}
            className="ios-button ios-button-secondary h-16 px-6"
          >
            <Image className="w-5 h-5" />
            Upload
          </button>
          <input
            id="desktop-upload"
            type="file"
            accept="image/*"
            className="hidden"
            onChange={handleFileSelect}
          />
        </div>
      </div>
    </div>
  );
};
