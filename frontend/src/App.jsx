import React, { useState } from 'react';
import { CameraView } from './components/CameraView';
import { ResultsView } from './components/ResultsView';
import { sendPrediction } from './api';
import { Leaf, Image as ImageIcon, Loader2, ChevronLeft } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

function App() {
  const [view, setView] = useState('camera'); // 'camera', 'loading', 'results'
  const [imageSrc, setImageSrc] = useState(null);
  const [detections, setDetections] = useState([]);
  const [error, setError] = useState(null);

  const handleCapture = async (file, previewUrl, dimensions) => {
    setImageSrc(previewUrl);
    setView('loading');
    setError(null);

    try {
      const data = await sendPrediction(file);
      if (data && data.detections) {
        setDetections(data.detections);
        setView('results');
      } else {
        throw new Error('No detections found');
      }
    } catch (err) {
      console.error(err);
      setError(err.message || 'Failed to analyze image');
      setView('camera');
    }
  };

  return (
    <div className="flex flex-col min-h-screen min-h-[100dvh] ios-scroll-hide">

      {/* iOS Navigation Bar */}
      <header className="ios-navbar">
        <div className="flex items-center justify-between">
          {view === 'results' ? (
            <button
              onClick={() => setView('camera')}
              className="flex items-center gap-1 text-[var(--ios-blue)] ios-body font-normal active:opacity-60 transition-opacity"
            >
              <ChevronLeft className="w-5 h-5" />
              Back
            </button>
          ) : (
            <div className="w-16" />
          )}

          <div className="flex items-center gap-2">
            <Leaf className="w-6 h-6 text-[var(--ios-green)]" />
            <span className="ios-headline">EcoScan</span>
          </div>

          <div className="w-16" />
        </div>
      </header>

      {/* Large Title (iOS style) - only on camera view */}
      {view === 'camera' && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="px-4 pt-4 pb-2"
        >
          <h1 className="ios-large-title text-white">Scan Waste</h1>
          <p className="ios-subhead text-[var(--text-secondary)] mt-1">
            Take a photo to identify and sort your waste
          </p>
        </motion.div>
      )}

      {/* Main Content Area */}
      <main className="flex-1 px-4 pb-8">
        <AnimatePresence mode='wait'>
          {view === 'camera' && (
            <motion.div
              key="camera"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="w-full"
            >
              <CameraView onCapture={handleCapture} isStreaming={true} />
            </motion.div>
          )}

          {view === 'loading' && (
            <motion.div
              key="loading"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex flex-col items-center justify-center gap-4 ios-card-solid p-12 mt-4"
              style={{ minHeight: '60vh' }}
            >
              <div className="relative">
                <Loader2 className="w-12 h-12 text-[var(--ios-blue)] animate-spin" />
              </div>
              <p className="ios-headline text-white">Analyzing...</p>
              <p className="ios-footnote text-[var(--text-secondary)]">
                Identifying waste type
              </p>
            </motion.div>
          )}

          {view === 'results' && (
            <motion.div
              key="results"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
            >
              <ResultsView
                imageSrc={imageSrc}
                detections={detections}
                onReset={() => setView('camera')}
              />
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* Error Toast (iOS style) */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ y: 100, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            exit={{ y: 100, opacity: 0 }}
            className="fixed bottom-8 left-4 right-4 mx-auto max-w-sm z-50"
          >
            <div className="ios-card-solid p-4 flex items-center gap-3 border border-[var(--ios-red)]/30">
              <div className="w-8 h-8 rounded-full bg-[var(--ios-red)] flex items-center justify-center flex-shrink-0">
                <span className="text-white text-lg">!</span>
              </div>
              <div className="flex-1">
                <p className="ios-footnote text-[var(--ios-red)]">Error</p>
                <p className="ios-subhead text-white">{error}</p>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default App;
