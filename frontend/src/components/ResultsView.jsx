import React, { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronDown, ChevronUp, Recycle, Leaf, BarChart3 } from 'lucide-react';

const binColors = {
    'cardboard': '#007AFF', 'paper': '#007AFF',
    'glass': '#34C759', 'brown-glass': '#34C759', 'green-glass': '#34C759', 'white-glass': '#34C759',
    'metal': '#FF9500', 'plastic': '#FFCC00', 'metal_plastic': '#FFCC00',
    'biological': '#8E6B4A',
    'battery': '#FF3B30',
    'clothes': '#AF52DE', 'shoes': '#AF52DE', 'textile': '#AF52DE',
    'trash': '#8E8E93', 'unknown': '#8E8E93'
};


const classDisplayNames = {
    'metal_plastic': 'Plastic & Metal',
    'glass': 'Glass',
    'paper': 'Paper',
    'textile': 'Textiles',
    'biological': 'Organic',
    'battery': 'Batteries',
    'trash': 'Mixed Waste'
};

export const ResultsView = ({ imageSrc, detections, onReset }) => {
    const canvasRef = useRef(null);
    const [expandedMaterial, setExpandedMaterial] = useState(false);
    const [expandedComparison, setExpandedComparison] = useState(false);
    const [comparisons, setComparisons] = useState(null);
    const [loadingComparisons, setLoadingComparisons] = useState(true);

    // Draw bounding boxes
    useEffect(() => {
        if (!canvasRef.current || !detections || !imageSrc) return;

        const img = new Image();
        img.src = imageSrc;
        img.onload = () => {
            const canvas = canvasRef.current;
            const ctx = canvas.getContext('2d');

            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);

            detections.forEach((det) => {
                const [x1, y1, x2, y2] = det.box;
                let bx = x1, by = y1, bw = x2 - x1, bh = y2 - y1;

                if (det.box_norm) {
                    const [nx1, ny1, nx2, ny2] = det.box_norm;
                    bx = nx1 * canvas.width;
                    by = ny1 * canvas.height;
                    bw = (nx2 - nx1) * canvas.width;
                    bh = (ny2 - ny1) * canvas.height;
                }

                const color = det.binColor || binColors[det.class.toLowerCase()] || '#007AFF';

                ctx.strokeStyle = color;
                ctx.lineWidth = Math.max(3, canvas.width / 200);
                ctx.beginPath();
                const r = 8;
                ctx.moveTo(bx + r, by);
                ctx.lineTo(bx + bw - r, by);
                ctx.quadraticCurveTo(bx + bw, by, bx + bw, by + r);
                ctx.lineTo(bx + bw, by + bh - r);
                ctx.quadraticCurveTo(bx + bw, by + bh, bx + bw - r, by + bh);
                ctx.lineTo(bx + r, by + bh);
                ctx.quadraticCurveTo(bx, by + bh, bx, by + bh - r);
                ctx.lineTo(bx, by + r);
                ctx.quadraticCurveTo(bx, by, bx + r, by);
                ctx.stroke();
            });
        };
    }, [imageSrc, detections]);

    // Fetch model comparisons
    useEffect(() => {
        let cancelled = false;

        const fetchComparisons = async () => {
            if (!imageSrc) {
                setLoadingComparisons(false);
                return;
            }

            try {
                const response = await fetch(imageSrc);
                const blob = await response.blob();
                const file = new File([blob], "comparison.jpg", { type: "image/jpeg" });

                const { compareModels } = await import('../api');
                const data = await compareModels(file);

                if (!cancelled && data && data.detections) {
                    setComparisons(data.detections);
                }
            } catch (err) {
                console.error("Comparison fetch failed", err);
            } finally {
                if (!cancelled) setLoadingComparisons(false);
            }
        };

        const timer = setTimeout(fetchComparisons, 300);
        return () => {
            cancelled = true;
            clearTimeout(timer);
        };
    }, [imageSrc]);

    if (!detections || detections.length === 0) {
        return (
            <div className="ios-card-solid p-8 text-center">
                <p className="ios-headline text-white">No waste detected</p>
                <p className="ios-subhead text-[var(--text-secondary)] mt-2">Try another photo</p>
            </div>
        );
    }

    const mainDet = detections[0];
    const binType = mainDet.bin || 'MIXED';
    const binColor = mainDet.binColor || '#374151';

    // Bin display info - ENGLISH, focus on bin name
    const binDisplayInfo = {
        'PAPER': {
            name: 'Paper & Cardboard',
            icon: '📄',
            hint: 'Blue Bin',
            description: 'Recyclable paper products'
        },
        'PLASTIC': {
            name: 'Plastic & Metal',
            icon: '♻️',
            hint: 'Yellow Bin',
            description: 'Recyclable plastics and metals'
        },
        'GLASS': {
            name: 'Glass',
            icon: '🫙',
            hint: 'Green Bin',
            description: 'All glass containers'
        },
        'MIXED': {
            name: 'Mixed Waste',
            icon: '🗑️',
            hint: 'Black Bin',
            description: 'Non-recyclable waste'
        },
        'BIO': {
            name: 'Organic Waste',
            icon: '🍃',
            hint: 'Brown Bin',
            description: 'Food scraps and organic materials'
        },
        'HAZARDOUS': {
            name: 'Hazardous Waste',
            icon: '⚠️',
            hint: 'Collection Point',
            description: 'Batteries and dangerous materials'
        }
    };

    const info = binDisplayInfo[binType] || {
        name: 'Unknown',
        icon: '❓',
        hint: 'Check local guidelines',
        description: 'Classification uncertain'
    };

    const objectComparison = comparisons?.[0]?.comparison;

    return (
        <div className="flex flex-col gap-4 ios-animate-in">
            {/* Image Preview */}
            <div className="ios-card-solid overflow-hidden">
                <canvas ref={canvasRef} className="w-full h-auto block" />
            </div>

            {/* Main Result Card - BIN FOCUSED */}
            <div className="ios-card-solid overflow-hidden">
                {/* BIN Header - MAIN FOCUS */}
                <div className="p-6 text-center" style={{ backgroundColor: `${binColor}15` }}>
                    <div
                        className="w-20 h-20 rounded-3xl flex items-center justify-center text-4xl mx-auto mb-4"
                        style={{ backgroundColor: binColor }}
                    >
                        {info.icon}
                    </div>
                    <h1 className="ios-large-title text-white mb-2">
                        {info.name}
                    </h1>
                    <p className="ios-title3 mb-1" style={{ color: binColor }}>
                        {info.hint}
                    </p>
                    <p className="ios-caption1 text-[var(--text-secondary)]">
                        {info.description}
                    </p>
                </div>

                {/* Material Details - EXPANDABLE */}
                <div className="border-t border-white/10">
                    <button
                        onClick={() => setExpandedMaterial(!expandedMaterial)}
                        className="w-full p-4 flex items-center justify-between active:bg-white/5"
                    >
                        <div className="flex items-center gap-3">
                            <div className="text-2xl">🔍</div>
                            <div className="text-left">
                                <p className="ios-subhead text-white">Material Details</p>
                                <p className="ios-caption1 text-[var(--text-tertiary)] capitalize">
                                    {classDisplayNames[mainDet.class] || mainDet.class.replace(/[-_]/g, ' ')} • {mainDet.confidence}% confidence
                                </p>
                            </div>
                        </div>
                        {expandedMaterial ? (
                            <ChevronUp className="w-5 h-5 text-[var(--text-tertiary)]" />
                        ) : (
                            <ChevronDown className="w-5 h-5 text-[var(--text-tertiary)]" />
                        )}
                    </button>

                    <AnimatePresence>
                        {expandedMaterial && (
                            <motion.div
                                initial={{ height: 0, opacity: 0 }}
                                animate={{ height: 'auto', opacity: 1 }}
                                exit={{ height: 0, opacity: 0 }}
                                className="overflow-hidden"
                            >
                                <div className="px-4 pb-4 space-y-3">
                                    {/* Detected Material */}
                                    <div className="p-3 rounded-xl bg-[var(--ios-gray5)]">
                                        <p className="ios-caption1 text-[var(--text-tertiary)] mb-1">Detected Material</p>
                                        <p className="ios-headline text-white capitalize">
                                            {classDisplayNames[mainDet.class] || mainDet.class.replace(/[-_]/g, ' ')}
                                        </p>
                                        <p className="ios-caption1 text-[var(--text-secondary)] mt-1">
                                            Confidence: {mainDet.confidence}%
                                        </p>
                                    </div>

                                </div>
                            </motion.div>
                        )}
                    </AnimatePresence>
                </div>

                {/* Model Comparison */}
                <div className="border-t border-white/10">
                    <button
                        onClick={() => setExpandedComparison(!expandedComparison)}
                        className="w-full p-4 flex items-center justify-between active:bg-white/5"
                        disabled={loadingComparisons}
                    >
                        <div className="flex items-center gap-2">
                            <BarChart3 className="w-5 h-5 text-[var(--text-secondary)]" />
                            <span className="ios-subhead text-[var(--text-secondary)]">
                                {loadingComparisons ? 'Loading models...' : 'Compare AI Models'}
                            </span>
                        </div>
                        {!loadingComparisons && (
                            expandedComparison ? (
                                <ChevronUp className="w-5 h-5 text-[var(--text-tertiary)]" />
                            ) : (
                                <ChevronDown className="w-5 h-5 text-[var(--text-tertiary)]" />
                            )
                        )}
                    </button>

                    <AnimatePresence>
                        {expandedComparison && objectComparison && (
                            <motion.div
                                initial={{ height: 0, opacity: 0 }}
                                animate={{ height: 'auto', opacity: 1 }}
                                exit={{ height: 0, opacity: 0 }}
                                className="overflow-hidden"
                            >
                                <div className="px-4 pb-4 space-y-2">
                                    {Object.entries(objectComparison).map(([modelName, preds]) => {
                                        const prediction = Array.isArray(preds) && preds[0];
                                        if (!prediction) return null;

                                        const predColor = binColors[prediction.class?.toLowerCase()] || '#8E8E93';
                                        const isMainModel = modelName === 'MLP';

                                        return (
                                            <div
                                                key={modelName}
                                                className={`p-3 rounded-xl ${isMainModel ? 'border border-[var(--ios-blue)]/30' : ''}`}
                                                style={{ backgroundColor: 'var(--ios-gray5)' }}
                                            >
                                                <div className="flex items-center justify-between mb-2">
                                                    <div className="flex items-center gap-2">
                                                        <span className="ios-caption1 font-semibold text-white uppercase tracking-wide">
                                                            {modelName}
                                                        </span>
                                                        {isMainModel && (
                                                            <span className="px-2 py-0.5 rounded-full bg-[var(--ios-blue)] text-white text-[10px] font-bold">
                                                                BEST
                                                            </span>
                                                        )}
                                                    </div>
                                                    <span className="ios-body font-semibold text-[var(--ios-blue)]">
                                                        {Math.round(prediction.confidence)}%
                                                    </span>
                                                </div>
                                                <div className="flex items-center gap-2">
                                                    <div
                                                        className="w-3 h-3 rounded-full"
                                                        style={{ backgroundColor: predColor }}
                                                    />
                                                    <span className="ios-subhead text-white capitalize">
                                                        {classDisplayNames[prediction.class] || prediction.class?.replace(/[-_]/g, ' ')}
                                                    </span>
                                                </div>
                                            </div>
                                        );
                                    })}
                                </div>
                            </motion.div>
                        )}
                    </AnimatePresence>
                </div>
            </div>

            {/* Scan Again Button */}
            <button
                onClick={onReset}
                className="ios-button ios-button-primary w-full"
            >
                <Leaf className="w-5 h-5" />
                Scan Another Item
            </button>
        </div>
    );
};
