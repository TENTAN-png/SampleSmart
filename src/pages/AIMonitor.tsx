import React, { useState, useEffect } from 'react';
import { useProjectStore } from '../store/useProjectStore';
import { api } from '../lib/api';
import {
    Activity,
    Search,
    Terminal,
    Cpu,
    Zap,
    Eye,
    Mic2,
    FileText,
    RefreshCw,
    type LucideIcon
} from 'lucide-react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
    return twMerge(clsx(inputs));
}

interface PipelineStage {
    id: string;
    name: string;
    icon: LucideIcon;
    status: 'pending' | 'processing' | 'completed' | 'error';
    confidence: number;
    duration: string;
    logs: string[];
}

const initialStages: PipelineStage[] = [
    {
        id: 'yolo',
        name: 'YOLO Object Detection',
        icon: Search,
        status: 'completed',
        confidence: 98.2,
        duration: '12s',
        logs: ['[INFO] Model loaded: YOLOv11-Cinema', '[DETECTION] 4 actors detected', '[DETECTION] Primary camera movement: Pan Left']
    },
    {
        id: 'ocr',
        name: 'Slate OCR Parser',
        icon: FileText,
        status: 'completed',
        confidence: 99.5,
        duration: '4s',
        logs: ['[OCR] Scene: 12B', '[OCR] Take: 4', '[OCR] Roll: A002']
    },
    {
        id: 'sync',
        name: 'Audio-Video Sync',
        icon: Zap,
        status: 'completed',
        confidence: 100,
        duration: '2s',
        logs: ['[SYNC] Waveform matching successful', '[SYNC] Offset: -0.002s', '[SYNC] Sample Rate: 48kHz']
    },
    {
        id: 'whisper',
        name: 'Whisper Transcription',
        icon: Mic2,
        status: 'processing',
        confidence: 85.4,
        duration: 'Ongoing',
        logs: ['[TRANSCRIPT] Processing Character A...', '[STT] Confidence score: 0.85', '[TRANSCRIPT] 14 minutes remaining']
    },
    {
        id: 'nlp',
        name: 'NLP Sentiment Analysis',
        icon: Eye,
        status: 'pending',
        confidence: 0,
        duration: '-',
        logs: []
    },
    {
        id: 'scoring',
        name: 'Take Scoring Engine',
        icon: Activity,
        status: 'pending',
        confidence: 0,
        duration: '-',
        logs: []
    },
];

export const AIMonitor = () => {
    const getProcessingStatus = useProjectStore(state => state.getProcessingStatus);
    const [takes, setTakes] = useState<any[]>([]);
    const [selectedTakeId, setSelectedTakeId] = useState<number | null>(null);
    const [activeStage, setActiveStage] = useState<string>('Frame & Data Analysis');
    const [statusData, setStatusData] = useState<any>(null);
    const [isPolling, setIsPolling] = useState(false);

    // Fetch takes on mount
    useEffect(() => {
        let isMounted = true;
        let timeoutId: any;

        const fetchTakes = async () => {
            if (!isMounted) return;
            try {
                const response = await api.media.listTakes();
                if (isMounted) {
                    setTakes(response.data);
                    if (response.data.length > 0 && !selectedTakeId) {
                        setSelectedTakeId(response.data[0].id);
                    }
                }
            } catch (err) {
                console.error("Failed to fetch takes", err);
            } finally {
                if (isMounted) {
                    timeoutId = setTimeout(fetchTakes, 5000); // 5 second interval
                }
            }
        };

        fetchTakes();
        return () => {
            isMounted = false;
            clearTimeout(timeoutId);
        };
    }, [selectedTakeId]);

    // Poll status for selected take
    useEffect(() => {
        if (!selectedTakeId) return;

        let isMounted = true;
        let timeoutId: any;

        const poll = async () => {
            if (!isMounted) return;

            try {
                setIsPolling(true);
                const data = await getProcessingStatus(selectedTakeId);
                if (isMounted && data) {
                    setStatusData(data);
                }
            } catch (err) {
                console.error("Poll error", err);
            } finally {
                if (isMounted) {
                    setIsPolling(false);
                    timeoutId = setTimeout(poll, 3000); // 3 second interval
                }
            }
        };

        poll();
        return () => {
            isMounted = false;
            clearTimeout(timeoutId);
        };
    }, [selectedTakeId, getProcessingStatus]);

    // Map backend stages to UI components
    const stages: PipelineStage[] = [
        { id: 'Frame & Data Analysis', name: 'Frame & Data Analysis', icon: Search, status: statusData?.stages?.['Frame & Data Analysis'] || 'pending', confidence: Math.round(statusData?.cv?.confidence * 100) || 0, duration: '12s', logs: statusData?.logs?.filter((l: string) => l.toLowerCase().includes('frame') || l.toLowerCase().includes('data')) || [] },
        { id: 'Audio Processing', name: 'Audio Processing', icon: Mic2, status: statusData?.stages?.['Audio Processing'] || 'pending', confidence: Math.round(statusData?.audio?.confidence * 100) || 0, duration: 'Ongoing', logs: statusData?.logs?.filter((l: string) => l.toLowerCase().includes('audio')) || [] },
        { id: 'Script Alignment', name: 'Script Alignment', icon: FileText, status: statusData?.stages?.['Script Alignment'] || 'pending', confidence: Math.round(statusData?.nlp?.confidence * 100) || 0, duration: '-', logs: statusData?.logs?.filter((l: string) => l.toLowerCase().includes('script') || l.toLowerCase().includes('align')) || [] },
        { id: 'Intelligence Scoring', name: 'Intelligence Scoring', icon: Activity, status: statusData?.stages?.['Intelligence Scoring'] || 'pending', confidence: 100, duration: '-', logs: statusData?.logs?.filter((l: string) => l.toLowerCase().includes('score') || l.toLowerCase().includes('intellig')) || [] },
        { id: 'Intent Indexing', name: 'Intent Indexing', icon: Zap, status: statusData?.stages?.['Intent Indexing'] || 'pending', confidence: 100, duration: '-', logs: statusData?.logs?.filter((l: string) => l.toLowerCase().includes('intent') || l.toLowerCase().includes('index')) || [] },
    ];

    const selectedStage = stages.find(s => s.id === activeStage);
    const selectedTake = takes.find(t => t.id === selectedTakeId);

    const handleStartProcessing = async () => {
        if (!selectedTakeId) return;
        try {
            await api.processing.start(selectedTakeId);
            // Re-fetch status immediately
            const data = await getProcessingStatus(selectedTakeId);
            if (data) setStatusData(data);
        } catch (err) {
            console.error("Failed to start processing", err);
        }
    };

    return (
        <div className="h-full flex flex-col p-8 space-y-6">
            <header className="flex justify-between items-start">
                <div>
                    <h1 className="text-3xl font-bold text-white mb-2">AI Processing Pipeline</h1>
                    <p className="text-editor-muted italic">Neural Engine monitoring cluster: Alpha-7</p>
                </div>
                <div className="flex gap-3">
                    {selectedTakeId && (
                        <button
                            onClick={handleStartProcessing}
                            className="flex items-center gap-2 px-4 py-2 bg-primary hover:bg-primary-dark rounded-md transition-colors text-xs font-bold uppercase tracking-widest text-white"
                        >
                            <Zap size={14} />
                            Start Processing
                        </button>
                    )}
                    <button className="flex items-center gap-2 px-4 py-2 bg-editor-track rounded-md hover:bg-editor-border transition-colors text-xs font-bold uppercase tracking-widest">
                        <RefreshCw size={14} className={isPolling ? "animate-spin" : ""} />
                        Refresh Status
                    </button>
                    <button className="flex items-center gap-2 px-4 py-2 bg-editor-track rounded-md hover:bg-editor-border transition-colors text-xs font-bold uppercase tracking-widest text-primary">
                        <Activity size={14} />
                        Live Feed
                    </button>
                </div>
            </header>

            <div className="flex-1 grid grid-cols-1 lg:grid-cols-4 gap-6 min-h-0">
                {/* Take Browser */}
                <div className="lg:col-span-1 glass-panel rounded-xl flex flex-col min-h-0">
                    <div className="p-4 border-b border-editor-border font-bold text-sm uppercase tracking-widest flex items-center gap-2">
                        <Zap size={16} className="text-accent" />
                        Takes Library
                    </div>
                    <div className="flex-1 overflow-y-auto p-4 space-y-2">
                        {takes.length === 0 ? (
                            <div className="text-center py-8 text-editor-muted text-xs italic">
                                No uploads found. Please ingest media first.
                            </div>
                        ) : (
                            takes.map((take) => (
                                <button
                                    key={take.id}
                                    onClick={() => setSelectedTakeId(take.id)}
                                    className={cn(
                                        "w-full p-4 rounded-lg transition-all border text-left",
                                        selectedTakeId === take.id
                                            ? "bg-accent/10 border-accent/50"
                                            : "bg-surface/30 border-white/5 hover:border-white/10"
                                    )}
                                >
                                    <div className="font-bold text-sm truncate text-white">{take.file_name}</div>
                                    <div className="flex items-center justify-between mt-2">
                                        <div className="text-[10px] text-editor-muted uppercase font-bold tracking-tighter">
                                            ID: {take.id} • Take {take.number}
                                        </div>
                                        {take.id === selectedTakeId && (
                                            <div className="w-2 h-2 bg-accent rounded-full animate-pulse" />
                                        )}
                                    </div>
                                </button>
                            ))
                        )}
                    </div>
                </div>

                {/* Pipeline Stages */}
                <div className="lg:col-span-1 glass-panel rounded-xl flex flex-col min-h-0">
                    <div className="p-4 border-b border-editor-border font-bold text-sm uppercase tracking-widest flex items-center gap-2">
                        <Cpu size={16} className="text-primary" />
                        Pipeline Stages
                    </div>
                    <div className="flex-1 overflow-y-auto p-4 space-y-2">
                        {!selectedTakeId ? (
                            <div className="text-center py-8 text-editor-muted text-xs">Select a take to view status</div>
                        ) : (
                            stages.map((stage) => (
                                <button
                                    key={stage.id}
                                    onClick={() => setActiveStage(stage.id)}
                                    className={cn(
                                        "w-full flex items-center gap-4 p-4 rounded-lg transition-all border text-left group",
                                        activeStage === stage.id
                                            ? "bg-primary/10 border-primary shadow-[0_0_15px_rgba(59,130,246,0.2)]"
                                            : "bg-surface/50 border-white/5 hover:border-white/20"
                                    )}
                                >
                                    <div className={cn(
                                        "w-10 h-10 rounded-full flex items-center justify-center transition-colors",
                                        stage.status === 'completed' ? "bg-success/20 text-success" :
                                            stage.status === 'processing' ? "bg-primary/20 text-primary animate-pulse" :
                                                "bg-editor-track text-editor-muted"
                                    )}>
                                        <stage.icon size={20} />
                                    </div>
                                    <div className="flex-1 min-w-0">
                                        <div className="font-bold text-sm truncate">{stage.name}</div>
                                        <div className="text-[10px] uppercase text-editor-muted font-bold tracking-tighter">
                                            {stage.status} • {stage.duration}
                                        </div>
                                    </div>
                                    {stage.status === 'completed' && (
                                        <div className="text-xs font-mono font-bold text-success">{stage.confidence}%</div>
                                    )}
                                </button>
                            ))
                        )}
                    </div>
                </div>

                {/* Live Console Output */}
                <div className="lg:col-span-2 flex flex-col gap-6 min-h-0">
                    <div className="glass-panel rounded-xl flex flex-col flex-1 min-h-0">
                        <div className="p-4 border-b border-editor-border font-bold text-sm uppercase tracking-widest flex items-center justify-between">
                            <div className="flex items-center gap-2">
                                <Terminal size={16} className="text-accent" />
                                Live Console Output: {selectedStage?.name}
                            </div>
                            {selectedTake && (
                                <div className="text-[10px] text-editor-muted flex items-center gap-2">
                                    <span className="bg-editor-track px-2 py-0.5 rounded">UUID: {selectedTake.id}</span>
                                    <span>Frame range: [Live Channel 1]</span>
                                </div>
                            )}
                        </div>
                        <div className="flex-1 bg-black/40 p-6 font-mono text-xs overflow-y-auto space-y-2">
                            {selectedStage?.logs.length === 0 ? (
                                <div className="text-editor-muted italic">Waiting for stage to initialize...</div>
                            ) : (
                                selectedStage?.logs.map((log, i) => (
                                    <div key={i} className="flex gap-4">
                                        <span className="text-editor-muted/40 font-bold">[{1024 + i}]</span>
                                        <span className={cn(
                                            log.includes('ERROR') ? "text-danger" :
                                                log.includes('Success') || log.includes('complete') ? "text-success" :
                                                    log.includes('Starting') ? "text-accent" :
                                                        "text-editor-muted"
                                        )}>{log}</span>
                                    </div>
                                ))
                            )}
                            {selectedStage?.status === 'processing' && (
                                <div className="flex gap-4 animate-pulse">
                                    <span className="text-editor-muted/40 font-bold">[{1024 + (selectedStage?.logs.length || 0)}]</span>
                                    <span className="text-primary italic">Neural pass in progress... (Syncing to cluster)</span>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};
