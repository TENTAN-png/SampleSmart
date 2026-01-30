import React, { useState } from 'react';
import {
    Smile,
    Activity,
    TrendingUp,
    Brain,
    BarChart2
} from 'lucide-react';
import {
    AreaChart,
    Area,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    Radar,
    RadarChart,
    PolarGrid,
    PolarAngleAxis,
    PolarRadiusAxis
} from 'recharts';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
    return twMerge(clsx(inputs));
}

const emotionData = [
    { frame: 0, joy: 10, sadness: 20, anger: 5, intensity: 15 },
    { frame: 100, joy: 65, sadness: 15, anger: 8, intensity: 45 },
    { frame: 200, joy: 85, sadness: 10, anger: 12, intensity: 80 },
    { frame: 300, joy: 40, sadness: 45, anger: 20, intensity: 60 },
    { frame: 400, joy: 15, sadness: 70, anger: 40, intensity: 75 },
    { frame: 500, joy: 5, sadness: 85, anger: 60, intensity: 90 },
    { frame: 600, joy: 2, sadness: 40, anger: 90, intensity: 95 },
];

const performanceComparison = [
    { subject: 'Emotional Depth', A: 120, B: 110, fullMark: 150 },
    { subject: 'Consistency', A: 98, B: 130, fullMark: 150 },
    { subject: 'Script Adherence', A: 86, B: 130, fullMark: 150 },
    { subject: 'Engagement', A: 99, B: 100, fullMark: 150 },
    { subject: 'Subtlety', A: 85, B: 90, fullMark: 150 },
    { subject: 'Micro-expressions', A: 65, B: 85, fullMark: 150 },
];

export const EmotionPerformance = () => {
    const [activeCharacter, setActiveCharacter] = useState('Kale');

    return (
        <div className="p-8 space-y-8 max-w-7xl mx-auto h-full overflow-y-auto">
            <header className="flex justify-between items-start">
                <div>
                    <h1 className="text-3xl font-bold text-white mb-2 flex items-center gap-3">
                        <Smile className="text-accent" />
                        Emotion & Performance
                    </h1>
                    <p className="text-editor-muted italic">Neural facial analysis & sentiment heatmap | Sequence: S12_A01</p>
                </div>
                <div className="flex bg-surface border border-editor-border rounded-lg p-1">
                    <button
                        onClick={() => setActiveCharacter('Kale')}
                        className={cn("px-4 py-2 rounded font-bold text-xs uppercase tracking-widest transition-all", activeCharacter === 'Kale' ? "bg-primary text-white" : "text-editor-muted hover:text-white")}
                    >
                        Character: Kale
                    </button>
                    <button
                        onClick={() => setActiveCharacter('Marcus')}
                        className={cn("px-4 py-2 rounded font-bold text-xs uppercase tracking-widest transition-all", activeCharacter === 'Marcus' ? "bg-primary text-white" : "text-editor-muted hover:text-white")}
                    >
                        Character: Marcus
                    </button>
                </div>
            </header>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                <div className="lg:col-span-2 space-y-8">
                    <div className="glass-panel p-6 rounded-xl">
                        <div className="flex justify-between items-center mb-6">
                            <h3 className="text-lg font-bold flex items-center gap-2">
                                <TrendingUp className="text-primary" size={20} />
                                Intensity Heatmap (Sequence Timeline)
                            </h3>
                            <div className="flex gap-4">
                                <LegendItem color="#8b5cf6" label="Joy" />
                                <LegendItem color="#3b82f6" label="Sadness" />
                                <LegendItem color="#ef4444" label="Intensity" />
                            </div>
                        </div>
                        <div className="h-80">
                            <ResponsiveContainer width="100%" height="100%">
                                <AreaChart data={emotionData}>
                                    <defs>
                                        <linearGradient id="colorJoy" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.3} />
                                            <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0} />
                                        </linearGradient>
                                        <linearGradient id="colorSad" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                                            <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                                        </linearGradient>
                                    </defs>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#333" vertical={false} />
                                    <XAxis dataKey="frame" stroke="#888" fontSize={10} axisLine={false} tickLine={false} />
                                    <YAxis stroke="#888" fontSize={10} axisLine={false} tickLine={false} />
                                    <Tooltip
                                        contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #333', borderRadius: '8px' }}
                                        itemStyle={{ color: '#fff' }}
                                    />
                                    <Area type="monotone" dataKey="joy" stroke="#8b5cf6" fillOpacity={1} fill="url(#colorJoy)" strokeWidth={2} />
                                    <Area type="monotone" dataKey="sadness" stroke="#3b82f6" fillOpacity={1} fill="url(#colorSad)" strokeWidth={2} />
                                    <Area type="monotone" dataKey="intensity" stroke="#ef4444" fillOpacity={0} strokeDasharray="5 5" strokeWidth={2} />
                                </AreaChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div className="glass-panel p-6 rounded-xl">
                            <h3 className="text-lg font-bold mb-6 flex items-center gap-2">
                                <BarChart2 className="text-accent" size={20} />
                                Performance Metrics
                            </h3>
                            <div className="h-64">
                                <ResponsiveContainer width="100%" height="100%">
                                    <RadarChart cx="50%" cy="50%" outerRadius="80%" data={performanceComparison}>
                                        <PolarGrid stroke="#444" />
                                        <PolarAngleAxis dataKey="subject" stroke="#888" fontSize={10} />
                                        <PolarRadiusAxis angle={30} domain={[0, 150]} stroke="#444" tick={false} axisLine={false} />
                                        <Radar name="Take 04" dataKey="A" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.4} />
                                        <Radar name="Take 05" dataKey="B" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.4} />
                                        <Tooltip contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #333' }} />
                                    </RadarChart>
                                </ResponsiveContainer>
                            </div>
                        </div>

                        <div className="glass-panel p-6 rounded-xl flex flex-col">
                            <h3 className="text-lg font-bold mb-6 flex items-center gap-2">
                                <Brain className="text-success" size={20} />
                                AI Observations
                            </h3>
                            <div className="flex-1 space-y-4 overflow-y-auto pr-2">
                                <ObservationItem
                                    title="Micro-expression detected"
                                    desc="Frame 452: Brief facial muscle contraction indicating suppression of anger."
                                    impact="High"
                                />
                                <ObservationItem
                                    title="Eye-line Drift"
                                    desc="Frame 1205: Subject eyes drifted 2.4° off-axis from target character Marcus."
                                    impact="Moderate"
                                />
                                <ObservationItem
                                    title="Peak Performance Sync"
                                    desc="Emotional peak (Sadness: 85) aligns perfectly with Script Index L3beat."
                                    impact="Positive"
                                />
                            </div>
                        </div>
                    </div>
                </div>

                <div className="space-y-6">
                    <div className="glass-panel p-6 rounded-xl bg-gradient-to-br from-accent/10 to-transparent">
                        <div className="flex items-center gap-2 text-accent mb-4">
                            <Activity size={20} />
                            <span className="font-bold text-sm uppercase">Voice Sentiment</span>
                        </div>
                        <div className="space-y-4">
                            <div className="flex justify-between items-end mb-1">
                                <span className="text-xs text-editor-muted">Pitch Variation</span>
                                <span className="text-xs font-mono text-white">High (42Hz)</span>
                            </div>
                            <div className="h-1 bg-editor-track rounded-full overflow-hidden">
                                <div className="h-full bg-accent w-[85%]" />
                            </div>

                            <div className="flex justify-between items-end mb-1">
                                <span className="text-xs text-editor-muted">Pause Rhythm</span>
                                <span className="text-xs font-mono text-white">Consistent</span>
                            </div>
                            <div className="h-1 bg-editor-track rounded-full overflow-hidden">
                                <div className="h-full bg-accent w-[92%]" />
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

const LegendItem = ({ color, label }: { color: string, label: string }) => (
    <div className="flex items-center gap-2">
        <div className="w-3 h-1 rounded-full" style={{ backgroundColor: color }} />
        <span className="text-[10px] text-editor-muted font-bold uppercase tracking-tighter">{label}</span>
    </div>
);

const ObservationItem = ({ title, desc, impact }: { title: string, desc: string, impact: string }) => (
    <div className="p-3 bg-editor-track/40 rounded border-l-2 border-editor-border hover:border-primary transition-colors cursor-default">
        <div className="flex justify-between items-center mb-1">
            <span className="text-xs font-bold text-white">{title}</span>
            <span className={cn(
                "text-[9px] px-1.5 py-0.5 rounded font-bold uppercase",
                impact === 'High' ? "bg-danger/20 text-danger" :
                    impact === 'Positive' ? "bg-success/20 text-success" :
                        "bg-warning/20 text-warning"
            )}>{impact}</span>
        </div>
        <div className="text-[11px] text-editor-muted leading-relaxed">{desc}</div>
    </div>
);
