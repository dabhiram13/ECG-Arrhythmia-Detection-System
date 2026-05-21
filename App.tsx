import { useState, useEffect } from 'react';
import { generateSyntheticEcg, EcgPoint } from './utils/ecgData';
import { EcgViewer } from './components/EcgViewer';
import { MetricsPanel } from './components/MetricsPanel';
import { ModelComparison } from './components/ModelComparison';
import { Play, RotateCw, Activity, Code2, Server } from 'lucide-react';

export default function App() {
  const [isSimulating, setIsSimulating] = useState(false);
  const [signalType, setSignalType] = useState<'normal' | 'arrhythmia'>('normal');
  const [data, setData] = useState<EcgPoint[]>([]);
  
  useEffect(() => {
    // Generate initial data
    setData(generateSyntheticEcg('normal'));
  }, []);

  const runSimulation = () => {
    setIsSimulating(true);
    // Simulate pipeline latency
    setTimeout(() => {
      const isArrhythmia = Math.random() > 0.5;
      setSignalType(isArrhythmia ? 'arrhythmia' : 'normal');
      setData(generateSyntheticEcg(isArrhythmia ? 'arrhythmia' : 'normal'));
      setIsSimulating(false);
    }, 1200);
  };

  return (
    <div className="min-h-screen w-full flex flex-col p-4 md:p-8 space-y-6">
      {/* Header */}
      <header className="flex justify-between items-end border-b border-white/10 pb-6 shrink-0">
        <div>
          <h1 className="text-3xl md:text-4xl font-light tracking-tight">
            ECG <span className="serif text-emerald-400">Arrhythmia</span> Detection
          </h1>
          <p className="text-xs uppercase tracking-widest text-white/40 mt-1 font-semibold">Clinical ML Pipeline | Version 2025.05.B</p>
        </div>
        <div className="text-right">
          <p className="text-xs text-white/30 uppercase tracking-widest">Pipeline Status</p>
          <p className={`${isSimulating ? 'text-indigo-400 animate-pulse' : 'text-emerald-500'} font-mono text-sm`}>
            {isSimulating ? 'PROCESSING_BATCH // INFERENCE' : 'SYSTEM_STABLE // OPTIMIZED'}
          </p>
        </div>
      </header>

      {/* Controls Row (keeping functionality) */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 glass p-4 rounded-xl shrink-0">
        <div className="space-y-1">
          <h2 className="text-xs font-semibold uppercase tracking-widest text-white/60">Inference Pipeline Sandbox</h2>
          <p className="text-xs text-white/40 font-mono">Injects synthetic dataset (n=1000) batch into 1D CNN + LSTM architecture.</p>
        </div>
        <button 
          onClick={runSimulation}
          disabled={isSimulating}
          className="flex items-center justify-center gap-2 bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-400 border border-emerald-500/30 disabled:opacity-50 disabled:cursor-not-allowed px-5 py-2.5 rounded-lg font-medium transition-all accent-glow text-sm uppercase tracking-wider"
        >
          {isSimulating ? (
            <RotateCw className="w-4 h-4 animate-spin" />
          ) : (
            <Play className="w-4 h-4" />
          )}
          {isSimulating ? 'Processing Batch...' : 'Run Analysis Pipeline'}
        </button>
      </div>

      {/* Main Content */}
      <main className="flex-1 grid grid-cols-1 md:grid-cols-12 gap-6 min-h-0">
        
        {/* Left: Signal Analysis */}
        <div className="md:col-span-8 flex flex-col space-y-6">
          <div className="relative flex-1 min-h-[300px]">
            {isSimulating && (
              <div className="absolute inset-0 z-20 bg-[#050505]/60 backdrop-blur-sm rounded-2xl flex items-center justify-center border border-white/5">
                <div className="flex flex-col items-center gap-4">
                  <RotateCw className="w-8 h-8 text-emerald-500 animate-spin" />
                  <div className="font-mono text-sm text-emerald-300 tracking-widest uppercase animate-pulse">Running forward pass...</div>
                </div>
              </div>
            )}
            <EcgViewer 
              data={data} 
              classification={isSimulating ? null : (signalType === 'normal' ? 'NORMAL' : 'PREMATURE VENTRICULAR (V)')} 
            />
          </div>

          <MetricsPanel />
        </div>

        {/* Right: Model Benchmarking */}
        <div className="md:col-span-4 flex flex-col space-y-6">
          <ModelComparison />
          
          <div className="p-6 rounded-2xl bg-emerald-500 text-black flex flex-col justify-between h-40 shrink-0">
            <h3 className="text-xs font-bold uppercase tracking-wider">Upcoming Benchmarking</h3>
            <p className="serif text-xl md:text-2xl leading-snug font-medium">
              Validation on MIT-BIH Arrhythmia Database
            </p>
            <div className="flex justify-between items-center text-black/80">
              <span className="text-[10px] font-bold uppercase">PhysioNet Standards</span>
              <Server className="w-5 h-5 flex-shrink-0" />
            </div>
          </div>
        </div>
        
      </main>
      
      {/* Footer metadata */}
      <footer className="flex justify-between items-center text-[10px] tracking-widest text-white/30 font-semibold shrink-0 pt-4 border-t border-white/5">
        <div className="flex space-x-4 md:space-x-6">
          <span>NUMPY</span>
          <span>PANDAS</span>
          <span>SCIKIT-LEARN</span>
          <span>PYTORCH</span>
        </div>
        <div className="text-right">DESIGNED BY RESEARCH // 1,000 SAMPLES SYNTHETIC DATASET V1</div>
      </footer>
    </div>
  );
}
