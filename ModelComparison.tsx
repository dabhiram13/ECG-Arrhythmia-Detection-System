import { Database, Network } from 'lucide-react';

export function ModelComparison() {
  return (
    <div className="flex-1 glass rounded-2xl p-6 flex flex-col">
      <h2 className="text-xs uppercase tracking-widest text-white/60 font-semibold mb-6">Architecture Comparison</h2>
      
      <div className="space-y-6 flex-1">
        <div className="space-y-2">
          <div className="flex justify-between items-center text-xs">
            <span className="font-semibold text-emerald-50">1D CNN (PyTorch)</span>
            <span className="font-mono text-emerald-400">98.2% Acc.</span>
          </div>
          <div className="h-1.5 w-full bg-white/5 rounded-full">
            <div className="h-full w-[98%] bg-emerald-500 rounded-full shadow-lg shadow-emerald-500/20"></div>
          </div>
        </div>

        <div className="space-y-2">
          <div className="flex justify-between items-center text-xs text-white/60">
            <span>LSTM Architectures</span>
            <span className="font-mono">96.5% Acc.</span>
          </div>
          <div className="h-1.5 w-full bg-white/5 rounded-full">
            <div className="h-full w-[96%] bg-emerald-500/60 rounded-full"></div>
          </div>
        </div>

        <div className="space-y-2">
          <div className="flex justify-between items-center text-xs text-white/60">
            <span>Random Forest</span>
            <span className="font-mono">92.1% Acc.</span>
          </div>
          <div className="h-1.5 w-full bg-white/5 rounded-full">
            <div className="h-full w-[92%] bg-white/20 rounded-full"></div>
          </div>
        </div>

        <div className="space-y-2">
          <div className="flex justify-between items-center text-xs text-white/60">
            <span>Linear SVM</span>
            <span className="font-mono">89.4% Acc.</span>
          </div>
          <div className="h-1.5 w-full bg-white/5 rounded-full">
            <div className="h-full w-[89%] bg-white/10 rounded-full"></div>
          </div>
        </div>
      </div>

      <div className="mt-8 border-t border-white/10 pt-6">
        <h3 className="text-[10px] uppercase tracking-widest text-white/40 font-bold mb-4">AAMI Categories Classification</h3>
        <div className="grid grid-cols-5 gap-2">
          <div className="flex flex-col items-center p-2 rounded bg-emerald-500/10 border border-emerald-500/20">
            <span className="text-xs font-bold text-emerald-400">N</span>
            <span className="text-[8px] text-white/30 hidden sm:block mt-1">Normal</span>
          </div>
          <div className="flex flex-col items-center p-2 rounded bg-white/5 border border-white/10 opacity-50">
            <span className="text-xs font-bold text-white">S</span>
            <span className="text-[8px] text-white/30 hidden sm:block mt-1">Supra</span>
          </div>
          <div className="flex flex-col items-center p-2 rounded bg-white/5 border border-white/10 opacity-50">
            <span className="text-xs font-bold text-white">V</span>
            <span className="text-[8px] text-white/30 hidden sm:block mt-1">Vent</span>
          </div>
          <div className="flex flex-col items-center p-2 rounded bg-white/5 border border-white/10 opacity-50">
            <span className="text-xs font-bold text-white">F</span>
            <span className="text-[8px] text-white/30 hidden sm:block mt-1">Fusion</span>
          </div>
          <div className="flex flex-col items-center p-2 rounded bg-white/5 border border-white/10 opacity-50">
            <span className="text-xs font-bold text-white">Q</span>
            <span className="text-[8px] text-white/30 hidden sm:block mt-1">Unk</span>
          </div>
        </div>
      </div>
    </div>
  );
}
