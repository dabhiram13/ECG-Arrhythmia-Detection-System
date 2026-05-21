import { Cpu, Zap, Activity } from 'lucide-react';

export function MetricsPanel() {
  return (
    <div className="h-40 glass rounded-2xl p-6 flex flex-col">
      <h2 className="text-xs uppercase tracking-widest text-white/60 font-semibold mb-3">Feature Engineering Matrix (13 Dimensions)</h2>
      <div className="flex-1 flex items-end justify-between space-x-1 px-4">
        <div className="w-full bg-emerald-500/20 rounded-t h-full flex flex-col justify-end items-center pb-2"><span className="text-[8px] text-emerald-400 font-bold mb-1">82%</span><div className="w-2 h-[82%] bg-emerald-500 rounded-full"></div></div>
        <div className="w-full bg-emerald-500/20 rounded-t h-full flex flex-col justify-end items-center pb-2"><span className="text-[8px] text-emerald-400 font-bold mb-1">64%</span><div className="w-2 h-[64%] bg-emerald-500 rounded-full"></div></div>
        <div className="w-full bg-white/10 rounded-t h-full flex flex-col justify-end items-center pb-2"><span className="text-[8px] text-white/40 font-bold mb-1">45%</span><div className="w-2 h-[45%] bg-white/20 rounded-full"></div></div>
        <div className="w-full bg-white/10 rounded-t h-full flex flex-col justify-end items-center pb-2"><span className="text-[8px] text-white/40 font-bold mb-1">91%</span><div className="w-2 h-[91%] bg-white/20 rounded-full"></div></div>
        <div className="w-full bg-white/10 rounded-t h-full flex flex-col justify-end items-center pb-2"><span className="text-[8px] text-white/40 font-bold mb-1">38%</span><div className="w-2 h-[38%] bg-white/20 rounded-full"></div></div>
        <div className="w-full bg-white/10 rounded-t h-full flex flex-col justify-end items-center pb-2"><span className="text-[8px] text-white/40 font-bold mb-1">72%</span><div className="w-2 h-[72%] bg-white/20 rounded-full"></div></div>
        <div className="w-full bg-emerald-500/20 rounded-t h-full flex flex-col justify-end items-center pb-2"><span className="text-[8px] text-emerald-400 font-bold mb-1">95%</span><div className="w-2 h-[95%] bg-emerald-500 rounded-full"></div></div>
        <div className="w-full bg-white/10 rounded-t h-full flex flex-col justify-end items-center pb-2"><span className="text-[8px] text-white/40 font-bold mb-1">50%</span><div className="w-2 h-[50%] bg-white/20 rounded-full"></div></div>
        <div className="w-full bg-white/10 rounded-t h-full flex flex-col justify-end items-center pb-2"><span className="text-[8px] text-white/40 font-bold mb-1">29%</span><div className="w-2 h-[29%] bg-white/20 rounded-full"></div></div>
        <div className="w-full bg-white/10 rounded-t h-full flex flex-col justify-end items-center pb-2"><span className="text-[8px] text-white/40 font-bold mb-1">68%</span><div className="w-2 h-[68%] bg-white/20 rounded-full"></div></div>
      </div>
      <div className="flex justify-between mt-2 text-[9px] text-white/30 uppercase tracking-tighter">
        <span>Time-Domain Characteristics</span>
        <span className="hidden sm:inline">Frequency Analysis</span>
        <span>Morphological Features</span>
      </div>
    </div>
  );
}
