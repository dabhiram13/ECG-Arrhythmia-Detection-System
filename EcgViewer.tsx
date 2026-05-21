import { HeartPulse, Activity, BrainCircuit } from 'lucide-react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, ReferenceLine } from 'recharts';
import { EcgPoint } from '../utils/ecgData';

export function EcgViewer({ data, classification }: { data: EcgPoint[], classification: string | null }) {
  // Find R-peaks to highlight
  const peaks = data.filter(d => d.isPeak);
  
  return (
    <div className="glass rounded-2xl p-6 h-full flex flex-col">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xs uppercase tracking-widest text-white/60 font-semibold">Active Signal Trace: Lead II</h2>
        <div className="flex items-center space-x-4">
          {classification ? (
            <span className="flex items-center space-x-2">
              <span className={`w-2 h-2 rounded-full ${classification === 'NORMAL' ? 'bg-emerald-500' : 'bg-rose-500'} animate-pulse`}></span>
              <span className={`text-[10px] uppercase font-bold ${classification === 'NORMAL' ? 'text-emerald-500' : 'text-rose-500'}`}>{classification === 'NORMAL' ? 'Normal Beat Detected' : 'Arrhythmia Detected'}</span>
            </span>
          ) : (
            <span className="flex items-center space-x-2 opacity-50">
              <span className="w-2 h-2 rounded-full bg-white/50 animate-pulse"></span>
              <span className="text-[10px] uppercase font-bold text-white/50">Computing...</span>
            </span>
          )}
          <span className="text-xs font-mono text-white/40">68 BPM</span>
        </div>
      </div>
      
      <div className="flex-1 w-full bg-black/40 rounded-lg overflow-hidden border border-white/5 relative min-h-[160px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 15, right: 0, left: -20, bottom: 15 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" vertical={true} horizontal={true} />
            <XAxis dataKey="time" type="number" domain={['dataMin', 'dataMax']} hide />
            <YAxis domain={[-2, 3]} hide />
            {peaks.map((p, i) => (
               <ReferenceLine key={i} x={p.time} stroke="#E53E3E" strokeOpacity={0.2} strokeDasharray="3 3" />
            ))}
            <Line 
              type="monotone" 
              dataKey="amplitude" 
              stroke="#10B981" 
              strokeWidth={1.5} 
              dot={false}
              isAnimationActive={false} 
            />
          </LineChart>
        </ResponsiveContainer>
        <div className="absolute inset-0 pointer-events-none bg-gradient-to-r from-black/20 via-transparent to-black/20"></div>
      </div>

      <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="p-3 bg-white/5 rounded-lg border border-white/5">
          <p className="text-[10px] uppercase text-white/40 mb-1">QRS Duration</p>
          <p className="text-lg font-light">84ms</p>
        </div>
        <div className="p-3 bg-white/5 rounded-lg border border-white/5">
          <p className="text-[10px] uppercase text-white/40 mb-1">PR Interval</p>
          <p className="text-lg font-light">162ms</p>
        </div>
        <div className="p-3 bg-white/5 rounded-lg border border-white/5">
          <p className="text-[10px] uppercase text-white/40 mb-1">QT/QTc</p>
          <p className="text-lg font-light">380/410ms</p>
        </div>
        <div className="p-3 bg-white/5 rounded-lg border border-white/5">
          <p className="text-[10px] uppercase text-white/40 mb-1">Sampling</p>
          <p className="text-lg font-light">360Hz</p>
        </div>
      </div>
    </div>
  );
}
