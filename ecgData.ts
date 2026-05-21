// Generates highly simplified synthetic ECG signal data mimicking a 10-second strip
export type EcgPoint = {
  time: number;
  amplitude: number;
  isPeak: boolean;
};

// Generate a synthetic single heart beat (P, Q, R, S, T waves)
function generateBeat(startTime: number, isPremature = false, isPathological = false): { points: EcgPoint[], duration: number } {
  const points: EcgPoint[] = [];
  const baseRate = isPremature ? 0.6 : 1.0; 
  let t = startTime;

  // Baseline
  for(let i = 0; i < 5; i++) {
    points.push({ time: t, amplitude: (Math.random() - 0.5) * 0.1, isPeak: false });
    t += 0.02;
  }

  // P wave
  for(let i = 0; i < 10; i++) {
    const amp = Math.sin((i / 9) * Math.PI) * 0.15;
    points.push({ time: t, amplitude: amp + (Math.random() - 0.5) * 0.05, isPeak: false });
    t += 0.02;
  }

  // PR segment
  for(let i = 0; i < 6; i++) {
    points.push({ time: t, amplitude: (Math.random() - 0.5) * 0.05, isPeak: false });
    t += 0.02;
  }

  // QRS complex
  if (isPathological) {
    // Wide and strange QRS (Ventricular ectopic)
    points.push({ time: t, amplitude: -0.8 + (Math.random() * 0.1), isPeak: false }); t += 0.02;
    points.push({ time: t, amplitude: 1.5 + (Math.random() * 0.1), isPeak: true }); t += 0.02;
    points.push({ time: t, amplitude: 0.8 + (Math.random() * 0.1), isPeak: false }); t += 0.02;
    points.push({ time: t, amplitude: -1.0 + (Math.random() * 0.1), isPeak: false }); t += 0.02;
  } else {
    // Normal QRS
    points.push({ time: t, amplitude: -0.2, isPeak: false }); t += 0.02;
    points.push({ time: t, amplitude: 2.5, isPeak: true }); t += 0.02;
    points.push({ time: t, amplitude: -0.4, isPeak: false }); t += 0.02;
  }

  // ST segment
  for(let i = 0; i < 8; i++) {
    points.push({ time: t, amplitude: (Math.random() - 0.5) * 0.05, isPeak: false });
    t += 0.02;
  }

  // T wave
  for(let i = 0; i < 15; i++) {
    let amp = Math.sin((i / 14) * Math.PI) * (isPathological ? -0.3 : 0.35);
    points.push({ time: t, amplitude: amp + (Math.random() - 0.5) * 0.02, isPeak: false });
    t += 0.02;
  }

  // Baseline end
  for(let i = 0; i < Math.floor(15 * baseRate); i++) {
    points.push({ time: t, amplitude: (Math.random() - 0.5) * 0.05, isPeak: false });
    t += 0.02;
  }

  return { points, duration: t - startTime };
}

export function generateSyntheticEcg(type: 'normal' | 'arrhythmia'): EcgPoint[] {
  const dataset: EcgPoint[] = [];
  let currentTime = 0;
  
  const numBeats = 10;
  
  for (let i = 0; i < numBeats; i++) {
    let isPremature = false;
    let isPathological = false;
    
    if (type === 'arrhythmia') {
      // Inject PVCs or early beats
      if (i === 3 || i === 7) {
        isPremature = true;
        isPathological = true;
      }
    }
    
    const { points, duration } = generateBeat(currentTime, isPremature, isPathological);
    dataset.push(...points);
    currentTime += duration;
  }
  
  return dataset;
}
