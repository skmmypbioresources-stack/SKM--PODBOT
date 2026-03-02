import React, { useState, useRef, useEffect, useCallback } from 'react';
import { GoogleGenAI, Modality, ThinkingLevel } from "@google/genai";
import { Mic, Square, Play, Loader2, Volume2, Circle } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';

// PCM 16kHz to Float32 conversion
function pcmToFloat32(base64Data: string): Float32Array {
  const binaryString = window.atob(base64Data);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  const int16Array = new Int16Array(bytes.buffer, 0, len / 2);
  const float32Array = new Float32Array(int16Array.length);
  for (let i = 0; i < int16Array.length; i++) {
    float32Array[i] = int16Array[i] / 32768.0;
  }
  return float32Array;
}

// Float32 to Int16 PCM conversion (for sending to Gemini)
function float32ToInt16(float32Array: Float32Array): Int16Array {
  const int16Array = new Int16Array(float32Array.length);
  for (let i = 0; i < float32Array.length; i++) {
    const s = Math.max(-1, Math.min(1, float32Array[i]));
    int16Array[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
  }
  return int16Array;
}

// Helper to convert Int16Array to Base64
function int16ToBase64(int16Array: Int16Array): string {
  const uint8Array = new Uint8Array(int16Array.buffer, int16Array.byteOffset, int16Array.byteLength);
  let binary = '';
  const chunk = 8192;
  for (let i = 0; i < uint8Array.length; i += chunk) {
    binary += String.fromCharCode.apply(null, Array.from(uint8Array.slice(i, i + chunk)));
  }
  return window.btoa(binary);
}

type SessionStatus = 'idle' | 'connecting' | 'listening' | 'speaking' | 'ending';

type TranscriptItem = {
  role: 'user' | 'ai';
  text: string;
};

const BIOLOGY_TOPICS = [
  "Characteristics and classification of living organisms",
  "Organisation of the organism",
  "Movement into and out of cells",
  "Biological molecules",
  "Enzymes",
  "Plant nutrition",
  "Human nutrition",
  "Transport in plants",
  "Transport in animals",
  "Diseases and immunity",
  "Gas exchange in humans",
  "Respiration",
  "Excretion in humans",
  "Coordination and response",
  "Drugs",
  "Reproduction",
  "Inheritance",
  "Variation and selection",
  "Organisms and their environment",
  "Human influences on ecosystems",
  "Biotechnology and genetic modification"
];

export default function App() {
  const [userEmail, setUserEmail] = useState<string>("");
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [isAdminView, setIsAdminView] = useState(false);
  const [adminPassword, setAdminPassword] = useState("");
  const [adminStats, setAdminStats] = useState<{ total: number, users: any[] } | null>(null);
  const [status, setStatus] = useState<SessionStatus>('idle');
  const [recordingUrl, setRecordingUrl] = useState<string | null>(null);
  const [recordingBlob, setRecordingBlob] = useState<Blob | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [transcriptItems, setTranscriptItems] = useState<TranscriptItem[]>([]);
  const [topic, setTopic] = useState<string>("");
  const [isSettingTopic, setIsSettingTopic] = useState(true);
  const [logs, setLogs] = useState<string[]>([]);
  const [micLevel, setMicLevel] = useState(0);
  const [aiLevel, setAiLevel] = useState(0);
  const [isAiThinking, setIsAiThinking] = useState(false);
  const [textInput, setTextInput] = useState("");
  const [currentTranscription, setCurrentTranscription] = useState("");
  const [bytesSent, setBytesSent] = useState(0);
  const statusRef = useRef<SessionStatus>('idle');

  const addLog = (msg: string) => {
    console.log(msg);
    setLogs(prev => [msg, ...prev].slice(0, 5));
  };

  const setStatusWithRef = (s: SessionStatus) => {
    statusRef.current = s;
    setStatus(s);
  };

  useEffect(() => {
    const apiKey = process.env.GEMINI_API_KEY as string;
    if (!apiKey) {
      addLog("WARNING: GEMINI_API_KEY is not set in environment.");
    } else {
      addLog("GEMINI_API_KEY is available.");
    }
    return () => {
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
    };
  }, []);

  // Refs for audio handling
  const audioContextRef = useRef<AudioContext | null>(null);
  const micStreamRef = useRef<MediaStream | null>(null);
  const mixedDestinationRef = useRef<MediaStreamAudioDestinationNode | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);
  const aiGainRef = useRef<GainNode | null>(null);
  const micGainRef = useRef<GainNode | null>(null);
  const sessionRef = useRef<any>(null);
  const audioQueueRef = useRef<Float32Array[]>([]);
  const isPlayingRef = useRef(false);
  const nextStartTimeRef = useRef<number>(0);
  const aiCompressorRef = useRef<DynamicsCompressorNode | null>(null);

  const transcriptEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (transcriptEndRef.current) {
      transcriptEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [transcriptItems]);

  // Initialize Audio Context and Nodes
  const initAudio = async () => {
    if (!audioContextRef.current) {
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
    }
    const ctx = audioContextRef.current;
    if (ctx.state === 'suspended') await ctx.resume();

    // Mixed destination for recording
    mixedDestinationRef.current = ctx.createMediaStreamDestination();

    // AI Audio output nodes
    const aiCompressor = ctx.createDynamicsCompressor();
    aiCompressor.threshold.setValueAtTime(-24, ctx.currentTime);
    aiCompressor.knee.setValueAtTime(30, ctx.currentTime);
    aiCompressor.ratio.setValueAtTime(12, ctx.currentTime);
    aiCompressor.attack.setValueAtTime(0.003, ctx.currentTime);
    aiCompressor.release.setValueAtTime(0.25, ctx.currentTime);
    aiCompressorRef.current = aiCompressor;

    aiGainRef.current = ctx.createGain();
    aiGainRef.current.gain.value = 1.5; // Slightly lower to avoid clipping, compressor will handle clarity
    addLog(`AI Gain set to: ${aiGainRef.current.gain.value}`);
    
    aiGainRef.current.connect(aiCompressor);
    aiCompressor.connect(ctx.destination);
    aiGainRef.current.connect(mixedDestinationRef.current);

    // Mic input nodes
    const micStream = await navigator.mediaDevices.getUserMedia({ 
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        sampleRate: 24000
      } 
    });
    micStreamRef.current = micStream;
    const micSource = ctx.createMediaStreamSource(micStream);
    micGainRef.current = ctx.createGain();
    addLog(`Mic Gain initialized.`);
    micSource.connect(micGainRef.current);
    micGainRef.current.connect(mixedDestinationRef.current);

    // Setup mic processing for Gemini - Use smaller buffer for lower latency
    const processor = ctx.createScriptProcessor(1024, 1, 1);
    micGainRef.current.connect(processor);
    
    // Analyzer for Mic
    const micAnalyzer = ctx.createAnalyser();
    micAnalyzer.fftSize = 256;
    micGainRef.current.connect(micAnalyzer);
    const micDataArray = new Uint8Array(micAnalyzer.frequencyBinCount);

    // Analyzer for AI
    const aiAnalyzer = ctx.createAnalyser();
    aiAnalyzer.fftSize = 256;
    aiGainRef.current.connect(aiAnalyzer);
    const aiDataArray = new Uint8Array(aiAnalyzer.frequencyBinCount);

    const updateLevels = () => {
      if (statusRef.current === 'idle') return;
      micAnalyzer.getByteFrequencyData(micDataArray);
      aiAnalyzer.getByteFrequencyData(aiDataArray);
      
      const mSum = micDataArray.reduce((a, b) => a + b, 0);
      const aSum = aiDataArray.reduce((a, b) => a + b, 0);
      
      setMicLevel(mSum / micDataArray.length);
      setAiLevel(aSum / aiDataArray.length);
      
      requestAnimationFrame(updateLevels);
    };
    updateLevels();

    // Create a silent gain node for the processor to prevent audio feedback to speakers
    const silentGain = ctx.createGain();
    silentGain.gain.value = 0;
    processor.connect(silentGain);
    silentGain.connect(ctx.destination);

    let totalBytes = 0;
    let lastLogTime = 0;
    processor.onaudioprocess = (e) => {
      const currentStatus = statusRef.current;
      if (sessionRef.current && (currentStatus === 'listening' || currentStatus === 'speaking')) {
        const inputData = e.inputBuffer.getChannelData(0);
        
        // Send all audio to Gemini for maximum responsiveness
        try {
          const pcmData = float32ToInt16(inputData);
          const base64Data = int16ToBase64(pcmData);
          sessionRef.current.sendRealtimeInput({
            media: { data: base64Data, mimeType: 'audio/pcm;rate=24000' }
          });
          
          totalBytes += pcmData.byteLength;
          const now = Date.now();
          const hasActivity = inputData.some(v => Math.abs(v) > 0.05);
          if (hasActivity) {
            setIsAiThinking(true);
          }

          if (now - lastLogTime > 2000) {
            setBytesSent(totalBytes);
            addLog(`Mic active: ${Math.round(totalBytes / 1024)}KB uploaded`);
            lastLogTime = now;
          }
        } catch (err) {
          addLog(`Send error: ${err}`);
          console.error("Error sending audio:", err);
        }
      }
    };

    // Start recording the mixed stream AFTER all connections are made
    recordedChunksRef.current = [];
    const mediaRecorder = new MediaRecorder(mixedDestinationRef.current.stream, {
      mimeType: 'audio/webm;codecs=opus'
    });
    mediaRecorderRef.current = mediaRecorder;
    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) recordedChunksRef.current.push(e.data);
    };
    mediaRecorder.onstop = () => {
      const blob = new Blob(recordedChunksRef.current, { type: 'audio/webm' });
      const url = URL.createObjectURL(blob);
      setRecordingUrl(url);
      setRecordingBlob(blob);
    };
    mediaRecorder.start(1000); // Record in 1s chunks
    addLog("Recorder started.");
    setIsRecording(true);

    return { ctx, micStream };
  };

  // Playback queue for AI audio using scheduling to eliminate gaps
  const scheduleNextInQueue = useCallback(async () => {
    if (audioQueueRef.current.length === 0 || !audioContextRef.current || !aiGainRef.current) {
      if (audioQueueRef.current.length === 0 && status === 'speaking' && !isPlayingRef.current) {
        setStatus('listening');
      }
      return;
    }

    const ctx = audioContextRef.current;
    if (ctx.state === 'suspended') await ctx.resume();

    const data = audioQueueRef.current.shift()!;
    const buffer = ctx.createBuffer(1, data.length, 24000);
    buffer.getChannelData(0).set(data);
    
    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(aiGainRef.current);
    
    // Schedule playback
    const now = ctx.currentTime;
    let startTime = Math.max(now, nextStartTimeRef.current);
    
    // If we're too far behind, catch up
    if (startTime < now) startTime = now;
    
    source.start(startTime);
    isPlayingRef.current = true;
    
    const duration = buffer.duration;
    nextStartTimeRef.current = startTime + duration;

    source.onended = () => {
      // Check if this was the last chunk
      if (ctx.currentTime >= nextStartTimeRef.current - 0.05) {
        isPlayingRef.current = false;
      }
      scheduleNextInQueue();
    };
  }, [status]);

  useEffect(() => {
    if (audioQueueRef.current.length > 0) {
      scheduleNextInQueue();
    }
  }, [audioQueueRef.current.length, scheduleNextInQueue]);

  const testAudio = async () => {
    try {
      if (!audioContextRef.current) {
        audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
      }
      const ctx = audioContextRef.current;
      if (ctx.state === 'suspended') await ctx.resume();
      
      const oscillator = ctx.createOscillator();
      oscillator.type = 'sine';
      oscillator.frequency.setValueAtTime(440, ctx.currentTime);
      
      const gain = ctx.createGain();
      gain.gain.setValueAtTime(0.1, ctx.currentTime);
      
      oscillator.connect(gain);
      gain.connect(ctx.destination);
      
      oscillator.start();
      oscillator.stop(ctx.currentTime + 0.5);
      addLog("Test beep played.");
    } catch (err) {
      addLog(`Test Audio Failed: ${err}`);
    }
  };

  const copyLogs = () => {
    const logText = logs.join('\n');
    navigator.clipboard.writeText(logText);
    alert("Logs copied to clipboard!");
  };

  const startSession = async () => {
    if (!topic.trim()) {
      alert("Please enter a topic first.");
      return;
    }
    try {
      if (audioContextRef.current && audioContextRef.current.state === 'suspended') {
        await audioContextRef.current.resume();
      }
      addLog("Starting session...");
      setIsSettingTopic(false);
      setStatusWithRef('connecting');
      setTranscriptItems([]);
      const { ctx, micStream } = await initAudio();
      addLog("Audio initialized.");

      const apiKey = process.env.GEMINI_API_KEY as string;
      if (!apiKey) {
        addLog("Error: GEMINI_API_KEY missing.");
        throw new Error("API Key missing");
      }
      addLog(`API Key found: ${apiKey.substring(0, 4)}...${apiKey.substring(apiKey.length - 4)}`);

      const ai = new GoogleGenAI({ apiKey });
      const sessionPromise = ai.live.connect({
        model: "gemini-2.5-flash-native-audio-preview-09-2025",
        config: {
          thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
          responseModalities: [Modality.AUDIO],
          speechConfig: {
            voiceConfig: { prebuiltVoiceConfig: { voiceName: "Puck" } },
          },
          outputAudioTranscription: {},
          inputAudioTranscription: {},
          systemInstruction: `You are SKM, a world-class tutor and conversationalist with a professional male voice. Topic: "${topic}". 
          
          CRITICAL: You MUST start the conversation immediately by saying: 'Hi, I am SKM, your PodBot. How can I help you with ${topic} today?'. 
          
          1. This is a LIVE, ultra-fast voice conversation. 
          2. Speak with a natural, authoritative yet friendly male tone.
          3. Match the user's energy and tone where appropriate to create a seamless conversation.
          4. When you hear the user speak, respond IMMEDIATELY. No long pauses.
          5. Keep responses brief (under 10 seconds) to maintain a fast back-and-forth.
          6. If the user is silent for more than 3 seconds, ask a quick follow-up question about ${topic}.`,
        },
        callbacks: {
          onopen: () => {
            addLog("Connection opened.");
            setStatusWithRef('listening');
            sessionPromise.then((session) => {
              addLog("Sending initial nudge...");
              // Send 200ms of silence at 24kHz to ensure the model wakes up
              const silentData = window.btoa(String.fromCharCode(...new Uint8Array(4800).fill(0)));
              session.sendRealtimeInput({
                media: { data: silentData, mimeType: 'audio/pcm;rate=24000' }
              });
              addLog("Nudge sent.");
            });
          },
          onmessage: async (message) => {
            setIsAiThinking(false);
            // Comprehensive debug log
            const msgKeys = Object.keys(message);
            addLog(`Msg: ${msgKeys.join(', ')}`);
            
            if (message.serverContent) {
              const content = message.serverContent;
              const contentKeys = Object.keys(content);
              addLog(`Content: ${contentKeys.join(', ')}`);
              
              if (content.modelTurn) {
                const parts = content.modelTurn.parts || [];
                addLog(`AI Turn: ${parts.length} parts`);
                const textPart = parts.find(p => p.text);
                if (textPart) addLog(`AI Text: ${textPart.text}`);
              }
              if ((content as any).userTurn) {
                const parts = (content as any).userTurn.parts || [];
                addLog(`User Turn: ${parts.length} parts (Transcribed)`);
                const textPart = parts.find(p => p.text);
                if (textPart) addLog(`User Text: ${textPart.text}`);
              }
              if (content.interrupted) addLog("AI Interrupted.");
            }
            
            // Handle AI Turn
            const modelParts = message.serverContent?.modelTurn?.parts;
            if (modelParts) {
              for (const part of modelParts) {
                if (part.inlineData?.data) {
                  addLog(`AI Audio received. Queue: ${audioQueueRef.current.length + 1}`);
                  const base64Audio = part.inlineData.data;
                  const float32Data = pcmToFloat32(base64Audio);
                  audioQueueRef.current.push(float32Data);
                  setStatusWithRef('speaking');
                  if (!isPlayingRef.current) {
                    addLog("Starting playback.");
                    scheduleNextInQueue();
                  }
                }
                if (part.text) {
                  addLog("AI Text received.");
                  setTranscriptItems(prev => {
                    const last = prev[prev.length - 1];
                    if (last && last.role === 'ai') {
                      return [...prev.slice(0, -1), { ...last, text: last.text + " " + part.text }];
                    }
                    return [...prev, { role: 'ai', text: part.text }];
                  });
                }
              }
            }

            // Handle User Turn (Transcription)
            const userParts = (message.serverContent as any)?.userTurn?.parts;
            if (userParts) {
              for (const part of userParts) {
                if (part.text) {
                  addLog(`User Text: ${part.text}`);
                  setCurrentTranscription(part.text);
                  setTranscriptItems(prev => {
                    const last = prev[prev.length - 1];
                    if (last && last.role === 'user') {
                      return [...prev.slice(0, -1), { ...last, text: part.text }];
                    }
                    return [...prev, { role: 'user', text: part.text }];
                  });
                  // Clear current transcription after a short delay
                  setTimeout(() => setCurrentTranscription(""), 3000);
                }
              }
            }

            if (message.serverContent?.interrupted) {
              addLog("AI Interrupted.");
              audioQueueRef.current = [];
              isPlayingRef.current = false;
              setStatusWithRef('listening');
            }
          },
          onclose: (event) => {
            addLog(`Connection closed: ${event?.reason || 'No reason'}`);
            endSession();
          },
          onerror: (err) => {
            addLog(`Error: ${err?.message || JSON.stringify(err)}`);
            endSession();
          }
        }
      });

      const session = await sessionPromise;
      sessionRef.current = session;
      addLog("Session resolved and ready.");
    } catch (err) {
      addLog(`Failed: ${err}`);
      setStatusWithRef('idle');
    }
  };

  const endSession = () => {
    setStatusWithRef('ending');
    addLog("Ending session...");
    if (sessionRef.current) {
      sessionRef.current.close();
      sessionRef.current = null;
    }
    audioQueueRef.current = [];
    isPlayingRef.current = false;
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      addLog("Stopping recorder...");
      mediaRecorderRef.current.stop();
    }
    if (micStreamRef.current) {
      addLog("Cleaning up mic tracks.");
      micStreamRef.current.getTracks().forEach(track => track.stop());
      micStreamRef.current = null;
    }
    setIsRecording(false);
    setStatusWithRef('idle');
  };

  const playRecording = () => {
    if (recordingUrl) {
      addLog("Playing recording...");
      const audio = new Audio(recordingUrl);
      audio.play();
    }
  };

  const saveSession = () => {
    if (recordingBlob) {
      addLog("Saving session...");
      const url = URL.createObjectURL(recordingBlob);
      const a = document.createElement('a');
      a.style.display = 'none';
      a.href = url;
      const fileName = topic.trim() ? `${topic.replace(/\s+/g, '_')}_session.webm` : 'session_recording.webm';
      a.download = fileName;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      alert("Session saved successfully!");
    }
  };

  const resetSession = () => {
    addLog("Resetting session...");
    setRecordingUrl(null);
    setRecordingBlob(null);
    setTopic("");
    setIsSettingTopic(true);
    setTranscriptItems([]);
    setStatusWithRef('idle');
  };

  const sendTextMessage = () => {
    if (sessionRef.current && textInput.trim()) {
      addLog(`Sending text: ${textInput}`);
      sessionRef.current.sendRealtimeInput({
        text: textInput
      });
      setTranscriptItems(prev => [...prev, { role: 'user', text: textInput }]);
      setTextInput("");
      setIsAiThinking(true);
    }
  };

  const manualNudge = () => {
    if (sessionRef.current && audioContextRef.current) {
      addLog("Sending audio tone nudge...");
      // Create a 440Hz tone for 200ms at 24kHz
      const sampleRate = 24000;
      const duration = 0.2;
      const numSamples = sampleRate * duration;
      const samples = new Float32Array(numSamples);
      for (let i = 0; i < numSamples; i++) {
        samples[i] = Math.sin(2 * Math.PI * 440 * i / sampleRate) * 0.5;
      }
      const pcmData = float32ToInt16(samples);
      const base64Data = int16ToBase64(pcmData);
      
      sessionRef.current.sendRealtimeInput({
        media: { data: base64Data, mimeType: 'audio/pcm;rate=24000' }
      });
      setIsAiThinking(true);
    } else {
      addLog("No active session.");
    }
  };

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    if (userEmail && userEmail.includes('@')) {
      try {
        await fetch("/api/login", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email: userEmail })
        });
      } catch (err) {
        console.error("Failed to log login", err);
      }
      setIsLoggedIn(true);
      addLog(`User logged in: ${userEmail}`);
    }
  };

  const fetchAdminStats = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const res = await fetch("/api/admin/stats", {
        headers: { "x-admin-password": adminPassword }
      });
      if (res.ok) {
        const data = await res.json();
        setAdminStats(data);
      } else {
        alert("Invalid Admin Password");
      }
    } catch (err) {
      alert("Failed to fetch stats");
    }
  };

  return (
    <div className="min-h-screen bg-[#f5f5f4] flex flex-col items-center justify-center p-6 font-sans text-[#1c1917]">
      <AnimatePresence mode="wait">
        {isAdminView ? (
          <motion.div 
            key="admin"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            className="w-full max-w-2xl bg-white rounded-[32px] shadow-sm border border-black/5 p-10 flex flex-col space-y-8"
          >
            <div className="flex justify-between items-center">
              <h1 className="text-2xl font-black tracking-tight text-stone-900">Admin Dashboard</h1>
              <button onClick={() => { setIsAdminView(false); setAdminStats(null); setAdminPassword(""); }} className="text-xs font-bold text-stone-400 hover:text-stone-600 uppercase tracking-widest">Back</button>
            </div>

            {!adminStats ? (
              <form onSubmit={fetchAdminStats} className="space-y-4">
                <div className="space-y-2">
                  <label className="text-[10px] font-black uppercase tracking-[0.2em] text-stone-400 ml-1">Admin Password</label>
                  <input 
                    type="password" 
                    required
                    value={adminPassword}
                    onChange={(e) => setAdminPassword(e.target.value)}
                    placeholder="Enter admin password..."
                    className="w-full px-4 py-4 bg-stone-50 border border-stone-200 rounded-2xl focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 transition-all text-sm"
                  />
                </div>
                <button type="submit" className="w-full py-4 bg-stone-900 text-white rounded-2xl font-bold text-sm shadow-lg transition-all active:scale-[0.98]">View Stats</button>
              </form>
            ) : (
              <div className="space-y-6">
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-indigo-50 p-6 rounded-2xl border border-indigo-100">
                    <p className="text-[10px] font-black uppercase tracking-[0.2em] text-indigo-400 mb-1">Total Sessions</p>
                    <p className="text-4xl font-black text-indigo-600">{adminStats.total}</p>
                  </div>
                  <div className="bg-stone-50 p-6 rounded-2xl border border-stone-100">
                    <p className="text-[10px] font-black uppercase tracking-[0.2em] text-stone-400 mb-1">Status</p>
                    <p className="text-4xl font-black text-stone-600">Active</p>
                  </div>
                </div>

                <div className="space-y-2">
                  <p className="text-[10px] font-black uppercase tracking-[0.2em] text-stone-400 ml-1">Recent Logins</p>
                  <div className="max-h-64 overflow-y-auto border border-stone-100 rounded-2xl divide-y divide-stone-100">
                    {adminStats.users.map((u, i) => (
                      <div key={i} className="p-4 flex justify-between items-center bg-white hover:bg-stone-50 transition-colors">
                        <span className="text-sm font-medium text-stone-700">{u.email}</span>
                        <span className="text-[10px] font-mono text-stone-400">{new Date(u.timestamp).toLocaleString()}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </motion.div>
        ) : !isLoggedIn ? (
          <motion.div 
            key="login"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            className="w-full max-w-md bg-white rounded-[32px] shadow-sm border border-black/5 p-10 flex flex-col items-center space-y-8"
          >
            <div className="w-20 h-20 bg-indigo-600 rounded-2xl flex items-center justify-center shadow-lg shadow-indigo-200 rotate-3">
              <span className="text-white text-3xl font-black tracking-tighter">SKM</span>
            </div>
            <div className="text-center space-y-2">
              <h1 className="text-3xl font-black tracking-tight text-stone-900">Welcome</h1>
              <p className="text-sm text-stone-500 font-medium">Please enter your email to start the session</p>
            </div>
            <form onSubmit={handleLogin} className="w-full space-y-4">
              <div className="space-y-2">
                <label className="text-[10px] font-black uppercase tracking-[0.2em] text-stone-400 ml-1">Email Address</label>
                <input 
                  type="email" 
                  required
                  value={userEmail}
                  onChange={(e) => setUserEmail(e.target.value)}
                  placeholder="teacher@school.com"
                  className="w-full px-4 py-4 bg-stone-50 border border-stone-200 rounded-2xl focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 transition-all text-sm"
                />
              </div>
              <button 
                type="submit"
                className="w-full py-4 bg-indigo-600 hover:bg-indigo-700 text-white rounded-2xl font-bold text-sm shadow-lg shadow-indigo-200 transition-all active:scale-[0.98]"
              >
                Continue to Biology Prep
              </button>
            </form>
            <div className="flex flex-col items-center space-y-4 w-full">
              <p className="text-[10px] text-stone-400 text-center leading-relaxed">
                By continuing, you agree to use PodBot for educational purposes.<br/>
                Your session data helps us improve the learning experience.
              </p>
              <button 
                onClick={() => setIsAdminView(true)}
                className="text-[10px] font-black uppercase tracking-[0.2em] text-stone-300 hover:text-indigo-400 transition-colors"
              >
                Admin Access
              </button>
            </div>
          </motion.div>
        ) : (
          <motion.div 
            key="app"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="w-full max-w-md bg-white rounded-[32px] shadow-sm border border-black/5 p-8 flex flex-col items-center space-y-12"
          >
            <div className="flex flex-col items-center space-y-4">
              <div className="w-20 h-20 bg-indigo-600 rounded-2xl flex items-center justify-center shadow-lg shadow-indigo-200 rotate-3">
                <span className="text-white text-3xl font-black tracking-tighter">SKM</span>
              </div>
              <div className="text-center space-y-1">
                <h1 className="text-3xl font-black tracking-tight text-stone-900">PodBot</h1>
                <p className="text-[10px] text-stone-400 font-bold uppercase tracking-[0.2em]">Live AI Voice Session</p>
              </div>
            </div>

            {/* Topic Setup or Session Info */}
        {isSettingTopic ? (
          <div className="w-full space-y-6">
            <div className="space-y-2">
              <label className="text-[10px] font-black uppercase tracking-[0.2em] text-indigo-500 ml-1">Biology Board Prep</label>
              <select 
                onChange={(e) => setTopic(e.target.value)}
                className="w-full px-4 py-3 bg-stone-50 border border-stone-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 transition-all text-sm font-medium text-stone-700"
                value={BIOLOGY_TOPICS.includes(topic) ? topic : ""}
              >
                <option value="" disabled>Select a Biology Topic...</option>
                {BIOLOGY_TOPICS.map((t, i) => (
                  <option key={i} value={t}>{i + 1}. {t}</option>
                ))}
              </select>
            </div>

            <div className="relative flex items-center">
              <div className="flex-grow border-t border-stone-100"></div>
              <span className="flex-shrink mx-4 text-[9px] font-black text-stone-300 uppercase tracking-[0.3em]">OR</span>
              <div className="flex-grow border-t border-stone-100"></div>
            </div>

            <div className="space-y-2">
              <label className="text-[10px] font-black uppercase tracking-[0.2em] text-stone-400 ml-1">Custom Discussion</label>
              <input 
                type="text" 
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
                placeholder="Type any other topic..."
                className="w-full px-4 py-3 bg-stone-50 border border-stone-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 transition-all text-sm"
              />
            </div>
          </div>
        ) : (
          <div className="text-center space-y-2">
            <div className="flex items-center justify-center space-x-2">
              <p className="text-xs font-bold uppercase tracking-widest text-indigo-500">Current Topic</p>
              <div className={`w-2 h-2 rounded-full ${status === 'idle' ? 'bg-stone-300' : 'bg-emerald-500 animate-pulse'}`} />
            </div>
            <p className="text-lg font-medium text-stone-800">{topic}</p>
          </div>
        )}

        {/* System Dashboard */}
        {!isSettingTopic && (
          <div className="w-full bg-stone-100/50 rounded-xl p-3 border border-stone-200 grid grid-cols-2 gap-2">
            <div className="flex items-center space-x-2">
              <div className={`w-1.5 h-1.5 rounded-full ${process.env.GEMINI_API_KEY ? 'bg-emerald-500' : 'bg-red-500'}`} />
              <span className="text-[9px] font-bold uppercase text-stone-500">API Link</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className={`w-1.5 h-1.5 rounded-full ${micStreamRef.current ? 'bg-emerald-500' : 'bg-stone-300'}`} />
              <span className="text-[9px] font-bold uppercase text-stone-500">Mic Active</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className={`w-1.5 h-1.5 rounded-full ${sessionRef.current ? 'bg-emerald-500' : 'bg-stone-300'}`} />
              <span className="text-[9px] font-bold uppercase text-stone-500">AI Core</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className={`w-1.5 h-1.5 rounded-full ${audioContextRef.current?.state === 'running' ? 'bg-emerald-500' : 'bg-stone-300'}`} />
              <span className="text-[9px] font-bold uppercase text-stone-500">Audio Engine</span>
            </div>
            <div className="flex items-center space-x-2 col-span-2 border-t border-stone-200 pt-1 mt-1">
              <span className="text-[8px] font-bold uppercase text-stone-400">Data Uploaded:</span>
              <span className="text-[8px] font-mono text-stone-600">{Math.round(bytesSent / 1024)} KB</span>
            </div>
          </div>
        )}

        {/* Status Indicator */}
        <div className="flex flex-col items-center space-y-6">
          <div className="relative flex items-center justify-center">
            {/* Mic Visualizer Ring */}
            <motion.div 
              animate={{ scale: 1 + (micLevel / 100) }}
              className="absolute w-40 h-40 rounded-full border-2 border-emerald-500/20"
            />
            {/* AI Visualizer Ring */}
            <motion.div 
              animate={{ scale: 1 + (aiLevel / 100) }}
              className="absolute w-48 h-48 rounded-full border-2 border-indigo-500/10"
            />

            <div className={`w-32 h-32 rounded-full flex items-center justify-center bg-stone-50 border border-stone-100 z-10 relative shadow-inner`}>
              {status === 'idle' && <Circle className="w-12 h-12 text-stone-300" />}
              {status === 'connecting' && <Loader2 className="w-12 h-12 text-stone-400 animate-spin" />}
              {status === 'listening' && (
                <div className="relative">
                  <Mic className="w-12 h-12 text-emerald-600" />
                  {micLevel > 10 && (
                    <motion.div 
                      className="absolute -top-1 -right-1 w-3 h-3 bg-emerald-500 rounded-full"
                      animate={{ scale: [1, 1.5, 1] }}
                      transition={{ repeat: Infinity, duration: 0.5 }}
                    />
                  )}
                </div>
              )}
              {status === 'speaking' && <Volume2 className="w-12 h-12 text-indigo-600 animate-pulse" />}
            </div>
          </div>
          
          <div className="flex flex-col items-center space-y-1">
            <AnimatePresence mode="wait">
              <motion.span
                key={status}
                initial={{ opacity: 0, y: 5 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -5 }}
                className="text-sm font-bold text-stone-700 uppercase tracking-tighter"
              >
                {status === 'idle' && "System Ready"}
                {status === 'connecting' && "Establishing Link..."}
                {status === 'listening' && "AI is Listening"}
                {status === 'speaking' && "AI is Responding"}
                {status === 'ending' && "Finalizing Tape..."}
              </motion.span>
            </AnimatePresence>
            {isAiThinking && (
              <motion.span 
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="text-[10px] text-indigo-400 font-medium animate-pulse"
              >
                AI is processing...
              </motion.span>
            )}
          </div>
        </div>

        {/* Level Meters */}
        {!isSettingTopic && status !== 'idle' && (
          <div className="w-full grid grid-cols-2 gap-4 px-4">
            <div className="space-y-1">
              <div className="flex justify-between text-[8px] uppercase font-bold text-stone-400">
                <span>Mic Input</span>
                <span>{Math.round(micLevel)}%</span>
              </div>
              <div className="h-1 bg-stone-200 rounded-full overflow-hidden">
                <motion.div 
                  className="h-full bg-emerald-500"
                  animate={{ width: `${micLevel}%` }}
                />
              </div>
            </div>
            <div className="space-y-1">
              <div className="flex justify-between text-[8px] uppercase font-bold text-stone-400">
                <span>AI Output</span>
                <span>{Math.round(aiLevel)}%</span>
              </div>
              <div className="h-1 bg-stone-200 rounded-full overflow-hidden">
                <motion.div 
                  className="h-full bg-indigo-500"
                  animate={{ width: `${aiLevel}%` }}
                />
              </div>
            </div>
          </div>
        )}

        {/* Transcription Area */}
        <AnimatePresence>
          {(status === 'speaking' || status === 'listening') && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 10 }}
              className="w-full bg-stone-900 rounded-2xl p-6 shadow-lg border border-white/10 max-h-80 overflow-y-auto space-y-4"
            >
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-indigo-500 rounded-full animate-pulse" />
                  <span className="text-[10px] font-bold uppercase tracking-widest text-stone-400">Live Transcript & Chat</span>
                </div>
              </div>
              
              <div className="space-y-3">
                {currentTranscription && (
                  <div className="flex flex-col items-end">
                    <span className="text-[9px] font-bold uppercase tracking-tighter text-emerald-500 mb-1 animate-pulse">
                      Live Transcription...
                    </span>
                    <p className="text-sm py-2 px-3 rounded-xl max-w-[90%] bg-emerald-900/20 text-emerald-100 border border-emerald-500/20 rounded-tr-none italic">
                      "{currentTranscription}"
                    </p>
                  </div>
                )}
                {transcriptItems.length === 0 && !currentTranscription && (
                  <p className="text-stone-500 text-xs italic text-center py-4">Waiting for conversation to start...</p>
                )}
                {transcriptItems.map((item, idx) => (
                  <div key={idx} className={`flex flex-col ${item.role === 'user' ? 'items-end' : 'items-start'}`}>
                    <span className="text-[9px] font-bold uppercase tracking-tighter text-stone-500 mb-1">
                      {item.role === 'user' ? 'You' : 'PodBot'}
                    </span>
                    <p className={`text-sm py-2 px-3 rounded-xl max-w-[90%] ${
                      item.role === 'user' 
                        ? 'bg-stone-800 text-stone-200 rounded-tr-none' 
                        : 'bg-indigo-900/40 text-indigo-100 border border-indigo-500/20 rounded-tl-none'
                    }`}>
                      {item.text}
                    </p>
                  </div>
                ))}
                <div ref={transcriptEndRef} />
              </div>

              {/* Text Fallback Input */}
              <div className="pt-4 border-t border-white/5 flex space-x-2">
                <input 
                  type="text"
                  value={textInput}
                  onChange={(e) => setTextInput(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && sendTextMessage()}
                  placeholder="Type to PodBot..."
                  className="flex-1 bg-stone-800 border border-white/10 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500"
                />
                <button 
                  onClick={sendTextMessage}
                  className="p-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors"
                >
                  <Play className="w-3 h-3" />
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Controls */}
        <div className="w-full grid grid-cols-1 gap-4">
          {isSettingTopic ? (
            <button
              onClick={startSession}
              className="w-full py-4 bg-[#1c1917] text-white rounded-2xl font-medium flex items-center justify-center space-x-2 hover:bg-stone-800 transition-colors"
            >
              <Mic className="w-5 h-5" />
              <span>Start Session</span>
            </button>
          ) : status !== 'idle' ? (
            <button
              onClick={endSession}
              disabled={status === 'ending'}
              className="w-full py-4 bg-red-50 text-red-600 border border-red-100 rounded-2xl font-medium flex items-center justify-center space-x-2 hover:bg-red-100 transition-colors disabled:opacity-50"
            >
              <Square className="w-5 h-5 fill-current" />
              <span>End Session</span>
            </button>
          ) : (
            <div className="space-y-4">
              <button
                onClick={playRecording}
                disabled={!recordingUrl}
                className="w-full py-4 bg-white text-stone-700 border border-stone-200 rounded-2xl font-medium flex items-center justify-center space-x-2 hover:bg-stone-50 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
              >
                <Play className="w-5 h-5 fill-current" />
                <span>Play Podcast Recording</span>
              </button>
              
              <button
                onClick={saveSession}
                disabled={!recordingBlob}
                className="w-full py-4 bg-indigo-600 text-white rounded-2xl font-medium flex items-center justify-center space-x-2 hover:bg-indigo-700 transition-colors disabled:opacity-30"
              >
                <Circle className="w-5 h-5 fill-current" />
                <span>Save Session</span>
              </button>

              <button
                onClick={manualNudge}
                disabled={status === 'idle'}
                className="w-full py-2 text-indigo-400 text-[10px] font-medium hover:text-indigo-600 transition-colors border border-dashed border-indigo-200 rounded-xl disabled:opacity-30"
              >
                Manual AI Nudge
              </button>

              <button
                onClick={testAudio}
                className="w-full py-2 text-stone-400 text-[10px] font-medium hover:text-stone-600 transition-colors border border-dashed border-stone-200 rounded-xl"
              >
                Test Audio Output
              </button>

              <button
                onClick={resetSession}
                className="w-full py-2 text-stone-400 text-xs font-medium hover:text-stone-600 transition-colors"
              >
                Start New Session
              </button>
            </div>
          )}
        </div>

        {isRecording && (
          <div className="flex items-center space-x-2 text-xs font-semibold text-red-500 uppercase tracking-widest">
            <motion.div 
              animate={{ opacity: [1, 0, 1] }}
              transition={{ repeat: Infinity, duration: 1 }}
              className="w-2 h-2 bg-red-500 rounded-full"
            />
            <span>Recording Active</span>
          </div>
        )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
