import React, { useState, useRef, useEffect } from 'react';
import { supabase } from './supabaseClient';
import './App.css';

function App() {
  // TTS Control Panel State
  const [textInput, setTextInput] = useState('');
  const [selectedVoiceUrl, setSelectedVoiceUrl] = useState('af_heart'); 
  const [clonedVoices, setClonedVoices] = useState([]); 
  const [isGenerating, setIsGenerating] = useState(false); 
  const CHARACTER_LIMIT = 3000;

  // Personalized Array Mixer Sliders
  const [pitchModifier, setPitchModifier] = useState(1.0); 
  const [speedModifier, setSpeedModifier] = useState(1.0); 

  // Client-Side Web Audio API Caching States
  const [audioBuffer, setAudioBuffer] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);

  // UI Tab Control
  const [activeTab, setActiveTab] = useState('record');

  // Voice Cloning State
  const [isRecording, setIsRecording] = useState(false);
  const [isUploading, setIsUploading] = useState(false); 
  const [audioUrl, setAudioUrl] = useState(null);
  const [audioBlob, setAudioBlob] = useState(null);
  const [recordingDuration, setRecordingDuration] = useState(0);
  const [uploadedFileName, setUploadedFileName] = useState('');
  const [customVoiceName, setCustomVoiceName] = useState('');

  // Refs for Web Audio API node routing graphs
  const audioCtxRef = useRef(null);
  const sourceNodeRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const timerRef = useRef(null);
  const fileInputRef = useRef(null);

  // Derived: is the currently selected voice a cloned sample (URL) vs a preset?
  // The backend's own resolve_voice_embedding() uses the same http/https check —
  // kept in sync with that logic so the frontend and backend agree on what
  // counts as a "clone request."
  const isCloneVoiceSelected = selectedVoiceUrl.startsWith('http');
  const MIN_CLONE_TEXT_CHARS = 80;
  const cloneTextTooShort =
    isCloneVoiceSelected &&
    textInput.trim().length > 0 &&
    textInput.trim().length < MIN_CLONE_TEXT_CHARS;

  // Fetch saved voices on component initialization
  useEffect(() => {
    fetchSavedVoices();
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      if (audioCtxRef.current) audioCtxRef.current.close();
    };
  }, []);

  // INSTANT TUNING FX LOOP: Updates playback rate instantly if audio is currently playing
  useEffect(() => {
    if (sourceNodeRef.current && isPlaying) {
      // Web Audio API uses continuous compound multipliers for scaling sample playback rates
      // Multiplying Pitch scale factor by Speed scale factor shifts arrays smoothly on the fly
      const targetPlaybackRate = parseFloat(pitchModifier) * parseFloat(speedModifier);
      sourceNodeRef.current.playbackRate.setValueAtTime(targetPlaybackRate, audioCtxRef.current.currentTime);
    }
  }, [pitchModifier, speedModifier, isPlaying]);

  const fetchSavedVoices = async () => {
    try {
      const { data, error } = await supabase
        .from('cloned_voices')
        .select('*')
        .order('created_at', { ascending: false });

      if (error) throw error;
      setClonedVoices(data || []);
    } catch (err) {
      console.error('Error loading custom voice directory:', err.message);
    }
  };

  const handleTextChange = (e) => {
    const value = e.target.value;
    if (value.length <= CHARACTER_LIMIT) {
      setTextInput(value);
    }
  };

  // --- STEP 1: Fetch Base Audio Array from Server and Cache Locally ---
  const handleGenerateSpeech = async () => {
    if (!textInput.trim()) return alert('Please enter some text first!');

    setIsGenerating(true);
    stopClientAudio();
    setAudioBuffer(null);

    try {
      console.log('Fetching raw baseline wave parameters from FastAPI engine...');
      
      const response = await fetch('http://localhost:8000/api/tts/download', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: textInput.trim(),
          voice: selectedVoiceUrl,
          pitch_modifier: 1.0, // Always request raw clean arrays so client controls manipulation
          speed_modifier: 1.0
        })
      });

      if (!response.ok) {
        const errorText = await response.text();
        let message = errorText || 'Server side parsing error.';
        try {
          const parsed = JSON.parse(errorText);
          if (parsed.detail) message = parsed.detail;
        } catch {
          // errorText wasn't JSON (e.g. a plain-text 500 page) — use as-is
        }
        throw new Error(message);
      }

      const arrayBuffer = await response.arrayBuffer();
      
      // Initialize Web Audio context environment map if uninitialized
      if (!audioCtxRef.current) {
        audioCtxRef.current = new (window.AudioContext || window.webkitAudioContext)();
      }

      // Decode the raw WAV stream binary directly into an immutable local client cache buffer
      const decodedBuffer = await audioCtxRef.current.decodeAudioData(arrayBuffer);
      setAudioBuffer(decodedBuffer);
      
      console.log('Success: Core audio array matrices successfully cached inside client memory.');
      
      // Instantly start playing the newly cached audio
      playCachedAudio(decodedBuffer);

    } catch (err) {
      console.error('TTS Generation Pipeline Failure:', err);
      alert(`Synthesis failed: ${err.message}`);
    } finally {
      setIsGenerating(false);
    }
  };

  // --- STEP 2: Playback Routing Node Graph Management ---
  const playCachedAudio = (bufferToPlay = audioBuffer) => {
    if (!bufferToPlay || !audioCtxRef.current) return;

    stopClientAudio();

    // Re-verify that the pipeline runtime hasn't suspended to save system power boundaries
    if (audioCtxRef.current.state === 'suspended') {
      audioCtxRef.current.resume();
    }

    // Build standard Web Audio node path tree
    const source = audioCtxRef.current.createBufferSource();
    source.buffer = bufferToPlay;

    // Apply baseline slider modifiers explicitly to the execution thread node context
    const currentRate = parseFloat(pitchModifier) * parseFloat(speedModifier);
    source.playbackRate.setValueAtTime(currentRate, audioCtxRef.current.currentTime);

    source.connect(audioCtxRef.current.destination);
    sourceNodeRef.current = source;
    setIsPlaying(true);

    source.onended = () => {
      setIsPlaying(false);
    };

    source.start(0);
  };

const stopClientAudio = () => {
    if (sourceNodeRef.current) {
      try {
        sourceNodeRef.current.stop();
      } catch (e) {
        // Safe tracking cleanup if audio already ended naturally
      }
      sourceNodeRef.current.disconnect();
      sourceNodeRef.current = null; // <-- Change 'None' to 'null' here!
    }
    setIsPlaying(false);
  };

  const switchTab = (tabName) => {
    if (isRecording) stopRecording();
    setActiveTab(tabName);
    setAudioUrl(null);
    setAudioBlob(null);
    setUploadedFileName('');
    setRecordingDuration(0);
  };

  // --- Voice Recorder Functions ---
  const startRecording = async () => {
    audioChunksRef.current = [];
    setAudioUrl(null);
    setAudioBlob(null);
    setRecordingDuration(0);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const options = { mimeType: 'audio/webm' };
      if (!MediaRecorder.isTypeSupported('audio/webm')) {
        options.mimeType = 'audio/ogg';
      }

      const mediaRecorder = new MediaRecorder(stream, options);
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const recordedBlob = new Blob(audioChunksRef.current, { type: options.mimeType });
        const url = URL.createObjectURL(recordedBlob);
        setAudioUrl(url);
        setAudioBlob(recordedBlob);
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);

      timerRef.current = setInterval(() => {
        setRecordingDuration((prev) => prev + 1);
      }, 1000);

    } catch (err) {
      console.error('Error accessing microphone:', err);
      alert('Could not access microphone.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      if (timerRef.current) clearInterval(timerRef.current);
    }
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs < 10 ? '0' : ''}${secs}`;
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    if (!file.type.startsWith('audio/')) {
      alert('Please upload an audio file');
      return;
    }

    setUploadedFileName(file.name);
    setAudioBlob(file);
    const url = URL.createObjectURL(file);
    setAudioUrl(url);
  };

  // --- Voice Sync Logic ---
  const handleCreateClone = async () => {
    if (!audioBlob) return alert('Please record or upload a sample first!');
    if (!customVoiceName.trim()) return alert('Please provide a name/label for your custom voice clone!');
    
    setIsUploading(true);

    try {
      const userId = "00000000-0000-0000-0000-000000000000"; 
      const fileExtension = activeTab === 'upload' ? audioBlob.name.split('.').pop() : 'webm';
      const fileName = `voice_${Date.now()}.${fileExtension}`;
      const filePath = `samples/${fileName}`;

      console.log('Step A: Transporting asset to Supabase Storage...');
      const { error: uploadError } = await supabase.storage
        .from('voice-samples')
        .upload(filePath, audioBlob, { cacheControl: '3600', upsert: false });

      if (uploadError) throw uploadError;

      const { data: { publicUrl } } = supabase.storage
        .from('voice-samples')
        .getPublicUrl(filePath);

      console.log('Step B: Storage upload verified. Synced URL:', publicUrl);
      const { error: dbError } = await supabase
        .from('cloned_voices')
        .insert([
          {
            user_id: userId,
            voice_name: customVoiceName.trim(),
            sample_file_url: publicUrl,
            provider_voice_id: 'local_kokoro_preset' 
          }
        ]);

      if (dbError) throw dbError;

      alert(`"${customVoiceName}" added successfully to your custom voice profile registry!`);
      
      setAudioUrl(null);
      setAudioBlob(null);
      setUploadedFileName('');
      setCustomVoiceName('');
      fetchSavedVoices(); 

    } catch (err) {
      console.error('Critical Architecture Operation Aborted:', err);
      alert(`Pipeline error: ${err.message}`);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>CurrenX Studio</h1>
        <p>Advanced Text-to-Speech & Voice Cloning Control Panel</p>
      </header>

      <main className="dashboard-grid">
        {/* Left Panel: Text Input & Generation */}
        <section className="card text-section">
          <h2>1. Text to Speech</h2>
          
          <div className="voice-selector">
            <label htmlFor="voice-select">Choose Voice Profiles:</label>
            <select 
              id="voice-select" 
              value={selectedVoiceUrl} 
              onChange={(e) => setSelectedVoiceUrl(e.target.value)}
            >
              <optgroup label="Premium Native Profiles (High Inflection)">
                <option value="af_heart">Kokoro Engine (af_heart - Neutral Female)</option>
                <option value="af_bella">Bella (American Female - Nostalgic/Clear)</option>
                <option value="af_nicole">Nicole (American Female - Crisp/Dynamic)</option>
                <option value="am_michael">Michael (American Male - Classic Standard)</option>
                <option value="am_adam">Adam (American Male - Deep Voice/Narrator)</option>
                <option value="bf_emma">Emma (British Female - Crisp Accent)</option>
                <option value="bm_george">George (British Male - Smooth Accent)</option>
              </optgroup>

              <optgroup label="Custom Blended Models (Multi-Vector Mixes)">
                <option value="af_bella+af_nicole">Hybrid Mix A (Bella + Nicole Duo)</option>
                <option value="am_michael+am_adam">Hybrid Mix B (Michael + Adam Deep Blend)</option>
                <option value="af_bella+am_michael">Hybrid Mix C (Cross-Gender Accent Fusion)</option>
              </optgroup>

              <optgroup label="Premium Celebrity Library">
                <option value="celebrity_jolie">jolie (Native Neural Vector)</option>
                <option value="celebrity_Mayor">Mayor (Native Neural Vector)</option>
              </optgroup>

              <optgroup label="Your Cloned Voices">
                {clonedVoices.length === 0 ? (
                  <option value="none" disabled>No cloned voices saved yet...</option>
                ) : (
                  clonedVoices.map((voice) => (
                    <option key={voice.id} value={voice.sample_file_url}>
                      {voice.voice_name} (Recorded Sample)
                    </option>
                  ))
                )}
              </optgroup>
            </select>
          </div>

          <div className="textarea-container">
            <textarea
              placeholder="Type or paste your text here (up to 3,000 characters)..."
              value={textInput}
              onChange={handleTextChange}
              rows={6}
            />
            <div className="textarea-footer">
              <span className={`char-counter ${textInput.length >= CHARACTER_LIMIT ? 'limit-reached' : ''}`}>
                {textInput.length.toLocaleString()} / {CHARACTER_LIMIT.toLocaleString()} characters
              </span>
            </div>
            {cloneTextTooShort && (
              <p style={{ color: '#fbbf24', fontSize: '0.8rem', margin: '6px 0 0 0' }}>
                ⚠️ Cloned voices need at least {MIN_CLONE_TEXT_CHARS} characters of text to work reliably — you have {textInput.trim().length}.
              </p>
            )}
          </div>

          {/* Interactive Live Voice Mixer Console */}
          <div className="voice-mixer-console" style={{ margin: '20px 0', padding: '15px', background: '#1e293b', borderRadius: '8px' }}>
            <h3 style={{ fontSize: '0.95rem', margin: '0 0 12px 0', color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              🎛️ Live Real-Time Fine-Tuning Console
            </h3>
            
            <div className="slider-group" style={{ marginBottom: '12px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', color: '#cbd5e1', fontSize: '0.9rem', marginBottom: '4px' }}>
                <label>Vocal Pitch Frequency Shift</label>
                <span style={{ fontWeight: 'bold', color: '#38bdf8' }}>{pitchModifier}x</span>
              </div>
              <input 
                type="range" 
                min="0.50" 
                max="1.50" 
                step="0.02" 
                value={pitchModifier}
                onChange={(e) => setPitchModifier(e.target.value)}
                style={{ width: '100%', accentColor: '#38bdf8' }}
              />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: '#64748b', marginTop: '2px' }}>
                <span>Deep Monster</span>
                <span>Original Base</span>
                <span>High Chipmunk</span>
              </div>
            </div>

            <div className="slider-group">
              <div style={{ display: 'flex', justifyContent: 'space-between', color: '#cbd5e1', fontSize: '0.9rem', marginBottom: '4px' }}>
                <label>Vocal Speed Multiplier</label>
                <span style={{ fontWeight: 'bold', color: '#a78bfa' }}>{speedModifier}x</span>
              </div>
              <input 
                type="range" 
                min="0.50" 
                max="2.00" 
                step="0.05" 
                value={speedModifier}
                onChange={(e) => setSpeedModifier(e.target.value)}
                style={{ width: '100%', accentColor: '#a78bfa' }}
              />
            </div>
          </div>

          <div style={{ display: 'flex', gap: '10px' }}>
            <button 
              className="btn btn-primary" 
              onClick={handleGenerateSpeech} 
              disabled={!textInput.trim() || isGenerating || cloneTextTooShort}
              style={{ flex: 1 }}
            >
              {isGenerating ? 'Synthesizing Stream Array...' : 'Generate Audio'}
            </button>

            {audioBuffer && (
              <button 
                className="btn"
                onClick={isPlaying ? stopClientAudio : () => playCachedAudio()}
                style={{ background: isPlaying ? '#ef4444' : '#10b981', color: 'white', padding: '0 20px' }}
              >
                {isPlaying ? '⏸️ Stop' : '▶️ Replay Audio'}
              </button>
            )}
          </div>

          {audioBuffer && (
            <div className="audio-preview-container output-container" style={{ marginTop: '20px', padding: '10px', borderLeft: '4px solid #10b981', background: '#0f172a' }}>
              <p style={{ color: '#10b981', fontWeight: 'bold', margin: 0, fontSize: '0.9rem' }}>
                {isPlaying ? '🔊 Streaming out live cached array loops...' : '⏸️ Audio processing ready in sandbox.'}
              </p>
              <p style={{ color: '#64748b', margin: '4px 0 0 0', fontSize: '0.8rem' }}>
                Move the sliders up and down while playing to listen to the instant digital morph.
              </p>
            </div>
          )}
        </section>

        {/* Right Panel: Custom Voice Cloning Engine */}
        <section className="card cloning-section">
          <h2>2. Custom Voice Cloning Engine</h2>
          
          <div className="voice-selector" style={{ marginBottom: '10px' }}>
            <label>Name Your Custom Voice Profile:</label>
            <input 
              type="text" 
              placeholder="e.g., My Cloned Voice V1" 
              value={customVoiceName}
              onChange={(e) => setCustomVoiceName(e.target.value)}
              className="voice-name-input"
              disabled={isUploading}
            />
          </div>

          <div className="tab-navigation">
            <button 
              className={`tab-btn ${activeTab === 'record' ? 'tab-active' : ''}`}
              onClick={() => switchTab('record')}
              disabled={isUploading}
            >
              🎤 Live Record
            </button>
            <button 
              className={`tab-btn ${activeTab === 'upload' ? 'tab-active' : ''}`}
              onClick={() => switchTab('upload')}
              disabled={isUploading}
            >
              📁 Upload File
            </button>
          </div>
          
          {activeTab === 'record' && (
            <div 
              className={`placeholder-zone ${isRecording ? 'recording-active' : ''}`}
              onClick={isRecording ? stopRecording : startRecording}
              style={{ pointerEvents: isUploading ? 'none' : 'auto' }}
            >
              <div className="icon-placeholder">{isRecording ? '🛑' : '🎙️'}</div>
              <p>{isRecording ? 'Recording live voice... Click to Stop' : 'Click to start recording voice input'}</p>
              {isRecording && <span className="timer-badge">{formatTime(recordingDuration)}</span>}
            </div>
          )}

          {activeTab === 'upload' && (
            <div 
              className="placeholder-zone upload-zone" 
              onClick={isUploading ? null : () => fileInputRef.current.click()}
              style={{ pointerEvents: isUploading ? 'none' : 'auto' }}
            >
              <input 
                type="file" 
                ref={fileInputRef}
                onChange={handleFileUpload}
                accept="audio/*"
                style={{ display: 'none' }}
              />
              <div className="icon-placeholder" style={{ color: '#60a5fa' }}>📤</div>
              <p>{uploadedFileName ? `Selected: ${uploadedFileName}` : 'Click to upload pre-recorded audio file'}</p>
              <span className="file-formats-tip">Supports .wav, .mp3, .m4a</span>
            </div>
          )}

          {audioUrl && (
            <div className="audio-preview-container">
              <label>Listen to Sample Reference:</label>
              <audio src={audioUrl} controls className="audio-preview" />
            </div>
          )}

          <button 
            className="btn btn-accent" 
            onClick={handleCreateClone}
            disabled={!audioBlob || isRecording || isUploading}
          >
            {isUploading ? 'Syncing Profile Details...' : 'Simulate & Clone Voice'}
          </button>
        </section>
      </main>
    </div>
  );
}

export default App;