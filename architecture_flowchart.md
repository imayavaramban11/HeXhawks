# Architecture Flowchart: Hybrid IA-FullSubNet+ & LMS

Here is the visual representation of our proposed pipeline. 

```mermaid
graph TD
    %% Styling
    classDef input fill:#f9f9f9,stroke:#333,stroke-width:2px,color:#000
    classDef ai fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#000
    classDef dsp fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000
    classDef output fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,color:#000
    classDef hardware fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,stroke-dasharray: 5 5,color:#000

    %% Nodes
    A["🎤 Primary Microphone<br/>(Speech + Gunshots/Engine)"]:::input
    
    subgraph AI_Brain ["🧠 AI Module (FullSubNet+)"]
        B["STFT<br/>(Extract Complex Spectrogram)"]:::ai
        C["Full-Band Extractor<br/>(Global Context / Attention)"]:::ai
        D["Sub-Band Extractor<br/>(Local Frequency Patterns)"]:::ai
        E["Complex Mask Generation<br/>(cIRM)"]:::ai
        F["ISTFT<br/>(Reconstruct Audio)"]:::ai
    end

    subgraph Adaptive_Reflex ["⚡ DSP Module (Residual LMS)"]
        G["Adaptive LMS Filter<br/>(Tracks Physical Acoustic Changes)"]:::dsp
        H["🎙️ Error Microphone<br/>(Inside Ear Cup)"]:::input
        I["Filter Weight Update<br/>(Minimizing Residual Error)"]:::dsp
    end

    J["🔊 Speaker Output<br/>(Clean Speech / Anti-Noise)"]:::output
    
    subgraph Deployment ["🖥️ Edge Hardware Target"]
        K["NVIDIA Jetson / Edge DSP<br/>(< 20ms Latency Budget)"]:::hardware
    end

    %% Connections
    A --> B
    B --> C
    B --> D
    C --> D
    D --> E
    E --> F
    
    F -- "Cleaned Reference Signal" --> G
    H -- "Physical Error" --> I
    I -. "Updates" .-> G
    G --> J
    
    %% Hardware mapping (invisible lines just to group)
    AI_Brain -.- K
    Adaptive_Reflex -.- K

```

### How to Read This Flowchart:

1. **The Input (Top):** The primary microphone picks up the chaotic battlefield environment (speech mixed with impulsive and stationary noises).
2. **The AI Brain (Blue):** The signal is converted into the complex domain. The Full-Band extractor looks at the "big picture" to identify what is speech and what is a gunshot. The Sub-Band extractor cleans up specific frequency ranges. It outputs a perfectly cleaned digital signal.
3. **The DSP Reflex (Orange):** Because the physical headset moves and acoustics change, we cannot play the AI's signal blindly. We use the AI's cleaned output as a guide for a lightning-fast LMS filter. The LMS compares the guide against what the error microphone is actually hearing inside the headset, and continuously adjusts itself.
4. **The Output (Green):** The final, physically accurate anti-noise is played through the speaker, canceling the environment while preserving the user's speech.
5. **The Deployment (Purple):** Both the AI and DSP modules must be optimized to run concurrently on embedded edge hardware with strict latency constraints.
