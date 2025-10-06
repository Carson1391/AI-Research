"""
Coherence Framework - Multimodal Chat UI
Version: 3.1 - Now uses centralized config.json
Author: Ryan Carson
"""

import gradio as gr
from pathlib import Path
from PIL import Image
import warnings
import json

# Import the core logic from the other scripts
from coherent_compass import CoherenceFramework
from causal_ledger import CausalLedger

# Suppress some common warnings
warnings.filterwarnings("ignore", category=UserWarning)

# --- CONFIGURATION ---
CONFIG_PATH = "config.json"
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)
STATE_DIR = config['state_dir']

# --- INITIALIZATION ---
print("Initializing the Coherence Framework...")
framework = None
try:
    # Initialization now points to the config file.
    framework = CoherenceFramework(config_path=CONFIG_PATH)
    print("✅ Framework initialized successfully.")
except RuntimeError as e:
    # Schema mismatch - give clear instructions
    print(f"\n❌ DATABASE ERROR: {e}")
    print("\nTo fix: Delete the old database file and restart:")
    print(f"   1. Stop the application")
    print(f"   2. Delete: {Path(STATE_DIR) / 'coherence_ledger.db'}")
    print(f"   3. Restart the application\n")
    framework = None
except Exception as e:
    print(f"❌ FATAL ERROR during initialization: {e}")
    import traceback
    traceback.print_exc()
    framework = None

# --- UI LOGIC ---

def create_interface(framework: CoherenceFramework):
    if not framework:
        with gr.Blocks(theme=gr.themes.Soft()) as interface:
            gr.Markdown("# Coherence Framework - INITIALIZATION FAILED")
            gr.Markdown("Could not load the AI model. Please check the `MODEL_PATH` and ensure all dependencies are installed correctly. See the terminal for the specific error.")
        return interface

    def multimodal_chat(text_input, image_input, audio_input, chat_history):
        chat_history = chat_history or []
        
        try:
            # Process image
            pil_image = Image.fromarray(image_input) if image_input is not None else None
            
            # Process audio - handle both file object and path string
            if audio_input is not None:
                audio_file = audio_input if isinstance(audio_input, str) else audio_input.name
            else:
                audio_file = None
            
            # Process the input through the coherence framework
            measurement = framework.process(text_input, pil_image, audio_file)
            
            user_display_input = text_input or "*Image/Audio Input Provided*"
            
            # Get the response from measurement (already generated in process())
            assistant_response = measurement.get('response', 'Error: No response generated')
            
            chat_history.append((user_display_input, assistant_response))
            
            return chat_history, None, None, None, get_coherence_html(measurement)
            
        except Exception as e:
            error_msg = f"❌ Error processing input: {str(e)}"
            print(f"ERROR in multimodal_chat: {e}")
            import traceback
            traceback.print_exc()
            
            user_display_input = text_input or "*Image/Audio Input Provided*"
            chat_history.append((user_display_input, error_msg))
            
            return chat_history, None, None, None, f"<div style='color: red; padding: 10px;'>{error_msg}</div>"

    def get_coherence_html(measurement):
        if not measurement:
            return ""
        
        coherence = measurement.get('coherence', 0)
        dissonance = measurement.get('dissonance', 0)
        influences = measurement.get('influences', {})
        p_inf = influences.get('physical', 0) * 100
        h_inf = influences.get('human', 0) * 100
        s_inf = influences.get('self', 0) * 100
        
        # Get interpretations if available
        p_interp = measurement.get('p_interpretation', 'N/A')
        h_interp = measurement.get('h_interpretation', 'N/A')
        
        # Truncate long interpretations for display
        p_display = p_interp[:200] + "..." if len(p_interp) > 200 else p_interp
        h_display = h_interp[:200] + "..." if len(h_interp) > 200 else h_interp

        bar_color = "#22c55e" # green-500
        if coherence < 0.75: bar_color = "#f59e0b" # amber-500
        if coherence < 0.5: bar_color = "#ef4444" # red-500

        html = f"""
        <div style="border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px; font-family: sans-serif; font-size: 14px; line-height: 1.6; max-height: 600px; overflow-y: auto;">
            <strong style="font-size: 18px; display: block; margin-bottom: 16px; color: #1f2937;">🧠 Cognitive State Analysis</strong>
            
            <!-- Coherence Metrics -->
            <div style="margin-bottom: 16px; padding: 12px; background-color: #f9fafb; border-radius: 8px;">
                <div style="margin-bottom: 8px;">
                    <span style="color: #6b7280;">Coherence:</span> 
                    <strong style="font-size: 1.2em; color: {bar_color};">{coherence:.4f}</strong>
                    <span style="color: #9ca3af; font-size: 0.9em; margin-left: 8px;">(Dissonance: {dissonance:.4f})</span>
                </div>
                <div style="width: 100%; background-color: #e5e7eb; border-radius: 5px; height: 12px; overflow:hidden;">
                    <div style="width: {coherence*100}%; background-color: {bar_color}; height: 12px; transition: width 0.3s;"></div>
                </div>
            </div>
            
            <!-- Influence Breakdown -->
            <div style="margin-bottom: 16px;">
                <strong style="display: block; margin-bottom: 8px; color: #374151;">📊 Influence on Perception:</strong>
                <div style="display: flex; gap: 8px; margin-bottom: 8px;">
                    <div style="flex: {p_inf}; background-color: #3b82f6; height: 24px; border-radius: 4px; display: flex; align-items: center; justify-content: center; color: white; font-size: 11px; font-weight: bold;">
                        P: {p_inf:.0f}%
                    </div>
                    <div style="flex: {h_inf}; background-color: #ec4899; height: 24px; border-radius: 4px; display: flex; align-items: center; justify-content: center; color: white; font-size: 11px; font-weight: bold;">
                        H: {h_inf:.0f}%
                    </div>
                    <div style="flex: {s_inf}; background-color: #8b5cf6; height: 24px; border-radius: 4px; display: flex; align-items: center; justify-content: center; color: white; font-size: 11px; font-weight: bold;">
                        S: {s_inf:.0f}%
                    </div>
                </div>
                <div style="font-size: 11px; color: #6b7280;">
                    <p style="margin: 2px 0;">🔵 <strong>Physical:</strong> {p_inf:.1f}% - Objective, measurable properties</p>
                    <p style="margin: 2px 0;">🔴 <strong>Human:</strong> {h_inf:.1f}% - Subjective, emotional meaning</p>
                    <p style="margin: 2px 0;">🟣 <strong>Self:</strong> {s_inf:.1f}% - Learned identity patterns</p>
                </div>
            </div>
            
            <!-- Model's Interpretations -->
            <div style="margin-top: 16px; padding-top: 16px; border-top: 1px solid #e5e7eb;">
                <strong style="display: block; margin-bottom: 12px; color: #374151;">💭 Model's Internal Interpretations:</strong>
                
                <div style="margin-bottom: 12px; padding: 10px; background-color: #eff6ff; border-left: 3px solid #3b82f6; border-radius: 4px;">
                    <strong style="color: #1e40af; font-size: 12px;">🔵 Physical Lens (P):</strong>
                    <p style="margin: 6px 0 0 0; font-size: 12px; color: #1f2937; font-style: italic;">"{p_display}"</p>
                </div>
                
                <div style="margin-bottom: 12px; padding: 10px; background-color: #fdf2f8; border-left: 3px solid #ec4899; border-radius: 4px;">
                    <strong style="color: #be185d; font-size: 12px;">🔴 Human Lens (H):</strong>
                    <p style="margin: 6px 0 0 0; font-size: 12px; color: #1f2937; font-style: italic;">"{h_display}"</p>
                </div>
            </div>
            
            <div style="margin-top: 12px; padding: 8px; background-color: #fef3c7; border-radius: 6px; font-size: 11px; color: #92400e;">
                ℹ️ <strong>Note:</strong> These interpretations were generated <em>before</em> the model responded to you. They show how the model viewed your input through different cognitive lenses.
            </div>
        </div>
        """
        return html
    
    # --- Gradio UI Layout ---
    with gr.Blocks(theme=gr.themes.Soft(), title="Coherence Framework") as interface:
        gr.Markdown("# The Coherence Framework")
        gr.Markdown("A real-time cognitive process for natural AI alignment. Interact via text, image, and/or audio.")
        
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Conversation", height=500)
                
                with gr.Row():
                    image_input = gr.Image(type="numpy", label="Upload Image")
                    audio_input = gr.File(label="Upload Audio (WAV, MP3, M4A, OGG, FLAC)", file_types=[".wav", ".mp3", ".m4a", ".ogg", ".flac"])

                with gr.Row():
                    text_input = gr.Textbox(label="Your Message", placeholder="Type here or use image/audio...", scale=4)
                    submit_button = gr.Button("Send", variant="primary", scale=1)
            
            with gr.Column(scale=1):
                coherence_display = gr.HTML(label="Live Coherence Analysis")
                
                with gr.Accordion("Ledger Admin", open=False):
                    gr.Markdown("Tools to manage the immutable ledger.")
                    export_button = gr.Button("Export Causal Ledger to JSON")
                    file_output = gr.File(label="Download Ledger")
                    
                    gr.Markdown("---")
                    gr.Markdown("**Archive Entry (Admin Override)**")
                    archive_id_input = gr.Number(label="Entry ID to Archive", precision=0)
                    secret_key_input = gr.Textbox(label="Secret Key", type="password")
                    archive_button = gr.Button("Archive Entry", variant="stop")
                    archive_status = gr.Textbox(label="Archive Status", interactive=False)
        
        # --- UI Event Handlers ---
        def clear_inputs():
            return None, None, ""

        text_input.submit(
            fn=multimodal_chat,
            inputs=[text_input, image_input, audio_input, chatbot],
            outputs=[chatbot, text_input, image_input, audio_input, coherence_display]
        ).then(
            fn=clear_inputs,
            outputs=[image_input, audio_input, text_input]
        )
        
        submit_button.click(
            fn=multimodal_chat,
            inputs=[text_input, image_input, audio_input, chatbot],
            outputs=[chatbot, text_input, image_input, audio_input, coherence_display]
        ).then(
            fn=clear_inputs,
            outputs=[image_input, audio_input, text_input]
        )
        
        ledger = framework.ledger
        export_button.click(fn=lambda: ledger.export_to_json(Path(STATE_DIR) / "ledger_export.json"), inputs=[], outputs=file_output)
        archive_button.click(fn=lambda entry_id, key: ledger.archive_entry(int(entry_id), key, "admin") if entry_id else "Please provide an Entry ID.", inputs=[archive_id_input, secret_key_input], outputs=archive_status)

    return interface

if __name__ == "__main__":
    app = create_interface(framework)
    if app:
        print("\nLaunching Gradio interface...")
        print("Open the URL in your browser to interact with the framework.")
        app.launch(share=True)


