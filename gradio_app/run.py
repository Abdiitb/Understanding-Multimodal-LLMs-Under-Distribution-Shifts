"""Launch script for the Gradio app."""
from gradio_app.app import build_ui

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
