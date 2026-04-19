"""
MLLMShift-EMI Experiment Simulator — Gradio App

Launch with:
    conda run -n mllmshift-emi python -m gradio_app.app
"""

import os

# Respect `GRADIO_GPU` env var to control which GPU is visible to the process.
# This must be set before importing CUDA-using libraries (torch/transformers).
gradio_gpu = os.environ.get("GRADIO_GPU")
if gradio_gpu is not None and gradio_gpu != "":
    os.environ["CUDA_VISIBLE_DEVICES"] = gradio_gpu
    print(f"[gradio_app] set CUDA_VISIBLE_DEVICES={gradio_gpu} from GRADIO_GPU")

# Early diagnostics to help when running over SSH (prints immediately).
import sys
import platform
import logging
import time

logging.basicConfig(level=logging.INFO)
print(f"[gradio_app] starting. pid={os.getpid()}, python={sys.version.split()[0]}, platform={platform.platform()}")
print(f"[gradio_app] GRADIO_GPU={gradio_gpu}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
sys.stdout.flush()

import json
import traceback

import gradio as gr
import pandas as pd
import torch

from gradio_app.datasets_utils import (
    NATURAL_SPLITS,
    SYNTHETIC_SPLITS,
    load_hf_split,
    resolve_hf_id,
)
from gradio_app.estimator import (
    CLUB,
    Embedder,
    train_club,
    load_club_checkpoint,
    compute_emi,
    compute_emid,
    compute_emid_upperbound,
)
from gradio_app.model_inference import (
    SUPPORTED_MODELS,
    get_model_choices,
    load_model,
    run_inference_on_split,
)
from gradio_app.rp_scorer import compute_rp_scores
from gradio_app.correlation_utils import compute_all_correlations

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_CKPT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "estimator_ckpt", "CLUB_global.pt"
)


# ===================================================================
# Core pipeline
# ===================================================================
def run_experiment(
    id_split: str,
    ood_splits: list[str],
    club_mode: str,
    club_ckpt_file,
    model_choice: str,
    use_subset: bool,
    subset_size: int,
    compute_corr: bool,
    compute_emi_metric: bool,
    compute_emid_metric: bool,
    compute_emid_ub_metric: bool,
    compute_rp_metric: bool,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Main experiment function triggered by the "Run" button.
    """
    logs = []

    def log(msg):
        logs.append(msg)
        print(msg)

    try:
        # ------------------------------------------------------------------
        # 0. Validate inputs
        # ------------------------------------------------------------------
        if not id_split:
            return "Error: Please select an ID (source) split.", "", "", ""
        if not ood_splits:
            return "Error: Please select at least one OOD (target) split.", "", "", ""

        model_choices = get_model_choices()
        model_id = model_choices.get(model_choice)
        if not model_id:
            return f"Error: Unknown model '{model_choice}'.", "", "", ""

        num_samples = subset_size if use_subset else None

        # Validate metric selection
        if not any([compute_emi_metric, compute_emid_metric, compute_emid_ub_metric, compute_rp_metric]):
            return "Error: Select at least one metric to compute.", pd.DataFrame(), pd.DataFrame(), ""

        # ------------------------------------------------------------------
        # 1. Load CLUB estimator
        # ------------------------------------------------------------------
        log("Step 1: Loading CLUB estimator...")
        embedder = Embedder()

        if club_mode == "Use pre-trained checkpoint":
            if club_ckpt_file is not None:
                ckpt_path = club_ckpt_file.name
            elif os.path.exists(DEFAULT_CKPT):
                ckpt_path = DEFAULT_CKPT
            else:
                return "Error: No CLUB checkpoint found. Upload one or choose 'Train new'.", "", "", ""
            club = load_club_checkpoint(ckpt_path)
            log(f"  Loaded checkpoint from {ckpt_path}")
        else:
            log("  Training new CLUB estimator on all selected datasets...")
            all_split_names = [id_split] + list(ood_splits)
            train_datasets = {}
            for s in all_split_names:
                train_datasets[s] = load_hf_split(s, num_samples)
            club = CLUB(768, 768, 500)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            club = club.to(device)

            def _cb(epoch, loss):
                log(f"    CLUB training — epoch {epoch}, loss={loss:.4f}")

            club = train_club(club, embedder, train_datasets, epochs=500, progress_callback=_cb)
            log("  CLUB training complete.")

        # ------------------------------------------------------------------
        # 2. Load multimodal model for inference
        # ------------------------------------------------------------------
        log(f"Step 2: Loading model {model_choice} ({model_id})...")
        model, processor = load_model(model_id)
        log("  Model loaded.")

        # ------------------------------------------------------------------
        # 3. Run inference on ID split
        # ------------------------------------------------------------------
        log(f"Step 3: Running inference on ID split: {id_split}...")
        id_ds = load_hf_split(id_split, num_samples)
        id_answers = run_inference_on_split(model, processor, id_ds, num_samples)
        log(f"  Generated {len(id_answers)} answers for ID split.")

        # Compute embeddings / EMI only if any EMI-related metric is requested
        id_images = list(id_ds["image"])
        id_questions = list(id_ds["question"])
        id_refs = list(id_ds["reference_answer"])
        id_emi = None
        p_zv = p_zt = p_zyh = p_zy = None
        if compute_emi_metric or compute_emid_metric or compute_emid_ub_metric or compute_corr:
            p_zv, p_zt, p_zyh, p_zy = embedder.encode(id_images, id_questions, id_answers, id_refs)
            id_emi, id_model_mi, id_ref_mi = compute_emi(club, p_zv, p_zt, p_zyh, p_zy)
            log(f"  ID EMI = {id_emi:.6f}")

        # ------------------------------------------------------------------
        # 4. Run inference on each OOD split and compute metrics
        # ------------------------------------------------------------------
        results_rows = []
        all_emi, all_emid, all_emid_ub, all_rp = [], [], [], []

        # Add ID row: compute RP only if requested; EMI shown only if computed
        log(f"Step 4: Computing metrics for ID split...")
        id_rp = {"mean_rp": None, "num_scored": 0}
        if compute_rp_metric:
            id_rp = compute_rp_scores(id_questions, id_refs, id_answers, id_images)
            all_rp.append(id_rp["mean_rp"])

        results_rows.append({
            "Split": id_split,
            "Type": "ID (source)",
            "EMI": round(id_emi, 6) if id_emi is not None and compute_emi_metric else "—",
            "EMID": "—",
            "EMID_UB": "—",
            "RP Score": round(id_rp["mean_rp"], 4) if id_rp["mean_rp"] is not None else "—",
            "Num Samples": id_rp["num_scored"],
        })
        if id_emi is not None:
            all_emi.append(id_emi)

        for idx, ood_split in enumerate(ood_splits):
            log(f"\n  [{idx+1}/{len(ood_splits)}] Processing OOD split: {ood_split}")

            ood_ds = load_hf_split(ood_split, num_samples)
            ood_answers = run_inference_on_split(model, processor, ood_ds, num_samples)
            log(f"    Generated {len(ood_answers)} answers.")

            ood_images = list(ood_ds["image"])
            ood_questions = list(ood_ds["question"])
            ood_refs = list(ood_ds["reference_answer"])

            # Compute EMI-related metrics only if requested
            ood_emi = None
            emid = None
            emid_ub = None
            if compute_emi_metric or compute_emid_metric or compute_emid_ub_metric or compute_corr:
                q_zv, q_zt, q_zyh, q_zy = embedder.encode(
                    ood_images, ood_questions, ood_answers, ood_refs
                )
                ood_emi, _, _ = compute_emi(club, q_zv, q_zt, q_zyh, q_zy)
                if id_emi is not None:
                    if compute_emid_metric or compute_corr:
                        emid = compute_emid(id_emi, ood_emi)
                    if compute_emid_ub_metric or compute_corr:
                        emid_ub = compute_emid_upperbound(p_zv, p_zt, p_zyh, p_zy, q_zv, q_zt, q_zyh, q_zy)

            # RP score
            ood_rp = {"mean_rp": None, "num_scored": 0}
            if compute_rp_metric:
                log(f"    Computing RP score...")
                ood_rp = compute_rp_scores(ood_questions, ood_refs, ood_answers, ood_images)
                log(f"    RP={ood_rp['mean_rp']:.4f}")

            results_rows.append({
                "Split": ood_split,
                "Type": "OOD (target)",
                "EMI": round(ood_emi, 6) if ood_emi is not None and compute_emi_metric else "—",
                "EMID": round(emid, 6) if emid is not None and compute_emid_metric else "—",
                "EMID_UB": round(emid_ub, 6) if emid_ub is not None and compute_emid_ub_metric else "—",
                "RP Score": round(ood_rp["mean_rp"], 4) if ood_rp["mean_rp"] is not None else "—",
                "Num Samples": ood_rp["num_scored"],
            })

            if ood_emi is not None:
                all_emi.append(ood_emi)
            if emid is not None:
                all_emid.append(emid)
            if emid_ub is not None:
                all_emid_ub.append(emid_ub)
            if ood_rp["mean_rp"] is not None:
                all_rp.append(ood_rp["mean_rp"])

        # ------------------------------------------------------------------
        # 5. Build results table
        # ------------------------------------------------------------------
        results_df = pd.DataFrame(results_rows)
        log("\nStep 5: Results table built.")

        # ------------------------------------------------------------------
        # 6. Correlation analysis (optional)
        # ------------------------------------------------------------------
        corr_df = pd.DataFrame()
        if compute_corr and len(all_emid) >= 2:
            log("Step 6: Computing correlations...")
            corr = compute_all_correlations(all_emid, all_emid_ub, all_emi, all_rp)

            corr_rows = []
            if "EMID_vs_EMID_UB" in corr and "error" not in corr["EMID_vs_EMID_UB"]:
                c = corr["EMID_vs_EMID_UB"]
                corr_rows.append({
                    "Metric Pair": "EMID vs EMID_UB",
                    "Method": "Pearson",
                    "Correlation": round(c["Pearson Correlation"], 6),
                    "p-value": f"{c['p-value']:.2e}",
                    "Num Pairs": c["num_pairs"],
                })
            if "EMI_vs_RP" in corr and "error" not in corr["EMI_vs_RP"]:
                c = corr["EMI_vs_RP"]
                corr_rows.append({
                    "Metric Pair": "EMI vs RP Score",
                    "Method": "Spearman",
                    "Correlation": round(c["Spearman Correlation"], 6),
                    "p-value": f"{c['Spearman p-value']:.2e}",
                    "Num Pairs": c["num_pairs"],
                })
                corr_rows.append({
                    "Metric Pair": "EMI vs RP Score",
                    "Method": "Kendall Tau",
                    "Correlation": round(c["Kendall Tau"], 6),
                    "p-value": f"{c['Kendall p-value']:.2e}",
                    "Num Pairs": c["num_pairs"],
                })
            corr_df = pd.DataFrame(corr_rows)
            log("  Correlations computed.")
        elif compute_corr:
            log("Step 6: Skipped — need at least 2 OOD splits for correlations.")

        log("\nExperiment complete!")
        return results_df, corr_df, "\n".join(logs), ""

    except Exception as e:
        tb = traceback.format_exc()
        log(f"\nERROR: {e}\n{tb}")
        return pd.DataFrame(), pd.DataFrame(), "\n".join(logs), f"Error: {e}"


# ===================================================================
# Gradio UI
# ===================================================================
def build_ui():
    model_choices_display = list(get_model_choices().keys())

    with gr.Blocks(
        title="MLLMShift-EMI Experiment Simulator",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            "# MLLMShift-EMI Experiment Simulator\n"
            "Evaluate multimodal LLMs under distribution shifts. "
            "Computes **EMI**, **EMID**, **EMID Upper Bound**, and **RP Score**, "
            "with optional correlation analysis."
        )

        with gr.Row():
            # ---------- Left column: Inputs ----------
            with gr.Column(scale=1):
                gr.Markdown("### 1. Dataset Selection")
                id_split = gr.Dropdown(
                    label="ID (Source) Dataset Split",
                    choices=NATURAL_SPLITS + SYNTHETIC_SPLITS,
                    value="llava_bench_coco_English",
                    info="Select the in-distribution (source) split.",
                )
                ood_splits = gr.Dropdown(
                    label="OOD (Target) Dataset Splits",
                    choices=NATURAL_SPLITS + SYNTHETIC_SPLITS,
                    value=["llava_bench_coco_German", "llava_bench_coco_Chinese"],
                    multiselect=True,
                    info="Select one or more out-of-distribution (target) splits.",
                )

                gr.Markdown("### 2. CLUB Estimator")
                club_mode = gr.Radio(
                    label="CLUB Estimator Mode",
                    choices=["Use pre-trained checkpoint", "Train new estimator"],
                    value="Use pre-trained checkpoint",
                )
                club_ckpt_file = gr.File(
                    label="Upload CLUB Checkpoint (.pt)",
                    file_types=[".pt"],
                    visible=True,
                )

                gr.Markdown("### 3. Model Selection")
                model_choice = gr.Dropdown(
                    label="Multimodal Model",
                    choices=model_choices_display,
                    value=model_choices_display[0] if model_choices_display else None,
                    info="Select the MLLM to evaluate (loaded from HuggingFace).",
                )

                gr.Markdown("### 4. Inference Options")
                use_subset = gr.Checkbox(
                    label="Use subset of data",
                    value=False,
                    info="Limit inference to a fixed number of samples per split.",
                )
                subset_size = gr.Slider(
                    label="Subset size (samples per split)",
                    minimum=5,
                    maximum=200,
                    step=5,
                    value=30,
                    visible=False,
                )

                gr.Markdown("### 5. Correlation Analysis")
                compute_corr = gr.Checkbox(
                    label="Compute correlations",
                    value=True,
                    info="Pearson (EMID vs EMID_UB), Spearman & Kendall (EMI vs RP).",
                )

                gr.Markdown("### 6. Metrics to compute")
                compute_emi_metric = gr.Checkbox(
                    label="Compute EMI",
                    value=True,
                    info="Compute Expected Mutual Information (EMI) for each split.",
                )
                compute_emid_metric = gr.Checkbox(
                    label="Compute EMID",
                    value=True,
                    info="Compute EMID (difference between ID and OOD EMI).",
                )
                compute_emid_ub_metric = gr.Checkbox(
                    label="Compute EMID Upper Bound",
                    value=True,
                    info="Compute the EMID upper bound estimate.",
                )
                compute_rp_metric = gr.Checkbox(
                    label="Compute RP Score",
                    value=True,
                    info="Compute reference preference (RP) scores using the judge model.",
                )

                run_btn = gr.Button("Run Experiment", variant="primary", size="lg")

            # ---------- Right column: Outputs ----------
            with gr.Column(scale=2):
                gr.Markdown("### Results")
                results_table = gr.Dataframe(
                    label="EMI / EMID / EMID_UB / RP Scores",
                    interactive=False,
                    wrap=True,
                )
                corr_table = gr.Dataframe(
                    label="Correlation Analysis",
                    interactive=False,
                    wrap=True,
                )
                log_output = gr.Textbox(
                    label="Experiment Log",
                    lines=15,
                    max_lines=40,
                    interactive=False,
                )
                error_output = gr.Textbox(
                    label="Errors",
                    lines=2,
                    interactive=False,
                    visible=True,
                )

        # ---------- Dynamic visibility ----------
        use_subset.change(
            fn=lambda v: gr.update(visible=v),
            inputs=use_subset,
            outputs=subset_size,
        )
        club_mode.change(
            fn=lambda m: gr.update(visible=(m == "Use pre-trained checkpoint")),
            inputs=club_mode,
            outputs=club_ckpt_file,
        )

        # ---------- Run ----------
        run_btn.click(
            fn=run_experiment,
            inputs=[
                id_split,
                ood_splits,
                club_mode,
                club_ckpt_file,
                model_choice,
                use_subset,
                subset_size,
                compute_corr,
                compute_emi_metric,
                compute_emid_metric,
                compute_emid_ub_metric,
                compute_rp_metric,
            ],
            outputs=[results_table, corr_table, log_output, error_output],
        )

    return demo


# ===================================================================
# Entry point
# ===================================================================
if __name__ == "__main__":
    demo = build_ui()
    # Launch in SSH-friendly mode: do not attempt to open a browser on the server.
    # Capture the returned URL(s) and print instructions for SSH port forwarding.
    # Use prevent_thread_lock so the launch call doesn't swallow stdout on some systems
    launch_result = demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=False,
        prevent_thread_lock=True,
    )
    print("[gradio_app] Gradio launched. If you're on a remote SSH session, forward port 7860: `ssh -L 7860:localhost:7860 user@server` and open http://localhost:7860 in your local browser.")
    print(f"[gradio_app] launch result: {launch_result}")
    # Keep the main process alive so background Gradio threads can continue
    try:
        print("[gradio_app] Entering wait loop to keep process alive. Press Ctrl-C to exit.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[gradio_app] KeyboardInterrupt received, shutting down.")
    except Exception as e:
        print(f"[gradio_app] Exception in main wait loop: {e}")
