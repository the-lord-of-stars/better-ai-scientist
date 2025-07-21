import gradio as gr
import subprocess
import os
import json
import pandas as pd
import time

# =================================================== Models ===================================================
AVAILABLE_LLMS = [
    "claude-3-5-sonnet-20240620",
    "claude-3-5-sonnet-20241022",
    "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "gpt-4.1",
    "gpt-4.1-2025-04-14",
    "gpt-4.1-mini",
    "gpt-4.1-mini-2025-04-14",
    "gpt-4.1-nano",
    "gpt-4.1-nano-2025-04-14",
    "o1",
    "o1-2024-12-17",
    "o1-preview-2024-09-12",
    "o1-mini",
    "o1-mini-2024-09-12",
    "o3-mini",
    "o3-mini-2025-01-31",
    "llama3.1-405b",
    "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
    "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
    "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
    "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
    "bedrock/anthropic.claude-3-opus-20240229-v1:0",
    "vertex_ai/claude-3-opus@20240229",
    "vertex_ai/claude-3-5-sonnet@20240620",
    "vertex_ai/claude-3-5-sonnet-v2@20241022",
    "vertex_ai/claude-3-sonnet@20240229",
    "vertex_ai/claude-3-haiku@20240307",
    "deepseek-chat",
    "deepseek-coder",
    "deepseek-reasoner",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-thinking-exp-01-21",
    "gemini-2.5-pro-preview-03-25",
    "gemini-2.5-pro-exp-03-25",
]

# ================================================== IDEA FUNCTION =========================================================
# Get the list of experiments from the templates directory
def get_experiment_choices():
    return [d for d in os.listdir("templates") if os.path.isdir(os.path.join("templates", d))]

# Get the ideas dataframe from the ideas.json file
def get_ideas_df(experiment):
    ideas_file = os.path.join("templates", experiment, "ideas.json")
    if not os.path.exists(ideas_file):
        return pd.DataFrame()
    with open(ideas_file, 'r') as f:
        ideas = json.load(f)

    df = pd.DataFrame(ideas)
    # rearrange columns
    wanted_cols = ["Name", "novel", "Interestingness", "Feasibility", "Novelty", "Title", "Experiment"]
    cols = [col for col in wanted_cols if col in df.columns]
    df = df[cols]
    return df    
    
# Get the names of ideas from the ideas.json file
def get_ideas_names(experiment):
    ideas_file = os.path.join("templates", experiment, "ideas.json")
    if not os.path.exists(ideas_file):
        return []
    with open(ideas_file, 'r') as f:
        ideas = json.load(f)
    return [f"{idea['Name']}: {idea['Title']}" for idea in ideas]

#　Save selected ideas to the ideas.json file
def save_selected_ideas(selected, experiment):
    ideas_file = os.path.join("templates", experiment, "ideas.json")
    if not os.path.exists(ideas_file):
        return "Can not find ideas.json", gr.update(), gr.update()
    with open(ideas_file, 'r') as f:
        ideas = json.load(f)
    selected_names = [s.split(":")[0] for s in selected]
    selected_ideas = [idea for idea in ideas if idea['Name'] in selected_names]
    save_file = os.path.join("templates", experiment, "ideas.json")
    with open(save_file, "w") as f:
        json.dump(selected_ideas, f, indent=2, ensure_ascii=False)

    # refresh the ideas dataframe
    df = pd.DataFrame(selected_ideas)
    wanted_cols = ["Name", "novel","Interestingness", "Feasibility", "Novelty", "Title", "Experiment"]
    cols = [col for col in wanted_cols if col in df.columns]
    df = df[cols]
    options = [f"{idea['Name']}: {idea['Title']}" for idea in selected_ideas]
    return f"Save {len(selected_ideas)}  idea to {save_file}!", gr.update(value=df), gr.update(choices=options, value=[]), gr.update(choices=options, value=None)

def refresh_ideas(experiment):
    df = get_ideas_df(experiment)
    options = get_ideas_names(experiment)
    return df, gr.update(choices=options, value=[]), "", gr.update(choices=options, value=None)   
 
def load_idea_for_edit(selected_idea, experiment):
    idea_name = selected_idea.split(":")[0] if selected_idea else None
    ideas_file = os.path.join("templates", experiment, "ideas.json")
    if not (idea_name and os.path.exists(ideas_file)):
        return False, "", "", 0, 0, 0
    with open(ideas_file, "r") as f:
        ideas = json.load(f)
    for idea in ideas:
        if idea["Name"] == idea_name:
            return (
                idea.get("novel", False),
                idea.get("Title", ""),
                idea.get("Experiment", ""),
                idea.get("Interestingness", 0),
                idea.get("Feasibility", 0),
                idea.get("Novelty", 0),
            )
    return False, "", "", 0, 0, 0

def save_idea_edit(selected_idea, novel, title, experiment_str, interestingness, feasibility, novelty, experiment):
    idea_name = selected_idea.split(":")[0] if selected_idea else None
    ideas_file = os.path.join("templates", experiment, "ideas.json")
    if not (idea_name and os.path.exists(ideas_file)):
        return "Save Fail", pd.DataFrame(), [], False, "", "", 0, 0, 0
    with open(ideas_file, "r") as f:
        ideas = json.load(f)
    for idea in ideas:
        if idea["Name"] == idea_name:
            idea["novel"] = novel 
            idea["Title"] = title
            idea["Experiment"] = experiment_str
            idea["Interestingness"] = interestingness
            idea["Feasibility"] = feasibility
            idea["Novelty"] = novelty
            break
    with open(ideas_file, "w") as f:
        json.dump(ideas, f, indent=2, ensure_ascii=False)
    df = pd.DataFrame(ideas)
    wanted_cols = ["Name", "novel","Interestingness", "Feasibility", "Novelty", "Title", "Experiment"]
    cols = [col for col in wanted_cols if col in df.columns]
    df = df[cols]
    options = [f"{idea['Name']}: {idea['Title']}" for idea in ideas]
    return "Save", df, options, False, "", "", 0, 0, 0

# ================================================== Main Function ==================================================
def run_ideas_realtime(model, experiment, num_ideas, phase):
    df = get_ideas_df(experiment)
    options = get_ideas_names(experiment)

    cmd = [
        "python", "-u", "launch_scientist.py",
        "--model", model,
        "--experiment", experiment,
        "--num-ideas", str(num_ideas),
        "--phase", phase,
    ]

    # realtime output
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    output = ""
    for line in iter(process.stdout.readline, ''):
        output += line
        yield output, None, None, None 
    
    process.stdout.close()
    process.wait()

    # refresh the ideas dataframe and options after the process completes
    time.sleep(1)
    df = get_ideas_df(experiment)
    options = get_ideas_names(experiment)
    
    yield output, df, gr.update(choices=options, value=[]), gr.update(choices=options, value=None)

# ================================================== Gradio UI ==================================================
with gr.Blocks() as demo:
    gr.Markdown("## AI Scientist")
    with gr.Tab("ideas"):
        phase = gr.State("ideas")
        # options
        model = gr.Dropdown(choices=AVAILABLE_LLMS, label="Model", value="deepseek-chat")
        experiment = gr.Dropdown(choices=get_experiment_choices(), label="Topic")
        num_ideas = gr.Number(label="Number of Ideas", value=2)
        btn = gr.Button("Run ideas phase")

        # output command lines=20
        output = gr.Textbox(label="Log / Result", lines=40, interactive=True)

        ideas_table = gr.Dataframe(
            value=get_ideas_df(get_experiment_choices()[0]),
            headers=["Name", "novel", "Interestingness", "Feasibility", "Novelty", "Title", "Experiment"],
            datatype=["str", "boolean", "number", "number", "number", "str", "str"],
            label="All ideas",
            interactive=False
        )

        # Use Dropdown
        selected_box = gr.Dropdown(
            choices=get_ideas_names(get_experiment_choices()[0]), 
            label="Keep the ideas you want（Name:Title）",
            multiselect=True
        )
        save_btn = gr.Button("Save ideas")
        save_result = gr.Markdown()

        # Edit the json file
        with gr.Row():
            edit_select = gr.Dropdown(
            choices=get_ideas_names(get_experiment_choices()[0]),
            label="Choose Idea",
            interactive=True,
            value=None
        )
        edit_title = gr.Textbox(label="Title")
        edit_experiment = gr.Textbox(label="Experiment", lines=4)
        edit_novel = gr.Checkbox(label="novel")
        edit_interestingness = gr.Number(label="Interestingness")
        edit_feasibility = gr.Number(label="Feasibility")
        edit_novelty = gr.Number(label="Novelty")
        edit_save_btn = gr.Button("Save Edit")
        edit_result = gr.Markdown()

        # ================================================== Logic ==================================================

        # Logic to refresh ideas when experiment changes
        experiment.change(
            refresh_ideas,
            inputs=experiment,
            outputs=[ideas_table, selected_box, save_result, edit_select]
        )

        edit_select.change(
            load_idea_for_edit,
            inputs=[edit_select, experiment],
            outputs=[edit_novel, edit_title, edit_experiment, edit_interestingness, edit_feasibility, edit_novelty]
        )

        # Button Logic
        btn.click(
            run_ideas_realtime,
            inputs=[model, experiment, num_ideas, phase],
            outputs=[output, ideas_table, selected_box, edit_select]
        )

        save_btn.click(
            save_selected_ideas,
            inputs=[selected_box, experiment],
            outputs=[save_result, ideas_table, selected_box, edit_select]
        )
        
        edit_save_btn.click(
            save_idea_edit,
            inputs=[edit_select, edit_novel, edit_title, edit_experiment, edit_interestingness, edit_feasibility, edit_novelty, experiment],
            outputs=[edit_result, ideas_table, selected_box, edit_novel, edit_title, edit_experiment, edit_interestingness, edit_feasibility, edit_novelty]
        )

    with gr.Tab("experiment"):
        gr.Markdown("**Not Available**")
    with gr.Tab("writeup"):
        gr.Markdown("**Not Available**")
    with gr.Tab("review"):
        gr.Markdown("**Not Available**")
    with gr.Tab("improve"):
        gr.Markdown("**Not Available**")
    with gr.Tab("all"):
        gr.Markdown("**Not Available**")

demo.launch()
