import argparse
import json
import multiprocessing
import openai
import os
import os.path as osp
import shutil
import sys
import time
import torch
from aider.coders import Coder
from aider.io import InputOutput
from aider.models import Model
from datetime import datetime

from ai_scientist.generate_ideas import generate_ideas, check_idea_novelty
from ai_scientist.llm import create_client, AVAILABLE_LLMS
from ai_scientist.perform_experiments import perform_experiments
from ai_scientist.perform_review import perform_review, load_paper, perform_improvement
from ai_scientist.perform_writeup import perform_writeup, generate_latex

from dotenv import load_dotenv
load_dotenv()


NUM_REFLECTIONS = 3


def print_time():
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run AI scientist experiments")
    parser.add_argument(
        "--skip-idea-generation",
        action="store_true",
        help="Skip idea generation and load existing ideas",
    )
    parser.add_argument(
        "--skip-novelty-check",
        action="store_true",
        help="Skip novelty check and use existing ideas",
    )
    # add type of experiment (nanoGPT, Boston, etc.)
    parser.add_argument(
        "--experiment",
        type=str,
        default="nanoGPT",
        help="Experiment to run AI Scientist on.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-3-5-sonnet-20240620",
        choices=AVAILABLE_LLMS,
        help="Model to use for AI Scientist.",
    )
    parser.add_argument(
        "--writeup",
        type=str,
        default="latex",
        choices=["latex"],
        help="What format to use for writeup",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=0,
        help="Number of parallel processes to run. 0 for sequential execution.",
    )
    parser.add_argument(
        "--improvement",
        action="store_true",
        help="Improve based on reviews.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated list of GPU IDs to use (e.g., '0,1,2'). If not specified, all available GPUs will be used.",
    )
    parser.add_argument(
        "--num-ideas",
        type=int,
        default=50,
        help="Number of ideas to generate",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="semanticscholar",
        choices=["semanticscholar", "openalex"],
        help="Scholar engine to use.",
    )
    # Add a CLI phase controller
    parser.add_argument(
        "--phase",
        type=str,
        default="all",
        choices=["all", "ideas", "experiment", "writeup", "review", "improve"],
        help="Which pipeline phase to run"
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help="Existing project folder for later phases (experiment/writeup/review/improve)"
    )

    return parser.parse_args()


def get_available_gpus(gpu_ids=None):
    if gpu_ids is not None:
        return [int(gpu_id) for gpu_id in gpu_ids.split(",")]
    return list(range(torch.cuda.device_count()))


def check_latex_dependencies():
    """
    Check if required LaTeX dependencies are installed on the system.
    Returns True if all dependencies are found, False otherwise.
    """
    import shutil
    import sys

    required_dependencies = ['pdflatex', 'chktex']
    missing_deps = []

    for dep in required_dependencies:
        if shutil.which(dep) is None:
            missing_deps.append(dep)
    
    if missing_deps:
        print("Error: Required LaTeX dependencies not found:", file=sys.stderr)
        return False
    
    return True
    
def worker(
        queue,
        base_dir,
        results_dir,
        model,
        client,
        client_model,
        writeup,
        improvement,
        gpu_id,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"Worker {gpu_id} started.")
    while True:
        idea = queue.get()
        if idea is None:
            break
        success = do_idea(
            base_dir,
            results_dir,
            idea,
            model,
            client,
            client_model,
            writeup,
            improvement,
            log_file=True,
        )
        print(f"Completed idea: {idea['Name']}, Success: {success}")
    print(f"Worker {gpu_id} finished.")


def do_idea(
        base_dir,
        results_dir,
        idea,
        model,
        client,
        client_model,
        writeup,
        improvement,
        log_file=False,
):
    ## CREATE PROJECT FOLDER
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    idea_name = f"{timestamp}_{idea['Name']}"
    folder_name = osp.join(results_dir, idea_name)
    assert not osp.exists(folder_name), f"Folder {folder_name} already exists."
    destination_dir = folder_name
    shutil.copytree(base_dir, destination_dir, dirs_exist_ok=True)
    with open(osp.join(base_dir, "run_0", "final_info.json"), "r") as f:
        baseline_results = json.load(f)
    # Check if baseline_results is a dictionary before extracting means
    if isinstance(baseline_results, dict):
        baseline_results = {k: v["means"] for k, v in baseline_results.items()}
    exp_file = osp.join(folder_name, "experiment.py")
    vis_file = osp.join(folder_name, "plot.py")
    notes = osp.join(folder_name, "notes.txt")
    with open(notes, "w") as f:
        f.write(f"# Title: {idea['Title']}\n")
        f.write(f"# Experiment description: {idea['Experiment']}\n")
        f.write(f"## Run 0: Baseline\n")
        f.write(f"Results: {baseline_results}\n")
        f.write(f"Description: Baseline results.\n")
    if log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        log_path = osp.join(folder_name, "log.txt")
        log = open(log_path, "a")
        sys.stdout = log
        sys.stderr = log
    try:
        print_time()
        print(f"*Starting idea: {idea_name}*")
        ## PERFORM EXPERIMENTS
        fnames = [exp_file, vis_file, notes]
        io = InputOutput(
            yes=True, chat_history_file=f"{folder_name}/{idea_name}_aider.txt"
        )
        if model == "deepseek-coder-v2-0724":
            main_model = Model("deepseek/deepseek-coder")
        elif model == "deepseek-reasoner":
            main_model = Model("deepseek/deepseek-reasoner")
        elif model == "llama3.1-405b":
            main_model = Model("openrouter/meta-llama/llama-3.1-405b-instruct")
        else:
            main_model = Model(model)
        coder = Coder.create(
            main_model=main_model,
            fnames=fnames,
            io=io,
            stream=False,
            use_git=False,
            edit_format="diff",
        )

        print_time()
        print(f"*Starting Experiments*")
        try:
            success = perform_experiments(idea, folder_name, coder, baseline_results)
        except Exception as e:
            print(f"Error during experiments: {e}")
            print(f"Experiments failed for idea {idea_name}")
            return False

        if not success:
            print(f"Experiments failed for idea {idea_name}")
            return False

        print_time()
        print(f"*Starting Writeup*")
        ## PERFORM WRITEUP
        if writeup == "latex":
            writeup_file = osp.join(folder_name, "latex", "template.tex")
            fnames = [exp_file, writeup_file, notes]
            if model == "deepseek-coder-v2-0724":
                main_model = Model("deepseek/deepseek-coder")
            elif model == "deepseek-reasoner":
                main_model = Model("deepseek/deepseek-reasoner")
            elif model == "llama3.1-405b":
                main_model = Model("openrouter/meta-llama/llama-3.1-405b-instruct")
            else:
                main_model = Model(model)
            coder = Coder.create(
                main_model=main_model,
                fnames=fnames,
                io=io,
                stream=False,
                use_git=False,
                edit_format="diff",
            )
            try:
                perform_writeup(idea, folder_name, coder, client, client_model, engine=args.engine)
            except Exception as e:
                print(f"Failed to perform writeup: {e}")
                return False
            print("Done writeup")
        else:
            raise ValueError(f"Writeup format {writeup} not supported.")

        print_time()
        print(f"*Starting Review*")
        ## REVIEW PAPER
        if writeup == "latex":
            try:
                paper_text = load_paper(f"{folder_name}/{idea['Name']}.pdf")
                review = perform_review(
                    paper_text,
                    model="gpt-4o-2024-05-13",
                    client=openai.OpenAI(),
                    num_reflections=5,
                    num_fs_examples=1,
                    num_reviews_ensemble=5,
                    temperature=0.1,
                )
                # Store the review in separate review.txt file
                with open(osp.join(folder_name, "review.txt"), "w") as f:
                    f.write(json.dumps(review, indent=4))
            except Exception as e:
                print(f"Failed to perform review: {e}")
                return False

        ## IMPROVE WRITEUP
        if writeup == "latex" and improvement:
            print_time()
            print(f"*Starting Improvement*")
            try:
                perform_improvement(review, coder)
                generate_latex(
                    coder, folder_name, f"{folder_name}/{idea['Name']}_improved.pdf"
                )
                paper_text = load_paper(f"{folder_name}/{idea['Name']}_improved.pdf")
                review = perform_review(
                    paper_text,
                    model="gpt-4o-2024-05-13",
                    client=openai.OpenAI(),
                    num_reflections=5,
                    num_fs_examples=1,
                    num_reviews_ensemble=5,
                    temperature=0.1,
                )
                # Store the review in separate review.txt file
                with open(osp.join(folder_name, "review_improved.txt"), "w") as f:
                    f.write(json.dumps(review))
            except Exception as e:
                print(f"Failed to perform improvement: {e}")
                return False
        return True
    except Exception as e:
        print(f"Failed to evaluate idea {idea_name}: {str(e)}")
        return False
    finally:
        print("FINISHED IDEA")
        if log_file:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log.close()


if __name__ == "__main__":
    args = parse_arguments()

    # Check available GPUs and adjust parallel processes if necessary
    available_gpus = get_available_gpus(args.gpus)
    if args.parallel > len(available_gpus):
        print(
            f"Warning: Requested {args.parallel} parallel processes, but only {len(available_gpus)} GPUs available. Adjusting to {len(available_gpus)}."
        )
        args.parallel = len(available_gpus)

    print(f"Using GPUs: {available_gpus}")

    # Check LaTeX dependencies before proceeding
    if args.writeup == "latex" and not check_latex_dependencies():
        sys.exit(1)

    # Create client
    client, client_model = create_client(args.model)

    # Create basic 
    base_dir = osp.join("templates", args.experiment)
    results_dir = osp.join("results", args.experiment)

    if args.phase == "ideas":
        # Only generate ideas 
        ideas = generate_ideas(
            base_dir,
            client=client,
            model=client_model,
            skip_generation=False,
            max_num_generations=args.num_ideas,
            num_reflections=3,
        )
        check_idea_novelty(
            ideas,
            base_dir=base_dir,
            client=client,
            model=client_model,
        )
        print("Idea generation + novelty check done.")

    elif args.phase == "experiment":
        # Only do experiment
        folder = args.folder
        if not (folder and osp.exists(folder)):
            print("--folder does not exist")
            sys.exit(1)
        ideas_file = osp.join(base_dir, "ideas.json")
        with open(ideas_file, "r") as f:
            ideas = json.load(f)
        idea = None
        for _idea in ideas:
            if _idea["Name"] in folder:
                idea = _idea
                break
        if idea is None:
            print("can not find idea")
            sys.exit(1)
        with open(osp.join(base_dir, "run_0", "final_info.json"), "r") as f:
            baseline_results = json.load(f)
        if isinstance(baseline_results, dict):
            baseline_results = {k: v["means"] for k, v in baseline_results.items()}
        fnames = [
            osp.join(folder, "experiment.py"),
            osp.join(folder, "plot.py"),
            osp.join(folder, "notes.txt"),
        ]
        io = InputOutput(yes=True, chat_history_file=f"{folder}/{idea['Name']}_aider.txt")
        main_model = Model(args.model)
        coder = Coder.create(
            main_model=main_model,
            fnames=fnames,
            io=io,
            stream=False,
            use_git=False,
            edit_format="diff",
        )
        perform_experiments(idea, folder, coder, baseline_results)
        print("experiment finished")

    elif args.phase == "writeup":
        # Only write final latex
        folder = args.folder
        if not (folder and osp.exists(folder)):
            print("--folder does not exist")
            sys.exit(1)
        ideas_file = osp.join(base_dir, "ideas.json")
        with open(ideas_file, "r") as f:
            ideas = json.load(f)
        idea = None
        for _idea in ideas:
            if _idea["Name"] in folder:
                idea = _idea
                break
        if idea is None:
            print("can not find idea")
            sys.exit(1)
        exp_file = osp.join(folder, "experiment.py")
        writeup_file = osp.join(folder, "latex", "template.tex")
        notes = osp.join(folder, "notes.txt")
        fnames = [exp_file, writeup_file, notes]
        io = InputOutput(yes=True, chat_history_file=f"{folder}/{idea['Name']}_aider.txt")
        main_model = Model(args.model)
        coder = Coder.create(
            main_model=main_model,
            fnames=fnames,
            io=io,
            stream=False,
            use_git=False,
            edit_format="diff",
        )
        perform_writeup(idea, folder, coder, client, client_model, engine=args.engine)
        print("finish writing.")
    
    elif args.phase == "review":
        # Only review
        folder = args.folder
        if not (folder and osp.exists(folder)):
            print("--folder does not exist")
            sys.exit(1)
        ideas_file = osp.join(base_dir, "ideas.json")
        with open(ideas_file, "r") as f:
            ideas = json.load(f)
        idea = None
        for _idea in ideas:
            if _idea["Name"] in folder:
                idea = _idea
                break
        if idea is None:
            print("Can not find idea")
            sys.exit(1)
        pdf_path = osp.join(folder, f"{idea['Name']}.pdf")
        text = load_paper(pdf_path)
        review = perform_review(
            text,
            model=args.model,
            client=client,
            num_reflections=5,
            num_fs_examples=1,
            num_reviews_ensemble=5,
            temperature=0.1,
        )
        with open(osp.join(folder, "review.txt"), "w") as f:
            json.dump(review, f, indent=4)
        print("Finish review.")

    elif args.phase == "improve":
        # Only do the improvement
        folder = args.folder
        if not (folder and osp.exists(folder)):
            print("--folder does not exist")
            sys.exit(1)
        with open(osp.join(folder, "review.txt")) as f:
            review = json.load(f)
        ideas_file = osp.join(base_dir, "ideas.json")
        with open(ideas_file, "r") as f:
            ideas = json.load(f)
        idea = None
        for _idea in ideas:
            if _idea["Name"] in folder:
                idea = _idea
                break
        if idea is None:
            print("Can not find idea")
            sys.exit(1)
        exp_file = osp.join(folder, "experiment.py")
        writeup_file = osp.join(folder, "latex", "template.tex")
        notes = osp.join(folder, "notes.txt")
        fnames = [exp_file, writeup_file, notes]
        io = InputOutput(yes=True, chat_history_file=f"{folder}/{idea['Name']}_aider.txt")
        main_model = Model(args.model)
        coder = Coder.create(
            main_model=main_model,
            fnames=fnames,
            io=io,
            stream=False,
            use_git=False,
            edit_format="diff",
        )
        perform_improvement(review, coder)
        generate_latex(coder, folder, f"{folder}/{idea['Name']}_improved.pdf")
        print("Finish improvement.")
    
    elif args.phase == "all":
        # Normal process
        ideas = generate_ideas(
            base_dir,
            client=client,
            model=client_model,
            skip_generation=args.skip_idea_generation,
            max_num_generations=args.num_ideas,
            num_reflections=NUM_REFLECTIONS,
        )
        if not args.skip_novelty_check:
            ideas = check_idea_novelty(
                ideas,
                base_dir=base_dir,
                client=client,
                model=client_model,
                engine=args.engine,
            )

        with open(osp.join(base_dir, "ideas.json"), "w") as f:
            json.dump(ideas, f, indent=4)

        novel_ideas = [idea for idea in ideas if idea["novel"]]
        # novel_ideas = list(reversed(novel_ideas))

        if args.parallel > 0:
            print(f"Running {args.parallel} parallel processes")
            queue = multiprocessing.Queue()
            for idea in novel_ideas:
                queue.put(idea)

            processes = []
            for i in range(args.parallel):
                gpu_id = available_gpus[i % len(available_gpus)]
                p = multiprocessing.Process(
                    target=worker,
                    args=(
                        queue,
                        base_dir,
                        results_dir,
                        args.model,
                        client,
                        client_model,
                        args.writeup,
                        args.improvement,
                        gpu_id,
                    ),
                )
                p.start()
                time.sleep(150)
                processes.append(p)

            # Signal workers to exit
            for _ in range(args.parallel):
                queue.put(None)

            for p in processes:
                p.join()

            print("All parallel processes completed.")
        else:
            for idea in novel_ideas:
                print(f"Processing idea: {idea['Name']}")
                try:
                    success = do_idea(
                        base_dir,
                        results_dir,
                        idea,
                        args.model,
                        client,
                        client_model,
                        args.writeup,
                        args.improvement,
                    )
                    print(f"Completed idea: {idea['Name']}, Success: {success}")
                except Exception as e:
                    print(f"Failed to evaluate idea {idea['Name']}: {str(e)}")
                    import traceback
                    print(traceback.format_exc())
        print("All ideas evaluated.")
