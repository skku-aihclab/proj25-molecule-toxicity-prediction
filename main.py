import os
import sys
import subprocess

# location of scrips to run the experiments
if __name__ == "__main__":
    SCRIPTS = [  
        # "experiments/multimodal/gph_smi_img/test.py",
        # --------- Graph ----------
        # "experiments/graph/train.py",
        # "experiments/graph/test.py",

        # --------- SMILES ----------
        # "experiments/smiles/train.py",
        # "experiments/smiles/test.py",

        # --------- Image ----------
        # "experiments/image/train.py",
        # "experiments/image/test.py",

        # --------- Spectrum ----------
        # "experiments/spectrum/train.py",
        # "experiments/spectrum/test.py",

        # --------- Graph + SMILES ----------
        # "experiments/multimodal/gph_smi/train.py",
        # "experiments/multimodal/gph_smi/test.py",

        # --------- Graph + Image ----------
        # "experiments/multimodal/gph_img/train.py",
        # "experiments/multimodal/gph_img/test.py",

        # --------- Graph + Spectrum ----------
        # "experiments/multimodal/gph_spec/train.py",
        # "experiments/multimodal/gph_spec/test.py",
        
        # --------- SMILES + Image ----------
        "experiments/multimodal/smi_img/train.py",
        "experiments/multimodal/smi_img/test.py",
        
        # --------- SMILES + Spectrum ----------
        "experiments/multimodal/smi_spec/train.py",
        "experiments/multimodal/smi_spec/test.py",
        
        # --------- Image + Spectrum ----------
        "experiments/multimodal/spec_img/train.py",
        "experiments/multimodal/spec_img/test.py",

        # --------- Graph + SMILES + Image (3-MoltiTox) ----------
        # "experiments/multimodal/gph_smi_img/train.py",
        # "experiments/multimodal/gph_smi_img/test.py",
        
        # --------- Graph + SMILES + Image + Spectrum (4-MoltiTox) ----------
        # "experiments/multimodal/moltitox/train.py",
        # "experiments/multimodal/moltitox/test.py",
    ]

    for script in SCRIPTS:
        script_dir  = os.path.dirname(script) # ex: "experiments/smiles"
        script_file = os.path.basename(script)      # "train.py"

        print(f"\n============== Running {script} ==============", flush=True)
        subprocess.run(
            [sys.executable, "-u", script_file],
            check=True,
            cwd=script_dir           
        )


"""
python -u main.py *> result.txt
"""
# This will run the main.py script and save the output to result.txt.
# You can then check the result.txt file for the output.