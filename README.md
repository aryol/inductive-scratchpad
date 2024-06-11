# Inductive scratchpad and syllogism

Here we provide the implementation for our paper "How Far Can Transformers Reason? The Locality Barrier and Inductive Scratchpad".

We note that the code is mainly build upon Andrej Karpathy's [NanoGPT](https://github.com/karpathy/nanoGPT) and follows its skeleton. Particularly, 

- `data/` folder includes dataset generation codes for the two cycles task (also random graphs), addition task, and modulo (parity) task. 
- `scripts/` the scripts for running different experiments and reproducing the results that are reported in the paper. More specifically:
    - `script_cycles_no_sp.sh` training Transformers of various sizes on the cycle task without scratchpad for Figure 1.
    - `script_cycles_sp.sh` training Transformers on the cycle task with DFS and inductive scratchpad as in Figure 4.
    - `script_cycles_ood.sh` reproducing the OOD experiment of Figure 4.
    - `script_modulo_random_space.sh` for running length generalization experiments for the parity task presented in Figure 5 of the paper. 
    - `script_addition_random_space.sh` for running length generalization experiments for the addition task using the random space method presented in Figure 5 of the paper. 
    - `script_addition_shift.sh` for running length generalization experiments for the addition task using the shift method presented in Figure 5 of the paper.
    - `script_ER` for the random graph experiment presented in Figure 7 of the paper.
    - `script_cycles_cur.sh` for mixed distribution/curriculum learning experiments on the cycle task presented in Figures 8,9.
    - `script_modulo_sp.sh` for learning parities with cumulative product scratchpads as presented in Figure 10.
- `config/` folder includes the config files for different experiments. Config files are passed to the models in the script files. Config files include the hyperparameters used for each experiment. 
- `model.py` includes the code of decoder-only Transformer model. 
-  `train.py` is the main file used for running experiments. 



    

    
