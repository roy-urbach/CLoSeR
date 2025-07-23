# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
        . /etc/bashrc
fi

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=

# User specific aliases and functions

# Change the following according to your conda!
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/projects/schneidmann/royu/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/projects/schneidmann/royu/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/projects/schneidmann/royu/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/projects/schneidmann/royu/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

# change to your environment name
conda activate tf-gpu-cuda-11-8


# aliases and functions

# default module to use when calling functions. To change, modify MODULE yourself or use modneur and modvis
MODULE="vision"
function modvis { MODULE="vision"; }
function modneur { MODULE="neuronal"; }


# gets the first word
function relast { grep -Eo "^[^ ]+" $@; }

# send a job to train a model
function train { python3  train_cmd_format.py -m $MODULE $@;  cat /tmp_cmd; echo; . /tmp_cmd; rm /tmp_cmd; }

# send a job to evaluate a model
function evaluate { 
			model=$1; 
			shift; 
			if [[ -f $MODULE/config/$model.json ]]; then 
				if [[ -d $MODULE/models/$model ]]; then
					if [[ ! -f $MODULE/models/$model/nan ]]; then 
						if [[ -f $MODULE/models/$model/is_evaluating ]]; then 
							echo already evaluating $model; 
						else 
							echo evaluating $model;
							bsub -q short -J $model -o $MODULE/models/$model/eval_out.o -e $MODULE/models/$model/eval_err.e -C 1 -R rusage[mem=30000] "python3 evaluate.py -j $model -m $MODULE $@";
						fi; 
					fi; 
				else
					echo "$MODULE/models/$model doesn't exist";
				fi;
			fi
		}

# if didn't evaluate before, sends to evaluation
function evaluate_if { model=$1; if [[ ! -f $MODULE/models/$model/classification_eval.json ]]; then evaluate $@; fi; }

# send a job to measure cross-path measures
function measure {
			model=$1; 
			shift; 
			if [[ -f $MODULE/config/$model.json ]]; then 
				if [[ -d $MODULE/models/$model ]]; then 
					if [[ ! -f $MODULE/models/$model/nan ]]; then 
						if [[ -f $MODULE/models/$model/is_measuring ]]; then 
							echo already evaluating $model; 
						else 
							echo measuring $model; 
							bsub -q short -J $model -o $MODULE/models/$model/meas_out.o -e $MODULE/models/$model/meas_err.e -C 1 -R rusage[mem=20000] "python3 measure.py -j $model -m $MODULE $@"; 
						fi; 
					fi; 
				else echo "$MODULE/models/$model doesn't exist"; 
				fi; 
			fi
			}

# if didn't measure before, sends to evaluation
function measure_if { 
			model=$1; 
			if [[ ! -f $MODULE/models/$model/measures.json ]]; 
				then measure $@; 
			fi; 
			}

# sends a job that calculates the mean class-class likelihood (see cls_likelihood.py and figure 2a)
function cls_like { model=$1;  bsub -q short -J $model -o $MODULE/models/$model/cls_like.o -e $MODULE/models/$model/cls_like.e -C 1 -R rusage[mem=30000] "python3 cls_likelihood.py -j $model $@"; }

# given a regex, prints all the models that matches it and their current epoch
function epoch_regex {
                    for model in $(ls $MODULE/models/ | grep $1);
                    do echo $model $(ls $MODULE/models/$model/checkpoints/ | grep model_weights_[0-9]*.index | grep -Eo [0-9]*);
                    done;
                    }

# look at the results
# all the functions bellow assume MODULE is correct, and the usage is just "<func> <model>"

## tail and watch models while training\evaluating
function tailm { tail -f $MODULE/models/$1/output.o; }              # tails and watches its training log
function taile { tail -f $MODULE/models/$1/error.e; }               # tails and watches its training error log
function tail_eval { tail -f $MODULE/models/$1/eval_out.o; }        # tails and watches its evaluation log
function tail_eval_err { tail -f $MODULE/models/$1/eval_err.e; }    # tails and watches its evaluation error log

## vim model's logs
function vimm { vim $MODULE/models/$1/output.o; }                   # vim training log
function vime { vim $MODULE/models/$1/error.e; }                    # vim training error log
function vim_eval { vim $MODULE/models/$1/eval_out.o; }             # vim evaluation log
function vim_eval_err { vim $MODULE/models/$1/eval_err.e; }         # vim evaluation error log
function vimc { vim $MODULE/models/$1/classification_eval.json; }   # vim evaluation results

## cat model's log
function catm { cat $MODULE/models/$1/output.o; }                   # training log
function cate { cat $MODULE/models/$1/error.e; }                    # training error log
function catc { cat $MODULE/models/$1/classification_eval.json; }   # evaluation results


# Configuration manuplations
function conf_cat { cat $MODULE/config/$1.json; }                       # prints a model configuration
function conf_cp { cp $MODULE/config/$1.json $MODULE/config/$2.json; }  # copies a model's configuration to another name
function conf_nano { nano $MODULE/config/$1.json; }                     # edit (using nano) a model's configuration


# given a jobid, returns the jobname
function jobname_from_jobid() { bjobs -l $1 | tr -d '\n' | tr -d ' ' | grep -Eo "JobName<[^>]+>" | grep -oP '(?<=<).*(?=>)'; }

# returns the names of all current jobs
function jobnames() { for jobid in $(bjobs | grep -Eo "^[0-9]+"); do echo $jobid $(jobname_from_jobid $jobid); done; }

# returns all the models that are currently training, and their epoch
function training() {
                for jobid in $(bjobs | grep gpu | grep -Eo "^[0-9]+"); do
                    jobname=$(jobname_from_jobid $jobid);
                    if [ -e "$MODULE/models/$jobname" ]; then
                        echo $jobid $(epoch_regex "$jobname$");
                    fi;
                done;
                }

# given a model's name, prints "True" if it reached nan, otherwise "False"
function isnan { model=$1; if [[ -f $MODULE/models/$model/nan ]]; then echo True; else echo False; fi; }

# sends a job that processes an allen institute session (see process_neuronal_dataset.py)
function processneur { ses=$1; bsub -q short -J $ses -o $MODULE/data/$ses.o -e $MODULE/data/$ses.e -C 1 -R rusage[mem=8000] "python3 process_neuronal_dataset.py --index $ses"; }


function save_sum { model=$1; sum=$(catm $model | grep -E "\s*Run time \s*:\s*([0-9]*)\s.*" | grep -Eo "[0-9]*" | awk '{total += $1} END {print total}'); echo "$sum" > $MODULE/models/$model/runtime.txt;  }

watch_func() {
  while true; do
    tmpfile=$(mktemp)
    training > "$tmpfile"
    printf "\033[3J\033[H\033[J"  # Clear scrollback + move cursor + clear screen
    cat "$tmpfile"
    rm -f "$tmpfile"
    sleep 10
  done
}
