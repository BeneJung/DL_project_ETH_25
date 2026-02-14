#!/bin/bash
#SBATCH -A deep_learning
#SBATCH -t 1440                # 24 hours
#SBATCH --job-name=fldd_none
#SBATCH --output=logs/fldd_baseline_none_%j.log
#SBATCH --error=logs/fldd_baseline_none_%j.log

source ~/venv/bin/activate

echo "Starting grid search: Baseline = None (use_baseline=False)"
echo "=================================================="
echo "Running all TC weights with use_baseline=False"
echo "=================================================="

# TC weight values
tc_weights=(10 1 0.1 0.01 0.001 0.0001 0)

# Counter
run_exps=0

for tc in "${tc_weights[@]}"; do
    ((run_exps++))
    
    echo "=== RUNNING combination $run_exps/${#tc_weights[@]}: tc_weight=$tc, use_baseline=False ==="
    echo "Start: $(date)"
    
    python fldd.py --tc_weight $tc --use_baseline False --transport sinkhorn_learnable
    
    if [ $? -eq 0 ]; then
        echo "Status: SUCCESS"
    else
        echo "Status: FAILED (exit code: $?)"
    fi
    
    echo "End: $(date)"
    echo ""
done

echo "=================================================="
echo "JOB 1 COMPLETE: Baseline = None"
echo "=================================================="
echo "Total experiments run: $run_exps"
echo "All TC weights with use_baseline=False completed"
echo "Finished at: $(date)"