python plot/plot_progress.py \
--workspace_template_path=workspace_templates/nanogpt \
--workspace_path=workspaces/nanogpt_speedrun_n3 \
--metric=train_time \
--yrescale=1.666666667e-5 \
--ylabel='Train time to target val loss (min)' \
--save_name='nanogpt-speedrun-n3-04022025-train_time.pdf'


python plot/plot_progress.py \
--workspace_template_path=workspace_templates/nanogpt \
--workspace_path=workspaces/nanogpt_speedrun_n3 \
--metric=val_loss \
--ylabel='First validation loss past threshold' \
--ythreshold=3.28 \
--save_name='nanogpt-speedrun-n3-04022025-val_loss.pdf'


python plot/plot_progress.py \
--workspace_template_path=workspace_templates/nanogpt \
--workspace_path=workspaces/nanogpt_speedrun_n3 \
--metric=val_loss \
--ylabel='First validation loss past threshold' \
--ythreshold=3.28 \
--save_name='nanogpt-speedrun-n3-04022025-val_loss.pdf'


# -------- Knowledge conditioned, nanogpt-10-11-2024

python plot/plot_progress.py \
--workspace_template_path=workspace_templates/nanogpt_10112024 \
--workspace_path=workspaces/nanogpt_10112024_aide \
--metric=val_loss \
--ylabel='First validation loss past threshold' \
--ythreshold=3.28

python plot/plot_progress.py \
--workspace_template_path=workspace_templates/nanogpt_10112024 \
--workspace_path=workspaces/nanogpt_10112024_aide \
--metric=train_time \
--yrescale=1.666666667e-5 \
--ylabel='Train time to target val loss (min)' \
--ythreshold=22.3

python plot/plot_progress.py \
--workspace_template_path=workspace_templates/nanogpt_10112024 \
--workspace_path=workspaces/nanogpt_10112024_ks_1 \
--metric=val_loss \
--ylabel='First validation loss past threshold' \
--ythreshold=3.28

python plot/plot_progress.py \
--workspace_template_path=workspace_templates/nanogpt_10112024 \
--workspace_path=workspaces/nanogpt_10112024_ks_1 \
--metric=train_time \
--yrescale=1.666666667e-5 \
--ylabel='Train time to target val loss (min)' \
--ythreshold=22.3

# --------


python plot/plot_progress.py \
--workspace_template_path=workspace_templates/collatz \
--workspace_path=workspaces/archive_speed_up_collatz \
--metric=max_steps \
--ylabel='Max length of Collatz sequence found' \
--save_name='collatz-n3-04022025-max_steps.pdf'


python plot/plot_progress.py \
--workspace_template_path=workspace_templates/collatz \
--workspace_path=workspaces/archive_speed_up_collatz \
--metric=runtime \
--ylabel='Program runtime (s)' \
--save_name='collatz-n3-04022025-runtime.pdf'



