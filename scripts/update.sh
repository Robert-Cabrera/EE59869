#!/bin/bash

# Save the current branch name to return to it later
original_branch=$(git rev-parse --abbrev-ref HEAD)

echo "Fetching all remote branches and updates..."
git fetch --all #

for branch in $(git branch --format='%(refname:short)'); do
    # Check if the branch has an upstream
    upstream=$(git rev-parse --abbrev-ref --symbolic-full-name "$branch"@{upstream} 2>/dev/null)
    if [ -n "$upstream" ]; then
        echo "----------------------------------------------------"
        echo "Updating local branch: $branch from remote tracking branch: $upstream"
        echo "Running command: git checkout \"$branch\""
        git checkout "$branch"
        echo "Running command: git rebase origin/main and pushing changes"
        git rebase origin/main
	git push
    fi
done

echo "----------------------------------------------------"
echo "Returning to the original branch: $original_branch"
git checkout "$original_branch" # Return to the original branch
echo "Script finished."
