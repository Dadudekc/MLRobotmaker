import os
import shutil
import subprocess

def handle_remove_readonly(func, path, exc_info):
    os.chmod(path, 0o700)
    func(path)

def run_command(command, check=True):
    result = subprocess.run(command, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error: Command '{' '.join(command)}' failed with return code {result.returncode}")
        print(f"Output: {result.stdout}")
        print(f"Error Output: {result.stderr}")
        exit(1)
    return result

# Paths
local_repo_path = "C:/DaDudeKC/MLRobot"
new_repo_path = "C:/DaDudeKC/TradingRobotPlug"
new_repo_url = "https://github.com/Dadudekc/TradingRobotPlug.git"

# Remove existing directory if it exists
if os.path.exists(new_repo_path):
    print(f"Removing existing directory: {new_repo_path}")
    shutil.rmtree(new_repo_path, onerror=handle_remove_readonly)

# Clone the new repository
print("Cloning the repository...")
run_command(["git", "clone", new_repo_url, new_repo_path])

# Copy local changes to the new repository
print("Copying local changes...")
for item in os.listdir(local_repo_path):
    s = os.path.join(local_repo_path, item)
    d = os.path.join(new_repo_path, item)
    if os.path.isdir(s):
        shutil.copytree(s, d, dirs_exist_ok=True)
    else:
        shutil.copy2(s, d)

# Change to the new repository directory
os.chdir(new_repo_path)

# Ensure .gitattributes is updated
print("Updating .gitattributes...")
with open('.gitattributes', 'w') as f:
    f.write("""*.dll filter=lfs diff=lfs merge=lfs -text
*.pyd filter=lfs diff=lfs merge=lfs -text
*.lib filter=lfs diff=lfs merge=lfs -text
*.h5 !text !filter !merge !diff
*.pkl !text !filter !merge !diff
*.joblib !text !filter !merge !diff
""")

# Untrack files from LFS
print("Untracking files from LFS...")
run_command(["git", "lfs", "uninstall"])

# Clean LFS cache
print("Removing LFS cache...")
run_command(["git", "lfs", "prune"])

# Stage all changes
print("Staging and committing changes...")
run_command(["git", "add", "."])
run_command(["git", "commit", "-m", "Update repository with latest local changes"])

# Rewrite commit history to remove LFS objects
print("Rewriting commit history to remove LFS objects...")
run_command(["git", "filter-repo", "--strip-blobs-bigger-than", "1M", "--force"])

# Set remote repository URL
print("Setting remote repository URL...")
run_command(["git", "remote", "remove", "origin"], check=False)
run_command(["git", "remote", "add", "origin", new_repo_url])

# Force push to the remote repository
print("Force pushing changes to remote...")
result = run_command(["git", "push", "origin", "main", "--force"], check=False)
if result.returncode != 0:
    print("Error: Force push failed.")
    print(f"Output: {result.stdout}")
    print(f"Error Output: {result.stderr}")

print("Process completed successfully.")
